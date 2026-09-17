"""RolloutManager ← slime/ray/rollout.py.

asyncio bounded-queue pipeline (design v3.1 section 5.2):
  prompt groups -> concurrent generate (engine) -> sandbox/reward
  -> GROUP BARRIER (advantage needs the full group) -> pack train batch.

- abort/requeue: once `abort_after_groups` groups completed, cancel the rest;
  aborted samples go to the partial buffer and are re-queued first next round
  (mirrors slime partial rollout / RolloutDataSourceWithBuffer).
- zero-variance groups (all-equal rewards) are dropped DAPO-style and counted.
- L1/L2 repo-task execution path lands in T16; those lines are filtered here.
"""

from __future__ import annotations

import asyncio
import time

from rl_sim.data_source import RolloutDataSource, Task, task_by_id
from rl_sim.reward import compute_reward
from rl_sim.sandbox import SandboxPool
from rl_sim.types import Sample, SampleStatus

# lines this pipeline executes directly (L1/L2 repo tasks: T16 integration)
_SUPPORTED_LINES = {"A", "V1", "V2", "L3", "MOCK"}


class RolloutManager:
    def __init__(self, data_source: RolloutDataSource, engine, sandbox_pool: SandboxPool,
                 monitor=None, gen_concurrency: int = 16, drop_zero_var: bool = True,
                 abort_after_groups: int | None = None, multi_turn_fn=None) -> None:
        self.ds = data_source
        self.engine = engine
        self.sandbox = sandbox_pool
        self.monitor = monitor
        self.gen_concurrency = gen_concurrency
        self.drop_zero_var = drop_zero_var
        self.abort_after_groups = abort_after_groups
        self.multi_turn_fn = multi_turn_fn
        self.partial_buffer: list[Sample] = []  # aborted partial rollouts
        self.stats = {"zero_var_groups": 0, "aborted": 0, "requeued": 0}

    # ------------------------------------------------------------------ async
    async def _generate_one(self, sample: Sample, task: Task, sem: asyncio.Semaphore) -> Sample:
        async with sem:
            await asyncio.to_thread(self.engine.generate, sample, task)
        if sample.status is SampleStatus.GEN_ERROR:
            sample.reward = 0.0
            return sample
        if self.multi_turn_fn is not None:
            await self.multi_turn_fn(sample, task, self.engine)
        if task.tests:
            await asyncio.wrap_future(
                self.sandbox.submit(sample, sample.code, task.tests, tier=task.tier))
        elif sample.status is SampleStatus.PENDING:
            sample.status = SampleStatus.COMPLETED  # direct-answer lines (L3/V*)
        sample.reward = compute_reward(sample, task.reward_spec)
        return sample

    async def _process_group(self, group: list[Sample], sem: asyncio.Semaphore) -> list[Sample]:
        # group barrier: gather waits for the slowest sample (the RL straggler)
        task = task_by_id(group[0].task_id)
        return list(await asyncio.gather(*(self._generate_one(s, task, sem) for s in group)))

    async def generate_async(self, rollout_id: int, batch_size: int) -> dict:
        t0 = time.perf_counter()
        groups = self._drain_buffer() + self.ds.get_groups(batch_size)
        groups = [g for g in groups if g and g[0].task_line in _SUPPORTED_LINES]

        sem = asyncio.Semaphore(self.gen_concurrency)
        group_tasks = [asyncio.ensure_future(self._process_group(g, sem)) for g in groups]

        done_groups: list[list[Sample]] = []
        if self.abort_after_groups is None:
            done_groups = list(await asyncio.gather(*group_tasks))
        else:
            for fut in asyncio.as_completed(group_tasks):
                try:
                    done_groups.append(await fut)
                except (asyncio.CancelledError, Exception):  # noqa: BLE001 - a lost group is logged, not fatal
                    continue
                if len(done_groups) >= self.abort_after_groups:
                    break
            for t in group_tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*group_tasks, return_exceptions=True)

        # requeue unfinished (aborted) groups into the partial buffer
        finished_ids = {s.group_id for g in done_groups for s in g}
        for g in groups:
            if g[0].group_id not in finished_ids:
                for s in g:
                    s.status = SampleStatus.ABORTED
                    self.partial_buffer.append(s)
                self.stats["aborted"] += len(g)
                self.stats["requeued"] += len(g)

        return self._pack(rollout_id, done_groups, time.perf_counter() - t0)

    def generate(self, rollout_id: int, batch_size: int) -> dict:
        """Sync driver entry (mirrors slime's synchronous generate call)."""
        return asyncio.run(self.generate_async(rollout_id, batch_size))

    # ------------------------------------------------------------------ helpers
    def _drain_buffer(self) -> list[list[Sample]]:
        """Requeue aborted groups first (partial rollout buffer, FIFO by group)."""
        if not self.partial_buffer:
            return []
        by_group: dict[int, list[Sample]] = {}
        for s in self.partial_buffer:
            s.status = SampleStatus.PENDING
            by_group.setdefault(s.group_id, []).append(s)
        self.partial_buffer = []
        return list(by_group.values())

    def _pack(self, rollout_id: int, groups: list[list[Sample]], wall_s: float) -> dict:
        samples: list[Sample] = []
        for group in groups:
            rewards = [s.reward for s in group]
            if not rewards:
                continue
            mean_r = sum(rewards) / len(rewards)
            if self.drop_zero_var and len(set(rewards)) == 1:
                self.stats["zero_var_groups"] += 1
                continue
            for s in group:
                s.advantage = s.reward - mean_r  # V3.2 style: mean-only, no std
                samples.append(s)
        return {
            "rollout_id": rollout_id,
            "samples": samples,
            "num_groups": len(groups),
            "wall_s": wall_s,
            "mean_reward": (sum(s.reward for s in samples) / len(samples)) if samples else 0.0,
            "stats": dict(self.stats),
        }

    def eval(self, eval_tasks: list[Task], n: int = 1) -> dict:
        """Held-out pass rate (n samples per eval task)."""
        results = []
        for t in eval_tasks:
            for _ in range(n):
                s = Sample(prompt=t.prompt, task_id=t.task_id, task_line=t.line,
                           image_path=t.image_path)
                s = self.engine.generate(s, t)
                if t.tests:
                    self.sandbox.run_batch([(s, s.code, t.tests, t.tier)])
                elif s.status is SampleStatus.PENDING:
                    s.status = SampleStatus.COMPLETED
                s.reward = compute_reward(s, t.reward_spec)
                results.append(s.reward)
        return {"eval_pass_rate": sum(results) / len(results) if results else 0.0,
                "eval_n": len(results)}
