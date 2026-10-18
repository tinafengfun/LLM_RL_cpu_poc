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
import json
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

from rl_sim.data_source import RolloutDataSource, Task, task_by_id
from rl_sim.reward import compute_reward
from rl_sim.sandbox import SandboxPool
from rl_sim.types import Sample, SampleStatus

# lines this pipeline executes directly (L1/L2 repo tasks: T16 integration)
_SUPPORTED_LINES = {"A", "V1", "V2", "L3", "MOCK"}


@dataclass
class RemoteSWEConfig:
    """node2 trajectory service binding (design 09 section 4/5.2)."""
    sandboxd_url: str
    adapter_url: str          # advertised to node2 agent_runners (node1 address!)
    git_dir: str              # on node2, e.g. /workspace/swebench/repo/sympy.git
    venv: str = ""            # on node2, e.g. /workspace/swebench/envs/py311
    max_turns: int = 40
    agent_timeout_s: float = 1800.0
    grade_timeout_s: float = 600.0
    poll_interval_s: float = 2.0
    http_timeout_s: float = 30.0
    adapter_log_dir: str = "adapter_log"  # node1-local; token/logprob source


def _post_json(url: str, obj: dict, timeout_s: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read())


def _get_json(url: str, timeout_s: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        return json.loads(r.read())


def reward_spec_of(task: Task) -> dict:
    """Task.reward_spec carries only the type; merge the answer field in."""
    spec = dict(task.reward_spec)
    if task.answer is not None:
        spec["answer"] = task.answer
    return spec


class RolloutManager:
    def __init__(self, data_source: RolloutDataSource, engine, sandbox_pool: SandboxPool,
                 monitor=None, gen_concurrency: int = 16, drop_zero_var: bool = True,
                 abort_after_groups: int | None = None, multi_turn_fn=None,
                 remote_swe: RemoteSWEConfig | None = None) -> None:
        self.ds = data_source
        self.engine = engine
        self.sandbox = sandbox_pool
        self.monitor = monitor
        self.gen_concurrency = gen_concurrency
        self.drop_zero_var = drop_zero_var
        self.abort_after_groups = abort_after_groups
        self.multi_turn_fn = multi_turn_fn
        self.remote_swe = remote_swe
        self.partial_buffer: list[Sample] = []  # aborted partial rollouts
        self.stats = {"zero_var_groups": 0, "aborted": 0, "requeued": 0}
        # live counters for observability/demo (read by sampler threads; GIL-safe)
        self.live: dict = {"gen_waiting": 0, "scoring": 0, "done": 0,
                           "total": 0, "groups_done": 0, "groups_total": 0,
                           "group_progress": {}}

    # ------------------------------------------------------------------ async
    async def _generate_one(self, sample: Sample, task: Task, sem: asyncio.Semaphore) -> Sample:
        self.live["gen_waiting"] += 1
        acquired = False
        try:
            async with sem:
                acquired = True
                self.live["gen_waiting"] -= 1
                await asyncio.to_thread(self.engine.generate, sample, task)
            if sample.status is SampleStatus.GEN_ERROR:
                sample.reward = 0.0
                return sample
            if self.multi_turn_fn is not None:
                await self.multi_turn_fn(sample, task, self.engine)
            self.live["scoring"] += 1
            if task.tests:
                await asyncio.wrap_future(
                    self.sandbox.submit(sample, sample.code, task.tests, tier=task.tier))
            elif sample.status is SampleStatus.PENDING:
                sample.status = SampleStatus.COMPLETED  # direct-answer lines (L3/V*)
            sample.reward = compute_reward(sample, reward_spec_of(task))
            return sample
        finally:
            if not acquired:
                self.live["gen_waiting"] -= 1
            self.live["scoring"] = max(0, self.live["scoring"] - 1)
            self.live["done"] += 1
            gid = sample.group_id
            if gid in self.live["group_progress"]:
                self.live["group_progress"][gid][0] += 1

    async def _process_group(self, group: list[Sample], sem: asyncio.Semaphore) -> list[Sample]:
        # group barrier: gather waits for the slowest sample (the RL straggler)
        if group[0].task_line == "SWE" and self.remote_swe is not None:
            out = list(await asyncio.gather(
                *(self._run_swe_one(s) for s in group)))
            self.live["groups_done"] += 1
            return out
        task = task_by_id(group[0].task_id)
        out = list(await asyncio.gather(*(self._generate_one(s, task, sem) for s in group)))
        self.live["groups_done"] += 1
        return out

    # ------------------------------------------------------------------ remote SWE
    async def _run_swe_one(self, sample: Sample) -> Sample:
        """Dispatch one trajectory to node2 sandboxd and poll it (design 09 §4).

        node1 does NO generation here: the agent loop lives on node2 and calls
        back to the node1 adapter per turn. Token/logprob data lands in the
        node1-local adapter log; reward comes from the node2 grader.
        NOTE: abort restarts the trajectory from scratch on requeue (no
        mid-trajectory state preservation — DSec pause/resume is out of scope).
        """
        cfg = self.remote_swe
        assert cfg is not None
        traj_id = f"{sample.task_id}-g{sample.group_id}-{uuid.uuid4().hex[:8]}"
        sample.metadata["traj_id"] = traj_id
        body = {"traj_id": traj_id, "swe": sample.metadata["swe"],
                "adapter_url": cfg.adapter_url, "git_dir": cfg.git_dir,
                "venv": cfg.venv, "max_turns": cfg.max_turns,
                "agent_timeout_s": cfg.agent_timeout_s,
                "grade_timeout_s": cfg.grade_timeout_s}
        self.live["gen_waiting"] += 1
        try:
            await asyncio.to_thread(
                _post_json, cfg.sandboxd_url + "/trajectory/start", body,
                cfg.http_timeout_s)
            while True:
                await asyncio.sleep(cfg.poll_interval_s)
                st = await asyncio.to_thread(
                    _get_json,
                    f"{cfg.sandboxd_url}/trajectory/{traj_id}/status",
                    cfg.http_timeout_s)
                if st["state"] != "running":
                    break
            if st["state"] == "done":
                r = st["result"]
                sample.reward = float(r.get("reward", 0.0))
                sample.turns = int(r.get("turns", 0))
                sample.status = SampleStatus.COMPLETED
                sample.sandbox.update({
                    "exec_wall_s": r.get("exec_wall_total_s"),
                    "gen_lat_s": r.get("gen_lat_total_s"),
                    "traj_status": r.get("status"),
                    "grade": r.get("grade", {}),
                })
                self._fill_logprob(sample, traj_id)
            else:
                sample.status = SampleStatus.SANDBOX_ERROR
                sample.reward = 0.0
                sample.sandbox["detail"] = f"trajectory {st['state']}"
            return sample
        except asyncio.CancelledError:
            try:  # abort the remote trajectory; group accounting continues
                await asyncio.to_thread(
                    _post_json, cfg.sandboxd_url + "/trajectory/abort",
                    {"traj_id": traj_id}, cfg.http_timeout_s)
            except Exception:  # noqa: BLE001 - best effort abort
                pass
            raise
        finally:
            self.live["gen_waiting"] -= 1
            self.live["done"] += 1
            gid = sample.group_id
            if gid in self.live["group_progress"]:
                self.live["group_progress"][gid][0] += 1

    def _fill_logprob(self, sample: Sample, traj_id: str) -> None:
        """Sum per-token logprobs the adapter retained for this trajectory."""
        cfg = self.remote_swe
        p = Path(cfg.adapter_log_dir) / f"{traj_id}.jsonl"
        total_lp, ntok = 0.0, 0
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    for t in rec.get("out_tokens", []):
                        total_lp += t.get("lp", 0.0)
                        ntok += 1
        except OSError:
            pass
        sample.rollout_logprob = total_lp
        sample.num_tokens = ntok

    async def generate_async(self, rollout_id: int, batch_size: int) -> dict:
        t0 = time.perf_counter()
        groups = self._drain_buffer() + self.ds.get_groups(batch_size)
        supported = set(_SUPPORTED_LINES)
        if self.remote_swe is not None:
            supported.add("SWE")
        groups = [g for g in groups if g and g[0].task_line in supported]
        self.live.update(total=sum(len(g) for g in groups), done=0,
                         groups_done=0, groups_total=len(groups),
                         group_progress={g[0].group_id: [0, len(g)] for g in groups})

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
        all_rewards: list[float] = []
        for group in groups:
            rewards = [s.reward for s in group]
            if not rewards:
                continue
            all_rewards.extend(rewards)
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
            "gen_errors": sum(1 for g in groups for s in g if s.status is SampleStatus.GEN_ERROR),
            # pre-filter mean over ALL completed groups (post-filter sample count can be 0)
            "mean_reward": (sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0,
            "kept_samples": len(samples),
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
                s.reward = compute_reward(s, reward_spec_of(t))
                results.append(s.reward)
        return {"eval_pass_rate": sum(results) / len(results) if results else 0.0,
                "eval_n": len(results)}
