"""T13 acceptance: rollout_manager asyncio pipeline (barrier/abort/advantage/pack)."""

import unittest

from rl_sim.data_source import ALL_TASKS, A_TASKS, RolloutDataSource, task_by_id
from rl_sim.engine import Engine, mutate
from rl_sim.rollout_manager import RolloutManager
from rl_sim.sandbox import SandboxPool
from rl_sim.types import Sample
import random


class StubEngine(Engine):
    """Deterministic: alternates correct/mutated code per call."""

    def __init__(self, correct_ratio_mod: int = 2):
        super().__init__()
        self.calls = 0
        self.mod = correct_ratio_mod
        self._rng = random.Random(0)

    def generate(self, sample, task=None):
        task = task or task_by_id(sample.task_id)
        self.calls += 1
        if self.calls % self.mod == 0:
            sample.code = task.solution
        else:
            sample.code = mutate(task.solution, self._rng)
        sample.response = f"```python\n{sample.code}```"
        sample.num_tokens = len(sample.response.split())
        sample.rollout_logprob = -0.5 * sample.num_tokens
        sample.weight_version = self._weight_version
        return sample


def make_rm(engine, n=4, workers=2, **kw):
    ds = RolloutDataSource(A_TASKS[:4], n_samples_per_prompt=n, seed=0)
    pool = SandboxPool(workers=workers)
    return RolloutManager(ds, engine, pool, **kw), pool


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.rm, self.pool = make_rm(StubEngine())

    def tearDown(self):
        self.pool.shutdown()

    def test_pack_contract(self):
        data = self.rm.generate(0, 2)
        for k in ("rollout_id", "samples", "num_groups", "wall_s", "mean_reward", "stats"):
            self.assertIn(k, data)
        for s in data["samples"]:
            self.assertIsNotNone(s.status)
            self.assertIn(s.advantage + 1e-9, [s.reward - (sum(g2.reward for g2 in data["samples"] if g2.group_id == s.group_id) / len([g3 for g3 in data["samples"] if g3.group_id == s.group_id])) + 1e-9 for _ in [0]])

    def test_group_advantage_mean_zero(self):
        data = self.rm.generate(0, 2)
        by_group = {}
        for s in data["samples"]:
            by_group.setdefault(s.group_id, []).append(s.advantage)
        for gid, advs in by_group.items():
            self.assertAlmostEqual(sum(advs), 0.0, places=6, msg=f"group {gid}")


class TestZeroVariance(unittest.TestCase):
    def test_all_correct_group_dropped(self):
        rm, pool = make_rm(StubEngine(correct_ratio_mod=1))  # always correct
        try:
            data = rm.generate(0, 1)
            self.assertEqual(data["samples"], [])
            self.assertGreaterEqual(data["stats"]["zero_var_groups"], 1)
        finally:
            pool.shutdown()


class TestAbortRequeue(unittest.TestCase):
    def test_abort_fills_buffer_and_next_round_drains(self):
        rm, pool = make_rm(StubEngine(), abort_after_groups=1)
        try:
            data = rm.generate(0, 3)
            self.assertEqual(len(data["samples"]) <= 4, True)
            self.assertGreater(data["stats"]["requeued"], 0)
            # second round: disable abort so drained groups complete for good
            rm.abort_after_groups = None
            data2 = rm.generate(1, 1)
            self.assertGreater(data2["stats"]["aborted"], 0)
            self.assertEqual(len(rm.partial_buffer), 0)  # drained and completed
            # the two requeued groups + 1 new = 3 groups processed
            self.assertEqual(data2["num_groups"], 3)
        finally:
            pool.shutdown()


class TestEval(unittest.TestCase):
    def test_eval_pass_rate(self):
        rm, pool = make_rm(StubEngine(correct_ratio_mod=1))
        try:
            out = rm.eval([task_by_id("a_second_max")], n=2)
            self.assertEqual(out["eval_n"], 2)
            self.assertEqual(out["eval_pass_rate"], 1.0)
        finally:
            pool.shutdown()


if __name__ == "__main__":
    unittest.main()
