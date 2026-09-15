"""T01 acceptance: rl_sim/types.py contracts."""

import unittest

from rl_sim.types import Sample, SampleStatus, TaskLine


class TestSampleStatus(unittest.TestCase):
    def test_status_set_covers_sandbox_outcomes(self):
        values = {s.value for s in SampleStatus}
        for expected in (
            "pending", "completed", "truncated", "aborted",
            "gen_error", "sandbox_error", "sandbox_timeout", "sandbox_oom",
        ):
            self.assertIn(expected, values)


class TestTaskLine(unittest.TestCase):
    def test_five_real_lines_plus_mock(self):
        self.assertEqual(
            {t.value for t in TaskLine},
            {"A", "L1", "L2", "L3", "V1", "V2", "MOCK"},
        )


class TestSample(unittest.TestCase):
    def test_defaults(self):
        s = Sample(prompt="p", task_id="t0")
        self.assertEqual(s.status, SampleStatus.PENDING)
        self.assertEqual(s.task_line, "A")
        self.assertEqual(s.reward, 0.0)
        self.assertEqual(s.weight_version, 0)
        self.assertEqual(s.sandbox, {})
        self.assertIsNone(s.train_logprob)

    def test_serialization_roundtrip(self):
        s = Sample(
            prompt="p", task_id="t1", task_line="L1", group_id=3,
            response="resp", reward=0.5, rollout_logprob=-12.3,
            weight_version=7, turns=4,
            sandbox={"queue_wait_s": 0.1, "exec_wall_s": 2.5, "maxrss_kb": 4096},
        )
        s.status = SampleStatus.COMPLETED
        s2 = Sample.from_dict(s.to_dict())
        self.assertEqual(s2, s)
        self.assertEqual(s2.status, SampleStatus.COMPLETED)
        self.assertEqual(s2.sandbox["exec_wall_s"], 2.5)

    def test_to_dict_status_is_string(self):
        d = Sample(prompt="p", task_id="t").to_dict()
        self.assertEqual(d["status"], "pending")


if __name__ == "__main__":
    unittest.main()
