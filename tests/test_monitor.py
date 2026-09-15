"""T02 acceptance: rl_sim/monitor.py contracts."""

import time
import unittest

from rl_sim.monitor import StageTimer, fmt_pct, fmt_stage_lines, percentiles


class TestStageTimer(unittest.TestCase):
    def test_records_wall_and_cpu(self):
        t = StageTimer()
        with t.stage("sleep"):
            time.sleep(0.02)
        snap = t.snapshot()
        self.assertIn("sleep", snap)
        wall, cpu = snap["sleep"]
        self.assertGreaterEqual(wall, 0.02)
        # sleeping burns ~no CPU: cpu must be much smaller than wall
        self.assertLess(cpu, wall)

    def test_accumulates_and_reset(self):
        t = StageTimer()
        with t.stage("a"):
            pass
        with t.stage("a"):
            pass
        self.assertEqual(len(t.snapshot()), 1)
        t.reset()
        self.assertEqual(t.snapshot(), {})

    def test_record_direct(self):
        t = StageTimer()
        t.record("x", 1.5, 0.5)
        self.assertEqual(t.snapshot()["x"], (1.5, 0.5))


class TestPercentiles(unittest.TestCase):
    def test_known_list(self):
        p = percentiles(list(range(1, 101)), ps=(50, 99))
        self.assertEqual(p[50], 50)
        self.assertEqual(p[99], 99)

    def test_empty(self):
        self.assertEqual(percentiles([]), {50: 0.0, 95: 0.0, 99: 0.0})

    def test_single(self):
        self.assertEqual(percentiles([3.0])[99], 3.0)


class TestFormat(unittest.TestCase):
    def test_stage_lines_dual(self):
        out = fmt_stage_lines({"infer": (38.3, 36.1), "sandbox": (11.8, 9.2)})
        self.assertIn("wall: infer 38.3s | sandbox 11.8s", out)
        self.assertIn("cpu : infer 36.1s | sandbox 9.2s", out)

    def test_empty(self):
        self.assertIn("no stages", fmt_stage_lines({}))

    def test_fmt_pct(self):
        self.assertEqual(fmt_pct([0.08] * 50 + [0.9] * 50), "p50 0.08s p95 0.90s p99 0.90s")


if __name__ == "__main__":
    unittest.main()
