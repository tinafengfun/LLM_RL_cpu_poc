"""T19 acceptance: run_experiments.py E0/E1/E4 harness."""

import json
import tempfile
import unittest
from pathlib import Path

from run_experiments import e0_baseline, e1_sandbox_sweep, e4_sync_vs_async


class TestE0(unittest.TestCase):
    def test_baseline_capture(self):
        with tempfile.TemporaryDirectory() as d:
            path = e0_baseline(duration_s=0.3, interval_s=0.1, out_dir=Path(d))
            lines = path.read_text().strip().splitlines()
            self.assertGreaterEqual(len(lines), 2)
            rec = json.loads(lines[-1])
            for k in ("ts", "load1", "mem_available_kb"):
                self.assertIn(k, rec)
            self.assertIn("cpu_busy_frac", rec)  # appears from 2nd sample on


class TestE1(unittest.TestCase):
    def test_sweep_rows(self):
        with tempfile.TemporaryDirectory() as d:
            rows = e1_sandbox_sweep([1, 2], 4, Path(d))
            self.assertEqual(len(rows), 2)
            for r in rows:
                for k in ("workers", "wall_s", "throughput_per_s",
                          "queue_wait_p50", "queue_wait_p99", "peak_busy", "peak_queue"):
                    self.assertIn(k, r)
                self.assertLessEqual(r["peak_busy"], r["workers"])
            self.assertTrue((Path(d) / "e1_sandbox_sweep.json").exists())
            # more workers should not be slower on this tiny burst
            self.assertLessEqual(rows[1]["wall_s"], rows[0]["wall_s"] + 1.0)


class TestE4(unittest.TestCase):
    def test_sync_vs_async_runs(self):
        with tempfile.TemporaryDirectory() as d:
            row = e4_sync_vs_async(2, 1, 2, 2, Path(d))
            self.assertIn("sync_wall_s", row)
            self.assertIn("async_wall_s", row)
            self.assertIn("speedup", row)


if __name__ == "__main__":
    unittest.main()
