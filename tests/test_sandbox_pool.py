"""T05 acceptance: SandboxPool concurrency, queue metrics, arrivals."""

import time
import unittest

from rl_sim.sandbox import SandboxPool
from rl_sim.types import Sample, SampleStatus

OK_CODE = "def add(a, b):\n    return a + b\n"
OK_TESTS = ["assert add(1, 2) == 3"]
SLOW_CODE = "import time\ndef add(a, b):\n    time.sleep(0.3)\n    return a + b\n"


def mk_tasks(n, code=OK_CODE):
    return [(Sample(prompt="p", task_id=f"t{i}"), code, OK_TESTS, "A") for i in range(n)]


class TestSandboxPool(unittest.TestCase):
    def test_burst_concurrency_bounded(self):
        pool = SandboxPool(workers=2)
        try:
            t0 = time.time()
            out = pool.run_batch(mk_tasks(6, SLOW_CODE))
            elapsed = time.time() - t0
            self.assertTrue(all(s.status is SampleStatus.COMPLETED for s in out))
            # 6 x 0.3s on 2 workers >= 0.9s; unbounded would be ~0.4s
            self.assertGreater(elapsed, 0.8)
            m = pool.metrics.snapshot()
            self.assertEqual(m["submitted"], 6)
            self.assertEqual(m["completed"], 6)
            self.assertLessEqual(m["peak_busy"], 2)
            self.assertGreaterEqual(m["peak_queue"], 1)
        finally:
            pool.shutdown()

    def test_order_preserved(self):
        pool = SandboxPool(workers=4)
        try:
            out = pool.run_batch(mk_tasks(10))
            self.assertEqual([s.task_id for s in out], [f"t{i}" for i in range(10)])
        finally:
            pool.shutdown()

    def test_queue_wait_and_exec_wall_measured(self):
        pool = SandboxPool(workers=1)
        try:
            pool.run_batch(mk_tasks(3, SLOW_CODE))
            m = pool.metrics.snapshot()
            self.assertEqual(len(m["queue_waits"]), 3)
            self.assertGreater(max(m["queue_waits"]), 0.2)  # serial queue backlog
            self.assertTrue(all(w >= 0.25 for w in m["exec_walls"]))
        finally:
            pool.shutdown()

    def test_poisson_arrival_count(self):
        pool = SandboxPool(workers=4)
        try:
            out = pool.arrive_poisson(mk_tasks(3), rate_per_s=200.0)
            self.assertEqual(len(out), 3)
            self.assertTrue(all(s.status is SampleStatus.COMPLETED for s in out))
        finally:
            pool.shutdown()

    def test_workers_validation(self):
        with self.assertRaises(ValueError):
            SandboxPool(workers=0)


if __name__ == "__main__":
    unittest.main()
