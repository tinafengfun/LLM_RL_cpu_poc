"""T04 acceptance: rl_sim/sandbox.py core execution."""

import unittest

from rl_sim.sandbox import PROFILES, run_sandbox_task
from rl_sim.types import Sample, SampleStatus

SOLVE_ADD = "def add(a, b):\n    return a + b\n"
TESTS = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(0, 0) == 0"]


def run(code, tests=TESTS, **kw):
    return run_sandbox_task(Sample(prompt="p", task_id="t"), code, tests, **kw)


class TestSandboxCore(unittest.TestCase):
    def test_passed(self):
        s = run(SOLVE_ADD)
        self.assertIs(s.status, SampleStatus.COMPLETED)
        self.assertEqual((s.tests_passed, s.tests_total), (3, 3))
        # metrics contract
        for k in ("cpu_s", "maxrss_kb", "launch_overhead_s", "exec_wall_s",
                  "submit_ts", "spawn_ts", "end_ts", "tier"):
            self.assertIn(k, s.sandbox, k)
        self.assertGreater(s.sandbox["maxrss_kb"], 0)
        self.assertGreaterEqual(s.sandbox["exec_wall_s"], 0.0)

    def test_failed_partial(self):
        s = run("def add(a, b):\n    return a - b\n")
        self.assertIs(s.status, SampleStatus.COMPLETED)
        self.assertEqual((s.tests_passed, s.tests_total), (1, 3))  # only add(0,0)
        self.assertIn("AssertionError", s.sandbox["detail"])

    def test_syntax_error(self):
        s = run("def add(a, b)\n    return a + b\n")
        self.assertIs(s.status, SampleStatus.SANDBOX_ERROR)
        self.assertEqual(s.tests_passed, 0)

    def test_wall_timeout(self):
        s = run("while True:\n    pass\n", tests=["assert True"],
                overrides={"wall_s": 2, "cpu_s": 60})
        self.assertIs(s.status, SampleStatus.SANDBOX_TIMEOUT)
        self.assertLess(s.sandbox["exec_wall_s"], 10)

    def test_cpu_limit_sigxcpu(self):
        s = run("while True:\n    pass\n", tests=["assert True"],
                overrides={"wall_s": 30, "cpu_s": 1})
        self.assertIs(s.status, SampleStatus.SANDBOX_TIMEOUT)
        self.assertIn("CPU limit", s.sandbox["detail"])

    def test_oom(self):
        code = "x = []\nfor _ in range(10**8):\n    x.append(' ' * 10**6)\n"
        s = run(code, tests=["assert True"], overrides={"mem_mb": 256, "wall_s": 20})
        self.assertIs(s.status, SampleStatus.SANDBOX_OOM)

    def test_profiles_exist(self):
        for tier in ("A", "B", "C", "MOCK"):
            p = PROFILES[tier]
            self.assertIn("cpu_s", p)
            self.assertIn("mem_mb", p)
            self.assertIn("wall_s", p)
        self.assertGreater(PROFILES["B"]["mem_mb"], PROFILES["A"]["mem_mb"])


if __name__ == "__main__":
    unittest.main()
