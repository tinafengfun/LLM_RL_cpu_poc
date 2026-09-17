"""T06 acceptance: sandbox isolation (unshare -n network block + nobody uid drop)."""

import unittest

from rl_sim.sandbox import run_sandbox_task
from rl_sim.types import Sample, SampleStatus
import pwd

NOBODY_UID = pwd.getpwnam("nobody").pw_uid

OK_CODE = "def add(a, b):\n    return a + b\n"
OK_TESTS = ["assert add(1, 2) == 3"]
NET_CODE = (
    "import socket\n"
    "s = socket.create_connection(('8.8.8.8', 53), timeout=2)\n"
    "s.close()\n"
)


def run(code, tests=OK_TESTS, **kw):
    return run_sandbox_task(Sample(prompt="p", task_id="t"), code, tests, **kw)


class TestIsolation(unittest.TestCase):
    def test_uid_dropped_to_nobody(self):
        s = run(OK_CODE, isolated=True)
        self.assertIs(s.status, SampleStatus.COMPLETED)
        self.assertEqual(s.sandbox.get("uid"), NOBODY_UID)

    def test_network_blocked(self):
        s = run(NET_CODE, isolated=True)
        self.assertIs(s.status, SampleStatus.SANDBOX_ERROR)
        self.assertRegex(s.sandbox.get("detail", ""), r"OSError|Network")

    def test_isolation_off_keeps_root_and_works(self):
        s = run(OK_CODE, isolated=False)
        self.assertIs(s.status, SampleStatus.COMPLETED)
        self.assertEqual(s.sandbox.get("uid"), 0)

    def test_limits_still_apply_under_isolation(self):
        s = run("while True:\n    pass\n", tests=["assert True"],
                isolated=True, overrides={"wall_s": 2, "cpu_s": 60})
        self.assertIs(s.status, SampleStatus.SANDBOX_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
