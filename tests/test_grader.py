"""T28 acceptance: grader on a real git fixture repo through the real
sandbox isolation path (in-process SandboxRegistry, no HTTP)."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest

from rl_sim.grader import grade
from rl_sim.sandboxd import SandboxRegistry, exec_in_sandbox


class LocalSandboxClient:
    """In-process create/exec/stop over SandboxRegistry (test double for the
    HTTP client; same isolation path)."""

    def __init__(self, registry):
        self.registry = registry

    def create(self, profile="B", sandbox_id=None):
        e = self.registry.create(profile, sandbox_id)
        return {"sandbox_id": e.sandbox_id, "workdir": e.workdir}

    def exec(self, sandbox_id, cmd, timeout_s=None, env=None):
        return exec_in_sandbox(self.registry, self.registry.get(sandbox_id),
                               cmd, timeout_s=timeout_s, env_extra=env)

    def stop(self, sandbox_id):
        return self.registry.stop(sandbox_id)


def _git(dir_, *args):
    subprocess.run(["git", *args], cwd=dir_, check=True,
                   capture_output=True, env={
                       "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                       "PATH": "/usr/bin:/bin:/usr/local/bin"})


class TestGrader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # fixture repo: buggy add(); test_patch adds the FAIL_TO_PASS test
        cls.mirror = tempfile.mkdtemp(prefix="grader_mirror_")
        src = tempfile.mkdtemp(prefix="grader_src_")
        with open(f"{src}/calc.py", "w") as f:
            f.write("def add(a, b):\n    return a - b  # BUG\n")
        with open(f"{src}/test_calc.py", "w") as f:
            f.write("from calc import add\n\n"
                    "def test_existing():\n    assert add(2, 0) == 2\n")
        _git(src, "init", "-q")
        _git(src, "add", "-A")
        _git(src, "commit", "-qm", "init")
        cls.commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=src,
            capture_output=True, text=True).stdout.strip()
        _git(src, "clone", "-q", "--bare", src, cls.mirror)
        # sandbox exec drops to nobody: make the mirror world-readable
        subprocess.run(["chmod", "-R", "a+rX", cls.mirror], check=True)
        cls.test_patch = (
            "diff --git a/test_calc.py b/test_calc.py\n"
            "--- a/test_calc.py\n"
            "+++ b/test_calc.py\n"
            "@@ -3,2 +3,5 @@\n"
            " def test_existing():\n"
            "     assert add(2, 0) == 2\n"
            "+\n"
            "+def test_add_fixed():\n"
            "+    assert add(2, 3) == 5\n")
        cls.swe = {"instance_id": "fix__repo-1", "repo": "fix/repo",
                   "base_commit": cls.commit, "test_patch": cls.test_patch,
                   "fail_to_pass": ["test_calc.py::test_add_fixed"],
                   "pass_to_pass": ["test_calc.py::test_existing"]}
        cls.registry = SandboxRegistry("/tmp/grader_test_scratch",
                                       mem_cap_mb=8192, ttl_s=3600)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.mirror, ignore_errors=True)
        shutil.rmtree("/tmp/grader_test_scratch", ignore_errors=True)

    def setUp(self):
        self.sandbox = LocalSandboxClient(self.registry)

    def test_gold_fix_passes(self):
        patch = ("diff --git a/calc.py b/calc.py\n"
                 "--- a/calc.py\n"
                 "+++ b/calc.py\n"
                 "@@ -1,2 +1,2 @@\n"
                 " def add(a, b):\n"
                 "-    return a - b  # BUG\n"
                 "+    return a + b\n")
        r = grade(self.swe, patch, self.sandbox, self.mirror, sys.executable)
        self.assertEqual(r["reward"], 1.0, r["log_tail"] + r["detail"])
        self.assertEqual(r["n_nodes"], 2)
        self.assertGreater(r["wall_s"], 0)

    def test_no_patch_fails(self):
        r = grade(self.swe, "", self.sandbox, self.mirror, sys.executable)
        self.assertEqual(r["reward"], 0.0)

    def test_bad_patch_fails_cleanly(self):
        r = grade(self.swe, "diff --git garbage\n@@ not a patch",
                  self.sandbox, self.mirror, sys.executable)
        self.assertEqual(r["reward"], 0.0)
        self.assertIn("apply failed", r["detail"])

    def test_wrong_fix_fails(self):
        patch = ("diff --git a/calc.py b/calc.py\n"
                 "--- a/calc.py\n"
                 "+++ b/calc.py\n"
                 "@@ -1,2 +1,2 @@\n"
                 " def add(a, b):\n"
                 "-    return a - b  # BUG\n"
                 "+    return a * b\n")
        r = grade(self.swe, patch, self.sandbox, self.mirror, sys.executable)
        self.assertEqual(r["reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
