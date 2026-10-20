"""T29-A acceptance: sandboxd /trajectory endpoints driving a real
trajectory_worker subprocess on localhost (fake adapter, fixture git repo)."""

import json
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rl_sim.sandboxd import SandboxRegistry, make_handler


def _git(dir_, *args):
    subprocess.run(["git", *args], cwd=dir_, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                        "PATH": "/usr/bin:/bin:/usr/local/bin"})


class FakeAdapterHandler(BaseHTTPRequestHandler):
    """Returns submit immediately -> agent loop ends at turn 0."""

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        json.loads(self.rfile.read(n) or b"{}")
        body = json.dumps({
            "text": "", "finish_reason": "stop",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "submit", "arguments": "{}"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def _post(port, path, obj):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


class TestTrajectoryEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # fixture git mirror with a test that fails without a patch
        src = tempfile.mkdtemp(prefix="traj_src_")
        with open(f"{src}/calc.py", "w") as f:
            f.write("def add(a, b):\n    return a - b  # BUG\n")
        with open(f"{src}/test_calc.py", "w") as f:
            f.write("from calc import add\n\n"
                    "def test_add():\n    assert add(2, 3) == 5\n")
        _git(src, "init", "-q")
        _git(src, "add", "-A")
        _git(src, "commit", "-qm", "init")
        cls.commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src,
                                    capture_output=True, text=True).stdout.strip()
        cls.mirror = tempfile.mkdtemp(prefix="traj_mirror_")
        _git(src, "clone", "-q", "--bare", src, cls.mirror)
        subprocess.run(["chmod", "-R", "a+rX", cls.mirror], check=True)

        cls.adapter_srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeAdapterHandler)
        threading.Thread(target=cls.adapter_srv.serve_forever, daemon=True).start()
        cls.adapter_port = cls.adapter_srv.server_address[1]

        cls.registry = SandboxRegistry("/tmp/traj_test_scratch",
                                       mem_cap_mb=8192, ttl_s=3600)
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.registry))
        cls.port = cls.srv.server_address[1]
        # workers spawned via /trajectory/start must reach THIS server
        cls.registry.self_url = f"http://127.0.0.1:{cls.port}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.adapter_srv.shutdown()
        shutil.rmtree(cls.mirror, ignore_errors=True)
        shutil.rmtree("/tmp/traj_test_scratch", ignore_errors=True)

    def test_trajectory_lifecycle(self):
        swe = {"instance_id": "fix__repo-1", "repo": "fix/repo",
               "base_commit": self.commit, "problem_statement": "fix add",
               "test_patch": "", "fail_to_pass": ["test_calc.py::test_add"],
               "pass_to_pass": []}
        r = _post(self.port, "/trajectory/start", {
            "traj_id": "traj-t1", "swe": swe,
            "adapter_url": f"http://127.0.0.1:{self.adapter_port}",
            "git_dir": self.mirror, "venv": "",
            "max_turns": 3, "agent_timeout_s": 60, "grade_timeout_s": 60})
        self.assertEqual(r["state"], "running")

        deadline = time.time() + 120
        st = {"state": "running"}
        while time.time() < deadline:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/trajectory/traj-t1/status",
                    timeout=10) as resp:
                st = json.loads(resp.read())
            if st["state"] != "running":
                break
            time.sleep(0.5)
        self.assertEqual(st["state"], "done",
                         str(st.get("log_tail", ""))[-500:])
        res = st["result"]
        self.assertEqual(res["status"], "submitted")
        self.assertEqual(res["turns"], 1)
        # empty patch -> f2p test fails -> reward 0 (graded for real)
        self.assertEqual(res["reward"], 0.0)
        self.assertIn("grade", res)

    def test_abort_unknown_and_status_404(self):
        r = _post(self.port, "/trajectory/abort", {"traj_id": "nope"})
        self.assertFalse(r["aborted"])
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/trajectory/nope/status", timeout=5)
            self.fail("expected 404")
        except Exception as e:
            self.assertIn("404", str(e))

    def test_duplicate_start_is_idempotent(self):
        # driver retries start after a connection reset; sandboxd must not
        # spawn a second worker for the same traj_id (T32)
        swe = {"instance_id": "fix__repo-1", "repo": "fix/repo",
               "base_commit": self.commit, "problem_statement": "fix add",
               "test_patch": "", "fail_to_pass": ["test_calc.py::test_add"],
               "pass_to_pass": []}
        body = {"traj_id": "traj-dupe", "swe": swe,
                "adapter_url": f"http://127.0.0.1:{self.adapter_port}",
                "git_dir": self.mirror, "venv": "",
                "max_turns": 3, "agent_timeout_s": 60, "grade_timeout_s": 60}
        r1 = _post(self.port, "/trajectory/start", body)
        self.assertEqual(r1["state"], "running")
        proc1 = self.registry.trajectories["traj-dupe"]["proc"]
        r2 = _post(self.port, "/trajectory/start", body)
        self.assertEqual(r2["state"], "running")
        self.assertIs(self.registry.trajectories["traj-dupe"]["proc"], proc1)
        _post(self.port, "/trajectory/abort", {"traj_id": "traj-dupe"})


if __name__ == "__main__":
    unittest.main()
