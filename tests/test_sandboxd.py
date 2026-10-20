"""T25 acceptance: sandboxd HTTP service on a local ephemeral port."""

import json
import threading
import time
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer

from rl_sim.sandboxd import SandboxRegistry, make_handler


def _post(port, path, obj):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class TestSandboxdHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = SandboxRegistry(scratch_root="/tmp/sandboxd_test",
                                       max_concurrent_exec=4,
                                       mem_cap_mb=8192, ttl_s=3600)
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0),
                                      make_handler(cls.registry))
        cls.port = cls.srv.server_address[1]
        cls.thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        import shutil
        shutil.rmtree("/tmp/sandboxd_test", ignore_errors=True)

    def test_healthz(self):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/healthz", timeout=5) as r:
            self.assertEqual(json.loads(r.read()), {"ok": True})

    def test_create_exec_echo_stop(self):
        code, r = _post(self.port, "/sandbox/create", {"profile": "A"})
        self.assertEqual(code, 200)
        sid = r["sandbox_id"]
        code, r = _post(self.port, "/sandbox/exec",
                        {"sandbox_id": sid, "cmd": "echo hello && pwd"})
        self.assertEqual(code, 200)
        self.assertEqual(r["rc"], 0)
        self.assertIn("hello", r["stdout"])
        self.assertIn(sid, r["stdout"])  # cwd is the sandbox workdir
        self.assertGreaterEqual(r["wall_s"], 0)
        self.assertIn("cpu_s", r)
        code, r = _post(self.port, "/sandbox/stop", {"sandbox_id": sid})
        self.assertTrue(r["stopped"])
        import os
        self.assertFalse(os.path.exists(f"/tmp/sandboxd_test/{sid}"))

    def test_stateful_workdir(self):
        _, r = _post(self.port, "/sandbox/create", {"profile": "A"})
        sid = r["sandbox_id"]
        _post(self.port, "/sandbox/exec",
              {"sandbox_id": sid, "cmd": "echo 42 > state.txt"})
        _, r = _post(self.port, "/sandbox/exec",
                     {"sandbox_id": sid, "cmd": "cat state.txt"})
        self.assertIn("42", r["stdout"])
        _post(self.port, "/sandbox/stop", {"sandbox_id": sid})

    def test_network_isolated(self):
        _, r = _post(self.port, "/sandbox/create", {"profile": "A"})
        sid = r["sandbox_id"]
        _, r = _post(self.port, "/sandbox/exec",
                     {"sandbox_id": sid,
                      "cmd": "curl -m 2 -s http://example.com || echo NET_BLOCKED"})
        self.assertIn("NET_BLOCKED", r["stdout"])
        _post(self.port, "/sandbox/stop", {"sandbox_id": sid})

    def test_cpu_limit_kills(self):
        _, r = _post(self.port, "/sandbox/create", {"profile": "A"})  # 5s CPU
        sid = r["sandbox_id"]
        t0 = time.time()
        _, r = _post(self.port, "/sandbox/exec",
                     {"sandbox_id": sid, "cmd": "yes > /dev/null",
                      "timeout_s": 30})
        self.assertLess(time.time() - t0, 25)  # killed by rlimit, not wall
        self.assertNotEqual(r["rc"], 0)
        _post(self.port, "/sandbox/stop", {"sandbox_id": sid})

    def test_unknown_sandbox_404(self):
        code, r = _post(self.port, "/sandbox/exec",
                        {"sandbox_id": "nope", "cmd": "true"})
        self.assertEqual(code, 404)

    def test_stats(self):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/stats", timeout=5) as r:
            s = json.loads(r.read())
        for k in ("active_sandboxes", "live_mem_mb", "created_total",
                  "exec_total", "rejected_total", "running_execs"):
            self.assertIn(k, s)


class TestRegistryLogic(unittest.TestCase):
    def test_memory_admission_cap(self):
        reg = SandboxRegistry("/tmp/sandboxd_test2", mem_cap_mb=600)
        reg.create("A")          # 512MB fits
        with self.assertRaises(Exception):
            reg.create("A")      # 1024MB > 600 cap
        import shutil
        shutil.rmtree("/tmp/sandboxd_test2", ignore_errors=True)

    def test_ttl_reap(self):
        reg = SandboxRegistry("/tmp/sandboxd_test3", ttl_s=0.05)
        e = reg.create("A")
        time.sleep(0.1)
        self.assertEqual(reg.reap_idle(), [e.sandbox_id])
        self.assertEqual(reg.stats()["active_sandboxes"], 0)

    def test_per_sandbox_ttl_overrides_registry_default(self):
        # trajectory-held sandboxes carry a long lease (agent think time);
        # the registry default must not reap them early (T32)
        reg = SandboxRegistry("/tmp/sandboxd_test4", ttl_s=0.05)
        leased = reg.create("A", ttl_s=3600)
        plain = reg.create("A")
        time.sleep(0.1)
        self.assertEqual(reg.reap_idle(), [plain.sandbox_id])
        self.assertEqual(reg.stats()["active_sandboxes"], 1)
        reg.stop(leased.sandbox_id)
        import shutil
        shutil.rmtree("/tmp/sandboxd_test4", ignore_errors=True)

    def test_bad_profile_rejected(self):
        reg = SandboxRegistry("/tmp/sandboxd_test4")
        with self.assertRaises(ValueError):
            reg.create("Z")


if __name__ == "__main__":
    unittest.main()
