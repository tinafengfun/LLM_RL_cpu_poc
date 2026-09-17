"""T11 acceptance: LocalEngine + LlamaServerManager.

Local tests use a stub llama-server. The real node integration test is guarded
by RL_SIM_NODE=1 and requires the built binary + GGUF on the node.
"""

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rl_sim.engine_local import (
    InstanceSpec, LlamaServerManager, LocalEngine, aggregate_logprobs,
)
from rl_sim.router import Router
from rl_sim.types import Sample, SampleStatus


class _StubLlama(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        assert payload.get("logprobs") is True  # engine must request logprobs
        toks = [{"token": "def", "logprob": -0.1}, {"token": " f", "logprob": -0.4}]
        body = json.dumps({
            "choices": [{"message": {"content": "```python\ndef f():\n    return 1\n```"},
                         "logprobs": {"content": toks}}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class TestBuildCmd(unittest.TestCase):
    def test_cmd_layout(self):
        mgr = LlamaServerManager("/bin/llama-server", [InstanceSpec(
            name="e0", port=9100, cores="0-15", slots=8,
            model="/m/q4.gguf", mmproj="/m/mmproj.gguf", extra=["--ctx-size", "8192"])])
        cmd = mgr.build_cmd(mgr.specs["e0"])
        self.assertEqual(cmd[:3], ["taskset", "-c", "0-15"])
        joined = " ".join(cmd)
        for frag in ("--model /m/q4.gguf", "--port 9100", "--parallel 8",
                     "--jinja", "--mmproj /m/mmproj.gguf", "--ctx-size 8192"):
            self.assertIn(frag, joined)

    def test_no_pinning_when_cores_empty(self):
        mgr = LlamaServerManager("/bin/ls", [InstanceSpec(name="e0", port=1)])
        self.assertNotIn("taskset", mgr.build_cmd(mgr.specs["e0"])[0])


class TestLocalEngineStubbed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), _StubLlama)
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_generate_real_logprob_aggregation(self):
        eng = LocalEngine({"e0": self.url})
        s = eng.generate(Sample(prompt="p", task_id="t", group_id=5))
        self.assertEqual(s.status, SampleStatus.PENDING)
        self.assertIn("def f", s.code)
        self.assertAlmostEqual(s.rollout_logprob, -0.5)
        self.assertEqual(s.num_tokens, 2)

    def test_via_router(self):
        router = Router()
        eng = LocalEngine({"e0": self.url}, router=router)
        s = eng.generate(Sample(prompt="p", task_id="t", group_id=5))
        self.assertEqual(s.num_tokens, 2)

    def test_gen_error_unreachable(self):
        eng = LocalEngine({"e0": "http://127.0.0.1:9"}, timeout_s=0.2)
        s = eng.generate(Sample(prompt="p", task_id="t"))
        self.assertIs(s.status, SampleStatus.GEN_ERROR)

    def test_aggregate_helper(self):
        self.assertEqual(aggregate_logprobs([{"logprob": -1.0}, {"logprob": -2.0}]), (-3.0, 2))


@unittest.skipUnless(os.environ.get("RL_SIM_NODE") == "1", "node-only integration")
class TestNodeLlamaServer(unittest.TestCase):
    """Run on the node: RL_SIM_NODE=1 RL_SIM_MODEL=/path.gguf python -m unittest ..."""

    def test_real_server_roundtrip(self):
        binary = os.environ.get("RL_SIM_BIN", "/mnt/nvme0/models/llama.cpp/build/bin/llama-server")
        model = os.environ["RL_SIM_MODEL"]
        mgr = LlamaServerManager(binary, [InstanceSpec(
            name="e0", port=19310, cores="0-15", slots=4, model=model)])
        mgr.start("e0")
        try:
            self.assertTrue(mgr.wait_healthy(300))
            eng = LocalEngine(mgr.urls())
            s = eng.generate(Sample(
                prompt="Reply with exactly: def f(): return 1", task_id="t"))
            self.assertEqual(s.status, SampleStatus.PENDING)
            self.assertLess(s.rollout_logprob, 0)
            self.assertGreater(s.num_tokens, 0)
        finally:
            mgr.stop_all()


if __name__ == "__main__":
    unittest.main()
