"""T26 acceptance: adapter service (fake router, no real engine)."""

import json
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from rl_sim.adapter import Adapter, make_handler


FAKE_RESPONSE = {
    "choices": [{
        "message": {"role": "assistant", "content": "hello world",
                    "tool_calls": [{"id": "c1", "type": "function",
                                    "function": {"name": "bash",
                                                 "arguments": "{\"cmd\": \"ls\"}"}}]},
        "finish_reason": "stop",
        "logprobs": {"content": [{"token": "hello", "logprob": -0.1},
                                 {"token": " world", "logprob": -0.2}]},
    }],
    "usage": {"prompt_tokens": 11, "completion_tokens": 2},
    "_backend": "eng0",
}


class FakeRing:
    counts = {"eng0": 1}


class FakeRouter:
    def __init__(self):
        self.ring = FakeRing()
        self.last_payload = None
        self.last_key = None
        self.fail = False

    def forward(self, path, payload, session_key):
        self.last_payload, self.last_key = payload, session_key
        if self.fail:
            raise RuntimeError("all backends failed")
        assert path == "/v1/chat/completions"
        return dict(FAKE_RESPONSE)


class TestAdapterHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="adapter_test_")
        cls.router = FakeRouter()
        cls.adapter = Adapter(cls.router, log_dir=cls.tmp)
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.adapter))
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _post(self, path, obj):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=json.dumps(obj).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_generate_happy_path(self):
        code, r = self._post("/generate", {
            "traj_id": "traj-1", "turn": 0,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function"}],
        })
        self.assertEqual(code, 200)
        self.assertEqual(r["text"], "hello world")
        self.assertEqual(r["tool_calls"][0]["function"]["name"], "bash")
        self.assertEqual(r["backend"], "eng0")
        self.assertEqual(r["usage"]["prompt_tokens"], 11)
        # router saw affinity key = traj_id and logprob request
        self.assertEqual(self.router.last_key, "traj-1")
        self.assertTrue(self.router.last_payload["logprobs"])
        self.assertTrue(self.router.last_payload["tools"])
        # thinking disabled (Qwen3 quirk from engine_local)
        self.assertEqual(self.router.last_payload["chat_template_kwargs"],
                         {"enable_thinking": False})

    def test_logprob_retention(self):
        self._post("/generate", {"traj_id": "traj-2", "turn": 3,
                                 "messages": [{"role": "user", "content": "x"}]})
        with open(f"{self.tmp}/traj-2.jsonl", encoding="utf-8") as f:
            rec = json.loads(f.readline())
        self.assertEqual(rec["traj_id"], "traj-2")
        self.assertEqual(rec["turn"], 3)
        self.assertEqual(rec["out_tokens"],
                         [{"t": "hello", "lp": -0.1}, {"t": " world", "lp": -0.2}])
        self.assertEqual(rec["prompt_tokens"], 11)

    def test_missing_traj_id_400(self):
        code, _ = self._post("/generate", {"messages": []})
        self.assertEqual(code, 400)

    def test_engine_failure_502(self):
        self.router.fail = True
        try:
            code, r = self._post("/generate", {"traj_id": "t", "messages": []})
            self.assertEqual(code, 502)
            self.assertIn("error", r)
        finally:
            self.router.fail = False

    def test_stats(self):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/stats", timeout=5) as r:
            s = json.loads(r.read())
        self.assertGreaterEqual(s["calls"], 2)
        self.assertGreaterEqual(s["tokens_out"], 4)
        self.assertEqual(s["backends"], {"eng0": 1})


if __name__ == "__main__":
    unittest.main()
