"""T09 acceptance: engine protocol, MockEngine, APIEngine (stub server)."""

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rl_sim.data_source import A_TASKS, task_by_id
from rl_sim.engine import APIEngine, MockEngine, extract_code, mutate
from rl_sim.types import Sample, SampleStatus

TASKS = {t.task_id: t for t in A_TASKS}


class TestMockEngine(unittest.TestCase):
    def test_fills_contract_fields(self):
        eng = MockEngine(TASKS, seed=1)
        s = eng.generate(Sample(prompt="p", task_id="a_reverse_string"))
        self.assertTrue(s.response.startswith("```python"))
        self.assertIn("def reverse_string", s.code)
        self.assertGreater(s.num_tokens, 0)
        self.assertLess(s.rollout_logprob, 0)
        self.assertEqual(s.weight_version, 0)
        self.assertEqual(s.turns, 1)

    def test_correctness_rises_with_version(self):
        eng = MockEngine(TASKS, seed=42)
        task = task_by_id("a_reverse_string")
        n = 300

        def hit_rate(v):
            eng.update_weights(v)
            hits = 0
            for _ in range(n):
                s = eng.generate(Sample(prompt="p", task_id=task.task_id))
                if "return s[::-1]" in s.code:
                    hits += 1
            return hits / n

        self.assertLess(hit_rate(0), hit_rate(9))

    def test_pause_blocks_generation(self):
        eng = MockEngine(TASKS, latency_s=0.01)
        eng.pause_generation()
        done = []
        t = threading.Thread(target=lambda: done.append(
            eng.generate(Sample(prompt="p", task_id="a_reverse_string")).task_id))
        t.start()
        time.sleep(0.05)
        self.assertEqual(done, [])  # blocked while paused
        eng.continue_generation()
        t.join(timeout=5)
        self.assertEqual(done, ["a_reverse_string"])

    def test_mutate_changes_code(self):
        import random
        rng = random.Random(0)
        sol = task_by_id("a_fizzbuzz").solution
        self.assertNotEqual(mutate(sol, rng), sol)


class TestExtractCode(unittest.TestCase):
    def test_block_and_raw(self):
        self.assertEqual(extract_code("here:\n```python\nx = 1\n```"), "x = 1")
        self.assertEqual(extract_code("x = 2"), "x = 2")


class _StubHandler(BaseHTTPRequestHandler):
    calls = {"n": 0}

    def do_POST(self):
        _StubHandler.calls["n"] += 1
        if _StubHandler.calls["n"] == 1:
            self.send_response(500)  # force one retry
            self.end_headers()
            return
        body = json.dumps({"choices": [{"message": {"content": "```python\nx = 1\n```"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class TestAPIEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_generate_with_retry(self):
        _StubHandler.calls["n"] = 0
        eng = APIEngine(f"http://127.0.0.1:{self.port}/v1", model="m", max_retries=3)
        eng.latency_floor_s = 0
        s = eng.generate(Sample(prompt="p", task_id="t"))
        self.assertEqual(s.status, SampleStatus.PENDING)
        self.assertEqual(s.code, "x = 1")
        self.assertGreaterEqual(_StubHandler.calls["n"], 2)  # retried after 500

    def test_gen_error_when_unreachable(self):
        eng = APIEngine("http://127.0.0.1:9/v1", model="m", timeout_s=0.2, max_retries=1)
        s = eng.generate(Sample(prompt="p", task_id="t"))
        self.assertIs(s.status, SampleStatus.GEN_ERROR)
        self.assertIn("error", s.metadata)


if __name__ == "__main__":
    unittest.main()
