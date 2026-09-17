"""T12 acceptance: tokenizer server-client + fallback."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rl_sim.tokenizer import Tokenizer


class _StubTok(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/tokenize":
            body = json.dumps({"tokens": list(range(len(payload["content"].split())))}).encode()
        else:  # /detokenize
            body = json.dumps({"content": "w " * len(payload["tokens"])}).encode().strip()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class TestTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), _StubTok)
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_server_tokenize_count(self):
        tk = Tokenizer(self.url)
        self.assertEqual(tk.tokenize("a b c"), [0, 1, 2])
        self.assertEqual(tk.count("a b c d"), 4)
        self.assertEqual(tk.stats()["backend"], "server")
        self.assertEqual(tk.stats()["calls"], 2)

    def test_detokenize(self):
        tk = Tokenizer(self.url)
        self.assertEqual(tk.detokenize([0, 1, 2]), "w w w")

    def test_fallback(self):
        tk = Tokenizer(None)
        self.assertEqual(tk.count("hello world"), 2)
        self.assertEqual(tk.stats()["backend"], "fallback")
        with self.assertRaises(RuntimeError):
            tk.detokenize([1, 2])


if __name__ == "__main__":
    unittest.main()
