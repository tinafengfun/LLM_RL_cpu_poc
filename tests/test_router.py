"""T10 acceptance: router session affinity, failover, HTTP forwarding."""

import json
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rl_sim.router import BackendRing, Router, serve


class _Echo(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        payload["echo"] = True
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def _start_backend():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Echo)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


class TestBackendRing(unittest.TestCase):
    def test_affinity_stable(self):
        ring = BackendRing()
        ring.add("a", "http://a")
        ring.add("b", "http://b")
        first = ring.pick("session-42")[0]
        for _ in range(10):
            self.assertEqual(ring.pick("session-42")[0], first)

    def test_distribution_not_degenerate(self):
        ring = BackendRing()
        for n in "abcd":
            ring.add(n, f"http://{n}")
        picks = {ring.pick(f"key-{i}")[0] for i in range(200)}
        self.assertGreaterEqual(len(picks), 3)  # 200 keys spread over 4 backends

    def test_failover_rehash(self):
        ring = BackendRing()
        ring.add("a", "http://a")
        ring.add("b", "http://b")
        victim = ring.pick("k")[0]
        ring.mark_failure(victim)
        self.assertNotEqual(ring.pick("k")[0], victim)
        self.assertEqual(ring.failures[victim], 1)

    def test_no_backends(self):
        self.assertIsNone(BackendRing().pick("x"))


class TestRouterHTTP(unittest.TestCase):
    def test_forward_via_http_front(self):
        b1, u1 = _start_backend()
        b2, u2 = _start_backend()
        self.addCleanup(b1.shutdown)
        self.addCleanup(b2.shutdown)
        router = Router()
        router.ring.add("b1", u1)
        router.ring.add("b2", u2)
        srv, _ = serve(router)
        self.addCleanup(srv.shutdown)
        url = f"http://127.0.0.1:{srv.server_address[1]}/generate"
        req = urllib.request.Request(
            url, data=json.dumps({"x": 1}).encode(),
            headers={"Content-Type": "application/json", "X-Session-Key": "s-1"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            out = json.loads(resp.read().decode())
        self.assertTrue(out["echo"])
        self.assertIn(out["_backend"], ("b1", "b2"))
        self.assertEqual(len(router.forward_lat_s), 1)

    def test_failover_when_backend_dead(self):
        b1, u1 = _start_backend()
        self.addCleanup(b1.shutdown)
        router = Router(timeout_s=2)
        router.ring.add("live", u1)
        router.ring.add("dead", "http://127.0.0.1:9")
        # force affinity onto the dead backend by retrying keys until hit
        key = next(k for k in (f"k{i}" for i in range(1000))
                   if router.ring.pick(k)[0] == "dead")
        out = router.forward("/generate", {"x": 1}, session_key=key)
        self.assertEqual(out["_backend"], "live")
        self.assertEqual(router.ring.failures["dead"], 1)


if __name__ == "__main__":
    unittest.main()
