"""netutil acceptance: internal cluster HTTP bypasses http_proxy env.

Root cause being pinned down (T31): node1's shell exports the corporate
proxy and its no_proxy covers 10.0.0.0/8 but not 172.16.0.0/12, so a
node1->node2 urllib call was proxied and 403'd. Note urllib intrinsically
bypasses proxies for loopback targets, so the functional roundtrip below
cannot prove the bypass by itself — the whitebox test on the pinned
ProxyHandler is the guarantee; the roundtrip guards regressions.
"""

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from rl_sim import netutil


class _H(BaseHTTPRequestHandler):
    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json({"ok": True})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        self._json({"echo": req})

    def log_message(self, *a):
        pass


class TestNetutil(unittest.TestCase):
    def test_burst_server_backlog(self):
        # cluster endpoints accept connection bursts (default backlog 5 RSTs)
        self.assertGreaterEqual(
            netutil.BurstThreadingHTTPServer.request_queue_size, 128)
        self.assertTrue(netutil.BurstThreadingHTTPServer.daemon_threads)

    def test_opener_never_proxies(self):        # Contract: no ProxyHandler with a non-empty proxy map. (CPython drops
        # ProxyHandler({}) from build_opener handlers entirely on some
        # versions, so "handler present" is NOT the contract — "no proxies" is.)
        from urllib.request import ProxyHandler
        proxied = [h for h in netutil._opener.handlers
                   if isinstance(h, ProxyHandler) and h.proxies]
        self.assertEqual(proxied, [])

    def test_roundtrip_under_poisoned_proxy_env(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        poison = {"http_proxy": "http://127.0.0.1:9",
                  "https_proxy": "http://127.0.0.1:9", "no_proxy": ""}
        try:
            with mock.patch.dict(os.environ, poison):
                self.assertEqual(netutil.get_json(url + "/x", 5), {"ok": True})
                out = netutil.post_json(url + "/y", {"a": 1}, 5)
                self.assertEqual(out, {"echo": {"a": 1}})
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
