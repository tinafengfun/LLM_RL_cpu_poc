"""T30 acceptance: DualMonitor alignment + /sysstats endpoint."""

import json
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rl_sim.dual_monitor import DualMonitor
from rl_sim.sandboxd import read_sys_stats


class FakeNode2(BaseHTTPRequestHandler):
    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/sysstats":
            return self._json({"ts": time.time(), "cpu_busy_pct": 42.0,
                               "load1": 1.0, "mem_total_mb": 1000,
                               "mem_used_mb": 500})
        if self.path == "/stats":
            return self._json({"active_sandboxes": 7, "exec_total": 99,
                               "rejected_total": 3})
        return self._json({"error": "?"}, 404)


class TestSysStats(unittest.TestCase):
    def test_read_sys_stats_shape(self):
        s = read_sys_stats(sample_s=0.01)
        for k in ("ts", "cpu_busy_pct", "load1", "mem_total_mb", "mem_used_mb"):
            self.assertIn(k, s)
        self.assertGreaterEqual(s["cpu_busy_pct"], 0.0)
        self.assertGreater(s["mem_total_mb"], 0)


class TestDualMonitor(unittest.TestCase):
    def test_aligned_rows_and_report(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeNode2)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        try:
            m = DualMonitor(f"http://127.0.0.1:{port}", interval_s=0.05)
            m.start()
            time.sleep(0.3)
            m.stop()
            self.assertGreaterEqual(len(m.rows), 2)
            row = m.rows[-1]
            # one tick carries both sides -> aligned by construction
            self.assertIn("node1_sys", row)
            self.assertEqual(row["node2_sys"]["cpu_busy_pct"], 42.0)
            self.assertEqual(row["node2_sandbox"]["active_sandboxes"], 7)
            rep = m.report()
            self.assertEqual(rep["node2_cpu_busy_mean"], 42.0)
            self.assertEqual(rep["node2_sandbox_peak_active"], 7)
            self.assertEqual(rep["node2_exec_total"], 99)
            self.assertEqual(rep["node2_rejected_total"], 3)
            self.assertEqual(rep["node2_unreachable_samples"], 0)
        finally:
            srv.shutdown()

    def test_node2_down_is_tolerated(self):
        m = DualMonitor("http://127.0.0.1:1", interval_s=0.05, timeout_s=0.2)
        row = m.sample_once()
        self.assertIn("node2_error", row)
        self.assertIn("node1_sys", row)
        self.assertEqual(m.report()["node2_unreachable_samples"], 1)


if __name__ == "__main__":
    unittest.main()
