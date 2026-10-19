"""T31 driver acceptance: train_swe.main end-to-end against a fake node2
(in-process HTTP: /trajectory/* + /sysstats + /stats)."""

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import train_swe


def make_fake_node2(state):
    class H(BaseHTTPRequestHandler):
        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/trajectory/start":
                state["started"].append(req["traj_id"])
                return self._json({"traj_id": req["traj_id"], "state": "running"})
            return self._json({"error": "?"}, 404)

        def do_GET(self):
            if self.path.startswith("/trajectory/"):
                return self._json({"state": "done", "result": {
                    "reward": 1.0, "turns": 2, "status": "submitted",
                    "gen_lat_total_s": 1.0, "exec_wall_total_s": 0.5,
                    "grade": {"reward": 1.0}}})
            if self.path == "/sysstats":
                return self._json({"ts": 0.0, "cpu_busy_pct": 1.5, "load1": 0.1,
                                   "mem_total_mb": 1024.0, "mem_used_mb": 512.0})
            if self.path == "/stats":
                return self._json({"active_sandboxes": 0, "exec_total": 0,
                                   "rejected_total": 0})
            return self._json({"error": "?"}, 404)

    return H


class TestTrainSweDriver(unittest.TestCase):
    def test_main_end_to_end(self):
        inst = {"instance_id": "sympy__sympy-1", "repo": "sympy/sympy",
                "base_commit": "abc", "problem_statement": "fix it",
                "test_patch": "", "FAIL_TO_PASS": ["test_x"],
                "PASS_TO_PASS": [], "version": "1.11"}
        tmp = Path(tempfile.mkdtemp())
        (tmp / "instances.jsonl").write_text(json.dumps(inst) + "\n")

        state = {"started": []}
        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_fake_node2(state))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        try:
            summary = train_swe.main([
                "--instances", str(tmp / "instances.jsonl"),
                "--n-samples-per-prompt", "2", "--rollout-batch-size", "1",
                "--sandboxd-url", f"http://127.0.0.1:{port}",
                "--adapter-url", "http://127.0.0.1:1",
                "--adapter-log-dir", str(tmp / "no_adapter_log"),
                "--poll-interval-s", "0.05", "--monitor-interval-s", "0.05",
                "--out", str(tmp / "out")])
        finally:
            srv.shutdown()

        self.assertEqual(summary["mode"], "dual_node_swe")
        h = summary["history"][0]
        self.assertEqual(h["num_groups"], 1)
        self.assertAlmostEqual(h["mean_reward"], 1.0)
        # drop_zero_var=False: an all-identical-reward group stays visible
        self.assertEqual(h["kept_samples"], 2)
        self.assertEqual(len(state["started"]), 2)
        dm = summary["dual_monitor"]
        self.assertGreaterEqual(dm["samples"], 1)
        self.assertEqual(dm["node2_unreachable_samples"], 0)
        self.assertEqual(dm["node2_cpu_busy_mean"], 1.5)
        disk = json.loads((tmp / "out" / "summary.json").read_text())
        self.assertEqual(disk["mode"], "dual_node_swe")


if __name__ == "__main__":
    unittest.main()
