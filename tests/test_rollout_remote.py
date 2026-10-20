"""T29-B acceptance: rollout_manager remote SWE dispatch against a fake
node2 sandboxd (in-process HTTP), incl. group barrier + abort behaviour."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rl_sim.data_source import RolloutDataSource, Task
from rl_sim.rollout_manager import RemoteSWEConfig, RolloutManager
from rl_sim.types import SampleStatus, TaskLine


class FakeNode2State:
    def __init__(self, hang_instance_ids=()):
        self.started = []
        self.aborted = []
        self.hang_instance_ids = set(hang_instance_ids)
        self.swe_by_traj = {}
        self.done_count = 0
        self.lock = threading.Lock()
        # fault injection: N next start/poll attempts get a dropped connection
        self.reset_starts = 0
        self.reset_polls = 0


def make_fake_node2(state: FakeNode2State):
    class H(BaseHTTPRequestHandler):
        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _drop(self):
            # close without responding -> client sees a reset/disconnect
            self.connection.close()

        def log_message(self, *a):
            pass

        def do_POST(self):
            if self.path == "/trajectory/start" and state.reset_starts > 0:
                state.reset_starts -= 1
                return self._drop()
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/trajectory/start":
                with state.lock:
                    state.started.append(req["traj_id"])
                    state.swe_by_traj[req["traj_id"]] = req["swe"]["instance_id"]
                return self._json({"traj_id": req["traj_id"], "state": "running"})
            if self.path == "/trajectory/abort":
                with state.lock:
                    state.aborted.append(req["traj_id"])
                return self._json({"aborted": True})
            return self._json({"error": "?"}, 404)

        def do_GET(self):
            if self.path.startswith("/trajectory/"):
                if state.reset_polls > 0:
                    state.reset_polls -= 1
                    return self._drop()
                traj_id = self.path.split("/")[2]
                with state.lock:
                    hang = state.swe_by_traj.get(traj_id) in state.hang_instance_ids
                if hang:
                    return self._json({"state": "running"})
                with state.lock:
                    state.done_count += 1
                    reward = 1.0 if state.done_count % 2 == 1 else 0.0
                return self._json({"state": "done", "result": {
                    "reward": reward, "turns": 3, "status": "submitted",
                    "gen_lat_total_s": 1.5, "exec_wall_total_s": 0.5,
                    "grade": {"reward": reward}}})
            return self._json({"error": "?"}, 404)

    return H


def _swe_task(task_id="swe_t1", instance_id="x"):
    return Task(task_id=task_id, line=TaskLine.SWE.value, tier="B",
                prompt="fix", reward_spec={"type": "swebench"}, max_turns=5,
                swe={"instance_id": instance_id, "repo": "r",
                     "base_commit": "c", "test_patch": "",
                     "fail_to_pass": ["t"], "pass_to_pass": []})


class TestRemoteSWE(unittest.TestCase):
    def _mgr(self, state, tasks=None, **kw):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_fake_node2(state))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        cfg = RemoteSWEConfig(sandboxd_url=f"http://127.0.0.1:{port}",
                              adapter_url="http://127.0.0.1:1",
                              git_dir="/x.git", poll_interval_s=0.05,
                              adapter_log_dir="/tmp/nonexistent_adapter_log")
        ds = RolloutDataSource(tasks or [_swe_task("swe_t1"), _swe_task("swe_t2")],
                               n_samples_per_prompt=2, seed=0)
        mgr = RolloutManager(ds, engine=None, sandbox_pool=None,
                             remote_swe=cfg, **kw)
        return mgr, srv

    def test_group_completes_and_rewards_flow(self):
        state = FakeNode2State()
        mgr, srv = self._mgr(state)
        try:
            out = mgr.generate(rollout_id=0, batch_size=1)
        finally:
            srv.shutdown()
        self.assertEqual(out["num_groups"], 1)
        # rewards alternate 1.0/0.0 -> non-zero variance -> group kept
        self.assertEqual(out["kept_samples"], 2)
        self.assertAlmostEqual(out["mean_reward"], 0.5)
        rewards = sorted(s.reward for s in out["samples"])
        self.assertEqual(rewards, [0.0, 1.0])
        self.assertTrue(all(s.status is SampleStatus.COMPLETED
                            for s in out["samples"]))
        self.assertTrue(all(s.turns == 3 for s in out["samples"]))
        self.assertEqual(len(state.started), 2)
        self.assertEqual(state.aborted, [])

    def test_abort_fires_remote_abort(self):
        # group on instance "hang" never finishes; the other completes ->
        # abort_after_groups=1 cancels the hanging one mid-flight
        state = FakeNode2State(hang_instance_ids={"hang"})
        tasks = [_swe_task("swe_ok", "ok"), _swe_task("swe_hang", "hang")]
        mgr, srv = self._mgr(state, tasks=tasks, abort_after_groups=1)
        try:
            out = mgr.generate(rollout_id=0, batch_size=2)
        finally:
            srv.shutdown()
        self.assertEqual(out["num_groups"], 1)  # only the ok group
        hung = [t for t in state.started
                if state.swe_by_traj[t] == "hang"]
        self.assertEqual(len(hung), 2)
        self.assertEqual(sorted(state.aborted), sorted(hung))
        self.assertEqual(mgr.stats["aborted"], 2)
        self.assertEqual(len(mgr.partial_buffer), 2)

    def test_start_retry_recovers_from_reset(self):
        state = FakeNode2State()
        state.reset_starts = 2  # first two connect attempts die (burst RST)
        mgr, srv = self._mgr(state)
        try:
            out = mgr.generate(rollout_id=0, batch_size=1)
        finally:
            srv.shutdown()
        self.assertEqual(out["kept_samples"], 2)
        self.assertEqual(sorted(s.reward for s in out["samples"]), [0.0, 1.0])
        self.assertEqual(len(state.started), 2)  # retried starts landed once

    def test_start_permanent_failure_degrades_to_sandbox_error(self):
        state = FakeNode2State()
        state.reset_starts = 99  # node2 unreachable for every attempt
        mgr, srv = self._mgr(state, drop_zero_var=False)
        try:
            out = mgr.generate(rollout_id=0, batch_size=1)
        finally:
            srv.shutdown()
        # the failed starts degrade to SANDBOX_ERROR samples; the rollout
        # itself must survive (T32: one RST used to kill all 44)
        self.assertEqual(len(out["samples"]), 2)
        self.assertTrue(all(s.status is SampleStatus.SANDBOX_ERROR
                            for s in out["samples"]))
        self.assertTrue(all("start failed" in s.sandbox["detail"]
                            for s in out["samples"]))
        self.assertEqual(state.started, [])

    def test_poll_reset_is_tolerated(self):
        state = FakeNode2State()
        state.reset_polls = 3  # first three status polls die, then recover
        mgr, srv = self._mgr(state)
        try:
            out = mgr.generate(rollout_id=0, batch_size=1)
        finally:
            srv.shutdown()
        self.assertEqual(out["kept_samples"], 2)
        self.assertEqual(sorted(s.reward for s in out["samples"]), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
