"""T27 acceptance: agent_runner loop with scripted fake adapter + fake sandbox."""

import json
import shutil
import tempfile
import unittest

from rl_sim.agent_runner import TOOLS, AgentRunner


def _tc(name, args, tc_id="c1"):
    return {"id": tc_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class FakeAdapter:
    """Plays back scripted responses; records the message history per call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate(self, traj_id, turn, messages, tools=None):
        self.calls.append({"turn": turn, "messages": list(messages),
                           "tools": tools})
        if not self.script:
            return {"text": "", "tool_calls": [], "finish_reason": "stop"}
        return self.script.pop(0)


class FakeSandbox:
    def __init__(self):
        self.execs = []
        self.stopped = False

    def create(self, profile="B"):
        return {"sandbox_id": "sb-fake", "workdir": "/tmp/sb-fake"}

    def exec(self, sandbox_id, cmd, timeout_s=None, env=None):
        self.execs.append(cmd)
        if "git diff" in cmd:
            return {"rc": 0, "stdout": "diff --git a/f.py b/f.py\n+fix",
                    "stderr": "", "wall_s": 0.01}
        return {"rc": 0, "stdout": "ok", "stderr": "", "wall_s": 0.01}

    def stop(self, sandbox_id):
        self.stopped = True
        return True


SWE = {"instance_id": "sympy__sympy-X", "repo": "sympy/sympy",
       "base_commit": "abc", "problem_statement": "fix it",
       "test_patch": "", "fail_to_pass": ["t"], "pass_to_pass": []}


class TestAgentRunner(unittest.TestCase):
    def setUp(self):
        self.run_dir = tempfile.mkdtemp(prefix="runner_test_")

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def _run(self, script, **kw):
        adapter, sandbox = FakeAdapter(script), FakeSandbox()
        r = AgentRunner(SWE, adapter, sandbox, git_dir="/x/y.git",
                        run_dir=self.run_dir, **kw)
        return r.run(), adapter, sandbox

    def test_submit_terminates_with_patch(self):
        meta, adapter, sandbox = self._run([
            {"text": "", "tool_calls": [_tc("bash", {"cmd": "ls"})]},
            {"text": "done", "tool_calls": [_tc("submit", {})]},
        ])
        self.assertEqual(meta["status"], "submitted")
        self.assertEqual(meta["turns"], 2)
        self.assertIn("diff --git", meta["patch"])
        # setup clone happened, then bash cmd, then patch extraction
        # (clone registers safe.directory in a sandbox-local HOME: exec drops
        # to nobody on node2, and git refuses a root-owned mirror otherwise)
        self.assertIn("safe.directory", sandbox.execs[0])
        self.assertIn("clone --shared", sandbox.execs[0])
        self.assertIn("ls", sandbox.execs[1])
        self.assertTrue(sandbox.stopped)
        # artifacts persisted
        with open(f"{self.run_dir}/messages.jsonl") as f:
            msgs = [json.loads(l) for l in f]
        self.assertEqual(msgs[0]["role"], "system")
        self.assertTrue(any(m["role"] == "tool" for m in msgs))
        # tools schema was passed to adapter
        self.assertEqual(adapter.calls[0]["tools"], TOOLS)

    def test_no_tool_call_nudges_then_stops(self):
        meta, adapter, _ = self._run(
            [{"text": "thinking out loud", "tool_calls": []}] * 5,
            max_nudges=2)
        self.assertEqual(meta["status"], "no_tool_call")
        self.assertEqual(meta["turns"], 3)
        # nudge messages were appended
        last_call = adapter.calls[-1]["messages"]
        self.assertIn("bash tool", last_call[-1]["content"])

    def test_max_turns(self):
        meta, _, _ = self._run(
            [{"text": "", "tool_calls": [_tc("bash", {"cmd": "ls"})]}] * 10,
            max_turns=3)
        self.assertEqual(meta["status"], "max_turns")
        self.assertEqual(meta["turns"], 3)

    def test_timeout(self):
        meta, _, _ = self._run(
            [{"text": "", "tool_calls": [_tc("bash", {"cmd": "ls"})]}],
            max_turns=50, agent_timeout_s=-1)  # already expired
        self.assertEqual(meta["status"], "timeout")
        self.assertEqual(meta["turns"], 0)

    def test_adapter_error(self):
        class BoomAdapter(FakeAdapter):
            def generate(self, *a, **k):
                raise ConnectionError("adapter down")
        sandbox = FakeSandbox()
        r = AgentRunner(SWE, BoomAdapter([]), sandbox, git_dir="/x/y.git",
                        run_dir=self.run_dir)
        meta = r.run()
        self.assertIn("adapter_error", meta["status"])
        self.assertTrue(sandbox.stopped)

    def test_observation_truncation(self):
        class LoudSandbox(FakeSandbox):
            def exec(self, sandbox_id, cmd, timeout_s=None, env=None):
                if "git diff" in cmd:
                    return super().exec(sandbox_id, cmd)
                return {"rc": 0, "stdout": "x" * 20000, "stderr": "",
                        "wall_s": 0.01}
        adapter = FakeAdapter([
            {"text": "", "tool_calls": [_tc("bash", {"cmd": "cat big"})]},
            {"text": "", "tool_calls": [_tc("submit", {})]},
        ])
        r = AgentRunner(SWE, adapter, LoudSandbox(), git_dir="/x/y.git",
                        run_dir=self.run_dir, obs_limit=1000)
        meta = r.run()
        self.assertEqual(meta["status"], "submitted")
        tool_msg = adapter.calls[1]["messages"][-1]
        self.assertLess(len(tool_msg["content"]), 1200)
        self.assertIn("truncated", tool_msg["content"])

    def test_unknown_tool_reported(self):
        meta, adapter, _ = self._run([
            {"text": "", "tool_calls": [_tc("read_file", {"p": "x"})]},
            {"text": "", "tool_calls": [_tc("submit", {})]},
        ])
        self.assertEqual(meta["status"], "submitted")
        tool_msg = adapter.calls[1]["messages"][-1]
        self.assertIn("unknown tool", tool_msg["content"])


if __name__ == "__main__":
    unittest.main()
