"""agent_runner: multi-turn agent loop living on node2 ← slime coding_agent_rl
(agent CLI in sandbox) / DSec worker-container pattern (design 09 section 5.3).

Per trajectory: create sandbox -> materialize repo at base_commit -> loop
(generate via node1 adapter -> execute tool calls via local sandboxd ->
append observations) -> extract patch via git -> persist run artifacts.

Clients are injectable so the loop is unit-testable without any network:
- adapter: object with .generate(traj_id, turn, messages, tools) -> dict
- sandbox: object with .create/.exec/.stop (SandboxHTTPClient in prod)
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
from pathlib import Path

from rl_sim import netutil

TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a bash command in the task sandbox (repo at ./repo).",
        "parameters": {"type": "object",
                       "properties": {"cmd": {"type": "string"}},
                       "required": ["cmd"]}}},
    {"type": "function", "function": {
        "name": "submit",
        "description": "Finish the task; the current repo diff becomes the answer.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

SYSTEM_PROMPT = (
    "You are fixing a bug in a source repository checked out at ./repo "
    "(use `cd repo` first). Inspect code with the bash tool, edit files, and "
    "run the relevant tests. When you believe the bug is fixed, call submit. "
    "Network access is disabled. Keep commands non-interactive."
)


class AdapterHTTPClient:
    def __init__(self, base_url: str, timeout_s: float = 660.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def generate(self, traj_id: str, turn: int, messages: list[dict],
                 tools: list[dict] | None = None) -> dict:
        return netutil.post_json(
            self.base_url + "/generate",
            {"traj_id": traj_id, "turn": turn,
             "messages": messages, "tools": tools},
            self.timeout_s)


class AgentRunner:
    def __init__(self, swe: dict, adapter, sandbox, git_dir: str,
                 run_dir: str, traj_id: str | None = None,
                 max_turns: int = 40, agent_timeout_s: float = 1800.0,
                 obs_limit: int = 8000, profile: str = "B",
                 env_extra: dict | None = None,
                 max_nudges: int = 2) -> None:
        self.swe = swe
        self.adapter = adapter
        self.sandbox = sandbox
        self.git_dir = git_dir
        self.run_dir = Path(run_dir)
        self.traj_id = traj_id or f"traj-{swe['instance_id']}-{int(time.time())}"
        self.max_turns = max_turns
        self.agent_timeout_s = agent_timeout_s
        self.obs_limit = obs_limit
        self.profile = profile
        self.env_extra = env_extra or {}
        self.max_nudges = max_nudges

    # -- steps -------------------------------------------------------------
    def _setup_repo(self, sandbox_id: str) -> None:
        commit = self.swe["base_commit"]
        # sandbox exec drops to nobody on node2; git refuses the root-owned
        # mirror as "dubious ownership". `-c safe.directory` does NOT cover
        # the clone's upload-pack child (git 2.43, verified) — a global
        # config under a sandbox-local HOME does.
        cmd = (f"mkdir -p .sandhome && export HOME=$PWD/.sandhome && "
               f"git config --global --add safe.directory {self.git_dir} && "
               f"git clone --shared --no-checkout {self.git_dir} repo "
               f"&& cd repo && git checkout -q {commit} && git log --oneline -1")
        r = self.sandbox.exec(sandbox_id, cmd, timeout_s=300)
        if r["rc"] != 0:
            raise RuntimeError(f"repo setup failed: {r['stderr'][-300:]}")

    def _extract_patch(self, sandbox_id: str) -> str:
        r = self.sandbox.exec(
            sandbox_id,
            "cd repo && git add -A >/dev/null 2>&1; git diff --cached",
            timeout_s=120)
        return r["stdout"] if r["rc"] == 0 else ""

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        half = limit // 2
        return (text[:half] + f"\n...[truncated {len(text) - limit} chars]...\n"
                + text[-half:])

    # -- main loop ---------------------------------------------------------
    def run(self) -> dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        t_start = time.time()
        status = "error"
        turn_stats: list[dict] = []
        messages: list[dict] = []
        patch = ""
        sandbox_id = None
        try:
            sandbox_id = self.sandbox.create(self.profile)["sandbox_id"]
            self._setup_repo(sandbox_id)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                 f"Issue to fix:\n\n{self.swe['problem_statement']}"},
            ]
            nudges = 0
            status = "max_turns"
            for turn in range(self.max_turns):
                if time.time() - t_start > self.agent_timeout_s:
                    status = "timeout"
                    break
                t_gen = time.time()
                try:
                    resp = self.adapter.generate(self.traj_id, turn,
                                                 messages, tools=TOOLS)
                except urllib.error.HTTPError as e:
                    # the adapter's 502 body carries the backend's own message
                    try:
                        detail = e.read().decode("utf-8", "replace")[:200]
                    except Exception:  # noqa: BLE001 - body unreadable
                        detail = ""
                    status = f"adapter_error: HTTP {e.code} {detail}".strip()
                    break
                except Exception as e:  # noqa: BLE001
                    status = f"adapter_error: {type(e).__name__}: {e}"
                    break
                gen_lat = time.time() - t_gen

                messages.append({"role": "assistant",
                                 "content": resp.get("text") or "",
                                 "tool_calls": resp.get("tool_calls") or []})
                tool_calls = resp.get("tool_calls") or []
                exec_wall = 0.0

                if any(tc["function"]["name"] == "submit" for tc in tool_calls):
                    status = "submitted"
                    turn_stats.append({"turn": turn, "gen_lat_s": round(gen_lat, 4),
                                       "exec_wall_s": 0.0, "n_tool_calls": 0})
                    break
                if not tool_calls:
                    nudges += 1
                    turn_stats.append({"turn": turn, "gen_lat_s": round(gen_lat, 4),
                                       "exec_wall_s": 0.0, "n_tool_calls": 0})
                    if nudges > self.max_nudges:
                        status = "no_tool_call"
                        break
                    messages.append({"role": "user", "content":
                                     "Use the bash tool to work on the repo, "
                                     "or call submit when done."})
                    continue
                nudges = 0

                for tc in tool_calls:
                    fn = tc["function"]
                    result_text = ""
                    if fn["name"] == "bash":
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                            cmd = args.get("cmd", "")
                        except json.JSONDecodeError:
                            cmd = ""
                            result_text = "error: malformed tool arguments"
                        if cmd:
                            r = self.sandbox.exec(sandbox_id, cmd,
                                                  env=self.env_extra)
                            exec_wall += r.get("wall_s", 0.0)
                            result_text = (f"exit={r['rc']}\n"
                                           + self._truncate(
                                               (r["stdout"] or "")
                                               + ("\nSTDERR:\n" + r["stderr"]
                                                  if r["stderr"] else ""),
                                               self.obs_limit))
                    else:
                        result_text = f"error: unknown tool {fn['name']}"
                    messages.append({"role": "tool",
                                     "tool_call_id": tc.get("id", ""),
                                     "content": result_text})
                turn_stats.append({"turn": turn, "gen_lat_s": round(gen_lat, 4),
                                   "exec_wall_s": round(exec_wall, 4),
                                   "n_tool_calls": len(tool_calls)})

            patch = self._extract_patch(sandbox_id)
        finally:
            if sandbox_id is not None:
                try:
                    self.sandbox.stop(sandbox_id)
                except Exception:  # noqa: BLE001
                    pass

        (self.run_dir / "patch.diff").write_text(patch)
        with open(self.run_dir / "messages.jsonl", "w", encoding="utf-8") as f:
            for m in messages:
                f.write(json.dumps(m) + "\n")
        meta = {
            "traj_id": self.traj_id,
            "instance_id": self.swe.get("instance_id"),
            "status": status,
            "turns": len(turn_stats),
            "wall_s": round(time.time() - t_start, 3),
            "gen_lat_total_s": round(sum(t["gen_lat_s"] for t in turn_stats), 3),
            "exec_wall_total_s": round(sum(t["exec_wall_s"] for t in turn_stats), 3),
            "turn_stats": turn_stats,
        }
        (self.run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        meta["patch"] = patch
        return meta


def main() -> None:
    from rl_sim.sandbox_client import SandboxHTTPClient

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter-url", required=True)
    ap.add_argument("--sandboxd-url", default="http://127.0.0.1:18101")
    ap.add_argument("--git-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--swe-file", required=True,
                    help="json file with the swe payload (instance spec)")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--agent-timeout-s", type=float, default=1800.0)
    ap.add_argument("--venv", default="",
                    help="venv path whose bin/ is prepended to PATH")
    args = ap.parse_args()

    swe = json.loads(Path(args.swe_file).read_text())
    env_extra = {}
    if args.venv:
        env_extra["PATH"] = f"{args.venv}/bin:/usr/local/bin:/usr/bin:/bin"
    runner = AgentRunner(
        swe, AdapterHTTPClient(args.adapter_url),
        SandboxHTTPClient(args.sandboxd_url),
        git_dir=args.git_dir, run_dir=args.run_dir,
        max_turns=args.max_turns, agent_timeout_s=args.agent_timeout_s,
        env_extra=env_extra)
    print(json.dumps(runner.run(), indent=2))


if __name__ == "__main__":
    main()
