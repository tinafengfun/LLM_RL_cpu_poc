"""trajectory_worker: one SWE trajectory end-to-end on node2 (design 09 §4).

= agent_runner (multi-turn loop, model calls cross to node1) + grader (fresh
sandbox). Spawned by sandboxd /trajectory/start as a tracked subprocess;
writes result.json into run_dir for the status endpoint and node1 polling.

Runs UNISOLATED as the trajectory orchestrator (trusted code); everything the
model touches goes through sandboxd, which isolates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rl_sim.agent_runner import AgentRunner, AdapterHTTPClient
from rl_sim.grader import grade
from rl_sim.sandbox_client import SandboxHTTPClient

GRADED_STATUSES = {"submitted", "max_turns", "timeout", "no_tool_call"}

# per-call timeout to the node1 adapter — MUST exceed the adapter's own
# --backend-timeout-s (default 1800s), else the agent gives up on a call the
# adapter is still patiently waiting on (T32 R3: 5 trajectories degraded to
# adapter_error at the 660s client default while engines were just queued)
ADAPTER_CALL_TIMEOUT_S = 2000.0


def run_trajectory(swe: dict, adapter_url: str, sandboxd_url: str,
                   git_dir: str, venv: str, run_dir: str,
                   traj_id: str | None = None, max_turns: int = 40,
                   agent_timeout_s: float = 1800.0,
                   grade_timeout_s: float = 600.0,
                   profile: str = "B") -> dict:
    sandbox = SandboxHTTPClient(sandboxd_url)
    env_extra = {"PATH": f"{venv}/bin:/usr/local/bin:/usr/bin:/bin"} if venv else {}
    runner = AgentRunner(swe, AdapterHTTPClient(adapter_url,
                                                timeout_s=ADAPTER_CALL_TIMEOUT_S),
                         sandbox,
                         git_dir=git_dir, run_dir=run_dir, traj_id=traj_id,
                         agent_timeout_s=agent_timeout_s,
                         max_turns=max_turns, env_extra=env_extra,
                         profile=profile)
    meta = runner.run()
    patch = meta.pop("patch")

    if meta["status"] in GRADED_STATUSES:
        g = grade(swe, patch, sandbox, git_dir,
                  venv_python=f"{venv}/bin/python" if venv else sys.executable,
                  timeout_s=grade_timeout_s)
    else:
        g = {"reward": 0.0, "detail": f"not graded (status={meta['status']})"}

    result = {**meta, "reward": g["reward"], "grade": g}
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(run_dir) / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter-url", required=True)
    ap.add_argument("--sandboxd-url", default="http://127.0.0.1:18101")
    ap.add_argument("--git-dir", required=True)
    ap.add_argument("--venv", default="")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--swe-file", required=True)
    ap.add_argument("--traj-id", default=None)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--agent-timeout-s", type=float, default=1800.0)
    ap.add_argument("--grade-timeout-s", type=float, default=600.0)
    args = ap.parse_args()

    swe = json.loads(Path(args.swe_file).read_text())
    result = run_trajectory(
        swe, args.adapter_url, args.sandboxd_url, args.git_dir, args.venv,
        args.run_dir, traj_id=args.traj_id, max_turns=args.max_turns,
        agent_timeout_s=args.agent_timeout_s,
        grade_timeout_s=args.grade_timeout_s)
    print(json.dumps({k: v for k, v in result.items() if k != "turn_stats"},
                     indent=2))


if __name__ == "__main__":
    main()
