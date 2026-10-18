"""grader: fresh-sandbox SWE-bench scoring on node2 ← slime coding_agent_rl
(fresh sandbox applies patch, runs official tests; design 09 section 5.4).

Flow: fresh sandbox -> clean checkout at base_commit -> apply model patch ->
apply official test_patch -> run resolved pytest nodes -> reward.

reward = 1.0 iff the pytest run over FAIL_TO_PASS + PASS_TO_PASS nodes exits 0.
"""

from __future__ import annotations

import base64
import json
import time

from rl_sim.swebench import resolve_test_nodes


def _write_file_cmd(path: str, content: str) -> str:
    b64 = base64.b64encode(content.encode()).decode()
    return f"echo {b64} | base64 -d > {path}"


def grade(swe: dict, patch: str, sandbox, git_dir: str,
          venv_python: str, profile: str = "B",
          timeout_s: float = 600.0) -> dict:
    """`sandbox` is a create/exec/stop client (HTTP or in-process test double)."""
    t0 = time.time()
    sandbox_id = sandbox.create(profile)["sandbox_id"]
    try:
        commit = swe["base_commit"]
        r = sandbox.exec(
            sandbox_id,
            f"git clone --shared --no-checkout {git_dir} repo "
            f"&& cd repo && git checkout -q {commit}",
            timeout_s=300)
        if r["rc"] != 0:
            return _result(0.0, t0, f"checkout failed: {r['stderr'][-300:]}")

        if patch.strip():
            r = sandbox.exec(sandbox_id,
                             _write_file_cmd("model.patch", patch)
                             + " && cd repo && git apply ../model.patch",
                             timeout_s=120)
            if r["rc"] != 0:
                return _result(0.0, t0, f"model patch apply failed: {r['stderr'][-300:]}")
        r = sandbox.exec(sandbox_id,
                         _write_file_cmd("test.patch", swe["test_patch"])
                         + " && cd repo && git apply ../test.patch",
                         timeout_s=120)
        if r["rc"] != 0:
            return _result(0.0, t0, f"test patch apply failed: {r['stderr'][-300:]}")

        nodes = resolve_test_nodes(swe)
        if not nodes:
            return _result(0.0, t0, "no test nodes resolvable")
        nodes_str = " ".join(f"'{n}'" for n in nodes)
        r = sandbox.exec(
            sandbox_id,
            f"cd repo && PYTHONPATH=$PWD {venv_python} -m pytest {nodes_str} "
            f"-q --tb=no -p no:cacheprovider",
            timeout_s=timeout_s,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin"})
        passed = r["rc"] == 0
        return {
            "reward": 1.0 if passed else 0.0,
            "rc": r["rc"],
            "n_nodes": len(nodes),
            "wall_s": round(time.time() - t0, 3),
            "test_wall_s": r.get("wall_s"),
            "test_cpu_s": r.get("cpu_s"),
            "log_tail": (r["stdout"] + "\n" + r["stderr"])[-2000:],
            "detail": "" if passed else "tests failed",
        }
    finally:
        try:
            sandbox.stop(sandbox_id)
        except Exception:  # noqa: BLE001
            pass


def _result(reward: float, t0: float, detail: str) -> dict:
    return {"reward": reward, "rc": None, "n_nodes": 0,
            "wall_s": round(time.time() - t0, 3),
            "test_wall_s": None, "test_cpu_s": None,
            "log_tail": "", "detail": detail}
