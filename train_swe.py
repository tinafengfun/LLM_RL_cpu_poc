#!/usr/bin/env python3
"""train_swe.py ← slime examples/coding_agent_rl: dual-node SWE-bench rollout driver.

Topology (design 09): this driver runs on node1 next to the adapter and the
llama.cpp engines; node2 runs sandboxd with the trajectory endpoints where
the agent loop and the grader actually live. The driver only dispatches
trajectories (POST /trajectory/start), polls them, and collects results —
no generation and no weight update happen on this path (G4 profiling gate).
Token-level logprobs are pulled from the node1-local adapter log by
RolloutManager._fill_logprob.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rl_sim.data_source import RolloutDataSource
from rl_sim.dual_monitor import DualMonitor
from rl_sim.rollout_manager import RemoteSWEConfig, RolloutManager
from rl_sim.swebench import load_tasks


def main(argv=None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--instances", default="/workspace/swebench/instances.jsonl")
    p.add_argument("--repo", default=None, help="keep only this repo (e.g. sympy/sympy)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-samples-per-prompt", type=int, default=2)
    p.add_argument("--num-rollout", type=int, default=1)
    p.add_argument("--rollout-batch-size", type=int, default=2)
    p.add_argument("--gen-concurrency", type=int, default=16)
    p.add_argument("--adapter-url", default="http://10.239.23.91:18090",
                   help="node1 adapter address AS SEEN FROM node2")
    p.add_argument("--sandboxd-url", default="http://172.16.116.170:18101")
    p.add_argument("--git-dir", default="/workspace/swebench/repo/sympy.git",
                   help="bare repo mirror on node2")
    p.add_argument("--venv", default="/workspace/swebench/envs/py311",
                   help="baked env on node2 (used in place, not relocatable)")
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--agent-timeout-s", type=float, default=1800.0)
    p.add_argument("--grade-timeout-s", type=float, default=600.0)
    p.add_argument("--poll-interval-s", type=float, default=2.0)
    p.add_argument("--adapter-log-dir", default="adapter_log",
                   help="node1-local dir; must match the adapter's --log-dir")
    p.add_argument("--monitor-interval-s", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/train_swe")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.instances, repo=args.repo, limit=args.limit,
                       max_turns=args.max_turns)
    print(f"[swe] {len(tasks)} instance(s) x n={args.n_samples_per_prompt} "
          f"-> node2 {args.sandboxd_url} via adapter {args.adapter_url}")

    ds = RolloutDataSource(tasks, n_samples_per_prompt=args.n_samples_per_prompt,
                           seed=args.seed)
    cfg = RemoteSWEConfig(
        sandboxd_url=args.sandboxd_url, adapter_url=args.adapter_url,
        git_dir=args.git_dir, venv=args.venv, max_turns=args.max_turns,
        agent_timeout_s=args.agent_timeout_s, grade_timeout_s=args.grade_timeout_s,
        poll_interval_s=args.poll_interval_s, adapter_log_dir=args.adapter_log_dir)
    # engine=None / sandbox_pool=None: the SWE line never touches node1-local
    # generation or sandboxes (see test_rollout_remote for the same wiring).
    # drop_zero_var=False: this is a profiling gate — keep all-zero-reward
    # groups visible instead of filtering them as untrainable.
    rm = RolloutManager(ds, engine=None, sandbox_pool=None,
                        remote_swe=cfg, gen_concurrency=args.gen_concurrency,
                        drop_zero_var=False)

    dmon = DualMonitor(args.sandboxd_url, interval_s=args.monitor_interval_s)
    dmon.start()
    history = []
    sample_rows = []
    t_start = time.perf_counter()
    try:
        for rid in range(args.num_rollout):
            data = rm.generate(rid, args.rollout_batch_size)
            line = (f"[swe rollout {rid}] groups {data['num_groups']} "
                    f"| samples {len(data['samples'])} kept {data['kept_samples']} "
                    f"| mean_reward {data['mean_reward']:.3f} "
                    f"| wall {data['wall_s']:.1f}s")
            print(line)
            history.append({"rollout_id": rid, **{k: v for k, v in data.items()
                                                  if k != "samples"}})
            for s in data["samples"]:
                row = {"rollout_id": rid, "task_id": s.task_id,
                       "group_id": s.group_id,
                       "traj_id": s.metadata.get("traj_id"),
                       "reward": s.reward, "turns": s.turns,
                       "status": s.status.value,
                       "rollout_logprob": round(s.rollout_logprob, 2),
                       "num_tokens": s.num_tokens, "sandbox": s.sandbox}
                sample_rows.append(row)
                print(f"  - {s.task_id} g{s.group_id} traj={row['traj_id']} "
                      f"reward={s.reward} turns={s.turns} status={s.status.value} "
                      f"logp={s.rollout_logprob:.2f} ntok={s.num_tokens} "
                      f"sandbox={s.sandbox}")
    finally:
        dmon.stop()

    summary = {"mode": "dual_node_swe",
               "instances": [t.task_id for t in tasks],
               "n_samples_per_prompt": args.n_samples_per_prompt,
               "num_rollout": args.num_rollout,
               "total_wall_s": round(time.perf_counter() - t_start, 2),
               "history": history,
               "sample_rows": sample_rows,
               "dual_monitor": dmon.report()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wall={summary['total_wall_s']}s -> {out / 'summary.json'}")
    return summary


if __name__ == "__main__":
    main()
