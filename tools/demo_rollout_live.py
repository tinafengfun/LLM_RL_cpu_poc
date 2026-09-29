#!/usr/bin/env python3
"""Live concurrency demo: watch one rollout's layers work in real time.

Layers shown (design doc 08 section 2):
  GEN      engine in-flight (bounded by --gen-concurrency, like GPU slots)
  SANDBOX  pool busy / queued (bounded by --sandbox-workers)
  BARRIER  per-group completion: groups partially done are WAITING on stragglers

Run:  python tools/demo_rollout_live.py            (offline, mock engine)
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_sim.data_source import A_TASKS, RolloutDataSource
from rl_sim.engine import MockEngine
from rl_sim.rollout_manager import RolloutManager
from rl_sim.sandbox import SandboxPool


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=4, help="prompts per rollout")
    p.add_argument("--n", type=int, default=8, help="samples per prompt (group size)")
    p.add_argument("--gen-concurrency", type=int, default=6)
    p.add_argument("--sandbox-workers", type=int, default=2)
    p.add_argument("--latency", default="0.2:1.2", help="mock gen latency lo:hi seconds")
    p.add_argument("--interval", type=float, default=0.5, help="refresh seconds")
    args = p.parse_args()

    lo, hi = (float(x) for x in args.latency.split(":"))
    tasks_by_id = {t.task_id: t for t in A_TASKS}
    engine = MockEngine(tasks_by_id, seed=0, base_correct=0.6, slope=0.0,
                        latency_s=(lo, hi))
    ds = RolloutDataSource(A_TASKS, n_samples_per_prompt=args.n, seed=0)
    pool = SandboxPool(workers=args.sandbox_workers)
    rm = RolloutManager(ds, engine, pool, gen_concurrency=args.gen_concurrency)

    total = args.batch * args.n
    print(f"=== 一次 rollout：{args.batch} 题 × {args.n} 份 = {total} 个样本 ===")
    print(f"闸门：生成并发 ≤{args.gen_concurrency} | sandbox 位 {args.sandbox_workers} | "
          f"组屏障：同组 {args.n} 份全到齐才结算\n")

    stop = threading.Event()
    result = {}

    def run():
        result["data"] = rm.generate(0, args.batch)
        stop.set()

    def watch():
        t0 = time.perf_counter()
        while not stop.is_set():
            lv = rm.live
            m = pool.metrics.snapshot()
            partial = sum(1 for d, t in lv["group_progress"].values() if 0 < d < t)
            line = (f"[t={time.perf_counter() - t0:5.1f}s] "
                    f"GEN 忙 {engine.inflight}/{args.gen_concurrency} 等待 {lv['gen_waiting']:2d} | "
                    f"SANDBOX 忙 {m['running']}/{args.sandbox_workers} 排队 {m['submitted'] - m['completed'] - m['running']:2d} | "
                    f"完成 {lv['done']:2d}/{total} | "
                    f"组齐 {lv['groups_done']}/{args.batch} (卡屏障 {partial})")
            print(line, flush=True)
            time.sleep(args.interval)

    th = threading.Thread(target=run)
    th.start()
    watch()
    th.join()
    pool.shutdown()

    data = result["data"]
    kept = data["kept_samples"]
    print(f"\n=== 结算 ===")
    print(f"总样本 {total} | mean_reward {data['mean_reward']:.3f} | "
          f"零方差组丢弃 {data['stats']['zero_var_groups']}/{data['num_groups']} | "
          f"进入训练 {kept}")
    m = pool.metrics.snapshot()
    from rl_sim.monitor import fmt_pct
    print(f"sandbox exec {fmt_pct(m['exec_walls'])} | queue_wait {fmt_pct(m['queue_waits'])}")
    print("\n看点回顾：GEN 满负荷时样本在等生成位；sandbox 位少时代码排队；"
          "\n组齐那栏的'卡屏障'就是 GRPO 组等待——最慢一条拖住整组。")


if __name__ == "__main__":
    main()
