#!/usr/bin/env python3
"""run_experiments.py: E0-E6 sweep harness (design v3.1 section 6).

E0  baseline noise capture (shared machine!) -> results/baseline/*.jsonl
E1  sandbox workers sweep: throughput vs queue wait (burst of A-tier tasks)
E3  inference concurrency vs driver CPU% (needs an engine; [node])
E4  sync vs async comparison (train.py vs train_async.py same config)
E6  engine instances x slots sweep (llama-server; [node])
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from rl_sim.data_source import ALL_TASKS, task_by_id
from rl_sim.monitor import percentiles


# --------------------------------------------------------------------------- E0
def e0_baseline(duration_s: float, interval_s: float, out_dir: Path) -> Path:
    """Sample node CPU/mem/IO while idle — the noise floor all experiments stand on."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"baseline_{int(time.time())}.jsonl"
    with open(path, "w") as f:
        t_end = time.time() + duration_s
        prev_cpu = prev_io = None
        while time.time() < t_end:
            with open("/proc/stat") as fh:
                parts = fh.readline().split()
            cpu = list(map(int, parts[1:]))
            with open("/proc/meminfo") as fh:
                mem = {l.split(":")[0]: int(l.split()[1]) for l in fh if ":" in l}
            with open("/proc/diskstats") as fh:
                io = sum(int(l.split()[9]) for l in fh if len(l.split()) > 10)
            with open("/proc/loadavg") as fh:
                load1 = float(fh.read().split()[0])
            rec = {"ts": time.time(), "load1": load1,
                   "mem_available_kb": mem.get("MemAvailable", 0)}
            if prev_cpu is not None:
                d = [b - a for a, b in zip(prev_cpu, cpu)]
                idle = d[3] + d[4]
                rec["cpu_busy_frac"] = round(1 - idle / max(sum(d), 1), 4)
                rec["io_write_sectors_delta"] = io - prev_io
            prev_cpu, prev_io = cpu, io
            f.write(json.dumps(rec) + "\n")
            f.flush()
            time.sleep(interval_s)
    return path


# --------------------------------------------------------------------------- E1
def e1_sandbox_sweep(workers_list: list[int], n_tasks: int, out_dir: Path) -> list[dict]:
    """Burst of A-tier tasks through SandboxPool at several pool sizes.

    Uses MockEngine-generated code (pure load baseline, NOT a training signal).
    """
    from rl_sim.sandbox import SandboxPool
    from rl_sim.types import Sample

    rows = []
    tasks_src = [t for t in ALL_TASKS if t.line == "A"]
    for w in workers_list:
        pool = SandboxPool(workers=w)
        batch = []
        for i in range(n_tasks):
            t = tasks_src[i % len(tasks_src)]
            batch.append((Sample(prompt=t.prompt, task_id=t.task_id), t.solution, t.tests, "A"))
        t0 = time.perf_counter()
        pool.run_batch(batch)
        wall = time.perf_counter() - t0
        m = pool.metrics.snapshot()
        qw = percentiles(m["queue_waits"])
        row = {"workers": w, "n_tasks": n_tasks, "wall_s": round(wall, 2),
               "throughput_per_s": round(n_tasks / wall, 2),
               "queue_wait_p50": round(qw[50], 3), "queue_wait_p99": round(qw[99], 3),
               "peak_busy": m["peak_busy"], "peak_queue": m["peak_queue"]}
        rows.append(row)
        print(f"[E1] {row}")
        pool.shutdown()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e1_sandbox_sweep.json").write_text(json.dumps(rows, indent=2))
    return rows


# --------------------------------------------------------------------------- E4
def e4_sync_vs_async(num_rollout: int, batch: int, n_samples: int,
                     workers: int, out_dir: Path) -> dict:
    import train
    import train_async
    common = ["--engine", "mock", "--num-rollout", str(num_rollout),
              "--rollout-batch-size", str(batch), "--n-samples-per-prompt", str(n_samples),
              "--sandbox-workers", str(workers), "--sim-tokens-per-s", "1e18"]
    s = train.main(common + ["--out", str(out_dir / "e4_sync")])
    a = train_async.main(common + ["--out", str(out_dir / "e4_async")])
    row = {"sync_wall_s": s["total_wall_s"], "async_wall_s": a["total_wall_s"],
           "speedup": round(s["total_wall_s"] / max(a["total_wall_s"], 1e-9), 2)}
    print(f"[E4] {row}")
    (out_dir / "e4_sync_vs_async.json").write_text(json.dumps(row, indent=2))
    return row


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=["E0", "E1", "E3", "E4", "E6"])
    p.add_argument("--out", default="results")
    p.add_argument("--duration-s", type=float, default=600.0)   # E0
    p.add_argument("--interval-s", type=float, default=5.0)     # E0
    p.add_argument("--workers", default="4,32,128")             # E1
    p.add_argument("--n-tasks", type=int, default=256)          # E1
    p.add_argument("--num-rollout", type=int, default=4)        # E4
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=4)
    args = p.parse_args(argv)

    out = Path(args.out)
    if args.exp == "E0":
        path = e0_baseline(args.duration_s, args.interval_s, out / "baseline")
        print(f"[E0] baseline written: {path}")
    elif args.exp == "E1":
        e1_sandbox_sweep([int(x) for x in args.workers.split(",")], args.n_tasks, out)
    elif args.exp == "E4":
        e4_sync_vs_async(args.num_rollout, args.batch, args.n_samples, 8, out)
    else:
        raise SystemExit(f"{args.exp} runs on the node with real engines; see README")


if __name__ == "__main__":
    main()
