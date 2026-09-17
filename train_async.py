#!/usr/bin/env python3
"""train_async.py ← slime train_async.py: one-step lookahead asynchronous pipeline.

While rollout k trains, rollout k+1 generates. Weight sync happens every
--update-weights-interval rollouts; per slime, in-flight generation is awaited
before pushing new weights. Samples generated during a sync window carry an
older weight_version -> staleness = trainer_version - sample_version; samples
beyond --max-staleness are dropped (GLM-5 Section 4 rule w' - w_0 > tau).
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rl_sim.monitor import StageTimer, fmt_stage_lines
from train import build_engine  # reuse wiring
from rl_sim.data_source import ALL_TASKS, RolloutDataSource
from rl_sim.rollout_manager import RolloutManager
from rl_sim.sandbox import SandboxPool
from rl_sim.trainer import MockMegatronTrainer
from rl_sim.weight_sync import WeightUpdater


def main(argv=None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["mock", "local", "api"], default="mock")
    p.add_argument("--num-rollout", type=int, default=5)
    p.add_argument("--rollout-batch-size", type=int, default=4)
    p.add_argument("--n-samples-per-prompt", type=int, default=4)
    p.add_argument("--gen-concurrency", type=int, default=16)
    p.add_argument("--sandbox-workers", type=int, default=8)
    p.add_argument("--update-weights-interval", type=int, default=2)
    p.add_argument("--max-staleness", type=int, default=None)
    p.add_argument("--sim-tokens-per-s", type=float, default=50000.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/train_async")
    p.add_argument("--llama-bin", default="/mnt/nvme0/models/llama.cpp/build/bin/llama-server")
    p.add_argument("--model", default="/mnt/nvme0/models/Qwen3-4B-Q4_K_M.gguf")
    p.add_argument("--engine-instances", type=int, default=2)
    p.add_argument("--engine-slots", type=int, default=8)
    p.add_argument("--base-port", type=int, default=19300)
    p.add_argument("--cores-base", type=int, default=0)
    p.add_argument("--cores-per-engine", type=int, default=16)
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks_by_id = {t.task_id: t for t in ALL_TASKS}
    engine = build_engine(args, tasks_by_id)
    ds = RolloutDataSource(ALL_TASKS, n_samples_per_prompt=args.n_samples_per_prompt,
                           seed=args.seed)
    pool = SandboxPool(workers=args.sandbox_workers)
    monitor = StageTimer()
    rm = RolloutManager(ds, engine, pool, monitor=monitor,
                        gen_concurrency=args.gen_concurrency)
    trainer = MockMegatronTrainer(monitor=monitor, sim_tokens_per_s=args.sim_tokens_per_s)
    updater = WeightUpdater(engine, monitor=monitor, real_sleep_cap_s=0.2)

    history = []
    t_start = time.perf_counter()
    dropped_stale_total = 0
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="lookahead") as gen_pool:
        next_data = gen_pool.submit(rm.generate, 0, args.rollout_batch_size)
        try:
            for rid in range(args.num_rollout):
                with monitor.stage("generate"):
                    data = next_data.result()  # await in-flight generation
                if rid + 1 < args.num_rollout:
                    # lookahead: start next rollout generation DURING training
                    next_data = gen_pool.submit(rm.generate, rid + 1, args.rollout_batch_size)

                if args.max_staleness is not None:
                    before = len(data["samples"])
                    data["samples"] = [
                        s for s in data["samples"]
                        if trainer.weight_version - s.weight_version <= args.max_staleness]
                    dropped_stale_total += before - len(data["samples"])

                metrics = trainer.async_train(rid, data)

                if (rid + 1) % args.update_weights_interval == 0:
                    updater.update_weights(trainer.weight_version)

                line = (f"[async rollout {rid}] samples {len(data['samples'])} "
                        f"| mean_reward {data['mean_reward']:.3f} "
                        f"| staleness_max {metrics['staleness_max']} "
                        f"| tis_masked {metrics['tis_masked_frac']:.2%} "
                        f"| wv {metrics['weight_version']}")
                print(line)
                history.append({"rollout_id": rid, **metrics,
                                "mean_reward": data["mean_reward"]})
        finally:
            pool.shutdown()

    summary = {"engine": args.engine, "mode": "async_lookahead",
               "num_rollout": args.num_rollout,
               "final_weight_version": trainer.weight_version,
               "dropped_stale_total": dropped_stale_total,
               "total_wall_s": round(time.perf_counter() - t_start, 2),
               "history": history}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wv={trainer.weight_version} dropped_stale={dropped_stale_total} "
          f"wall={summary['total_wall_s']}s")
    print(fmt_stage_lines(monitor.snapshot()))
    return summary


if __name__ == "__main__":
    main()
