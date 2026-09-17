#!/usr/bin/env python3
"""train.py ← slime train.py: synchronous RL driver loop.

Per rollout_id (mirrors slime's five steps):
  1. rollout_manager.generate   (groups -> engine -> sandbox -> reward -> advantage)
  2. trainer.async_train        (mock GPU: real scalars, simulated time, version+1)
  3. save_model                 (every --save-interval)
  4. weight_sync.update_weights (pause -> transfer -> resume)
  5. rollout_manager.eval       (every --eval-interval, held-out tasks)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rl_sim.data_source import ALL_TASKS, RolloutDataSource, task_by_id
from rl_sim.engine import APIEngine, MockEngine
from rl_sim.monitor import StageTimer, rollout_report
from rl_sim.rollout_manager import RolloutManager
from rl_sim.sandbox import SandboxPool
from rl_sim.trainer import MockMegatronTrainer
from rl_sim.weight_sync import WeightUpdater


def build_engine(args, tasks_by_id):
    if args.engine == "mock":
        return MockEngine(tasks_by_id, seed=args.seed)
    if args.engine == "api":
        return APIEngine.from_env()
    if args.engine == "local":
        from rl_sim.engine_local import InstanceSpec, LlamaServerManager, LocalEngine
        from rl_sim.router import Router
        specs = [
            InstanceSpec(name=f"e{i}", port=args.base_port + i,
                         cores=f"{args.cores_base + i * args.cores_per_engine}-"
                               f"{args.cores_base + (i + 1) * args.cores_per_engine - 1}",
                         slots=args.engine_slots, model=args.model)
            for i in range(args.engine_instances)
        ]
        mgr = LlamaServerManager(args.llama_bin, specs)
        mgr.start_all()
        if not mgr.wait_healthy(600):
            raise RuntimeError("engines failed to come up")
        router = Router()
        eng = LocalEngine(mgr.urls(), router=router)
        eng._manager = mgr  # caller owns lifecycle (stopped in main's finally)
        return eng
    raise ValueError(args.engine)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["mock", "local", "api"], default="mock")
    p.add_argument("--num-rollout", type=int, default=5)
    p.add_argument("--rollout-batch-size", type=int, default=4)
    p.add_argument("--n-samples-per-prompt", type=int, default=4)
    p.add_argument("--gen-concurrency", type=int, default=16)
    p.add_argument("--sandbox-workers", type=int, default=8)
    p.add_argument("--eps-clip", type=float, default=0.2)
    p.add_argument("--eps-clip-high", type=float, default=0.28)
    p.add_argument("--use-tis", action="store_true", default=True)
    p.add_argument("--save-interval", type=int, default=5)
    p.add_argument("--eval-interval", type=int, default=5)
    p.add_argument("--abort-after-groups", type=int, default=None)
    p.add_argument("--sim-tokens-per-s", type=float, default=50000.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/train_sync")
    # local engine
    p.add_argument("--llama-bin", default="/mnt/nvme0/models/llama.cpp/build/bin/llama-server")
    p.add_argument("--model", default="/mnt/nvme0/models/Qwen3-4B-Q4_K_M.gguf")
    p.add_argument("--engine-instances", type=int, default=2)
    p.add_argument("--engine-slots", type=int, default=8)
    p.add_argument("--base-port", type=int, default=19300)
    p.add_argument("--cores-base", type=int, default=0)
    p.add_argument("--cores-per-engine", type=int, default=16)
    p.add_argument("--task-lines", default="A",
                   help="comma list: A,L1,L2,L3,V1,V2 (V* needs a VLM engine; L1/L2 land in T16+)")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    task_lines = set(args.task_lines.split(","))
    tasks = [t for t in ALL_TASKS if t.line in task_lines]
    tasks_by_id = {t.task_id: t for t in tasks}
    engine = build_engine(args, {t.task_id: t for t in ALL_TASKS})
    ds = RolloutDataSource(tasks, n_samples_per_prompt=args.n_samples_per_prompt,
                           seed=args.seed)
    pool = SandboxPool(workers=args.sandbox_workers)
    monitor = StageTimer()
    rm = RolloutManager(ds, engine, pool, monitor=monitor,
                        gen_concurrency=args.gen_concurrency,
                        abort_after_groups=args.abort_after_groups)
    trainer = MockMegatronTrainer(monitor=monitor, eps_clip=args.eps_clip,
                                  eps_clip_high=args.eps_clip_high, use_tis=args.use_tis,
                                  sim_tokens_per_s=args.sim_tokens_per_s)
    updater = WeightUpdater(engine, monitor=monitor, real_sleep_cap_s=0.2)
    eval_tasks = [t for t in tasks if t.split == "eval"]

    history = []
    t_start = time.perf_counter()
    try:
        for rid in range(args.num_rollout):
            with monitor.stage("generate"):
                data = rm.generate(rid, args.rollout_batch_size)
            metrics = trainer.async_train(rid, data)

            if (rid + 1) % args.save_interval == 0 or rid == args.num_rollout - 1:
                trainer.save_checkpoint(str(out / f"ckpt_v{trainer.weight_version:04d}.json"),
                                        {"rollout_id": rid})

            updater.update_weights(trainer.weight_version)

            ev = {}
            if (rid + 1) % args.eval_interval == 0 or rid == args.num_rollout - 1:
                with monitor.stage("eval"):
                    ev = rm.eval(eval_tasks)

            eng_stats = engine.engine_stats() if hasattr(engine, "engine_stats") else None
            print(rollout_report(rid, monitor.snapshot(), pool.metrics.snapshot(),
                                 pool.workers, metrics, eng_stats, ev or None))
            print(f"  [data] mean_reward {data['mean_reward']:.3f} "
                  f"| kept {data['kept_samples']} | gen_errors {data['gen_errors']} "
                  f"| zero_var {data['stats']['zero_var_groups']}")
            monitor.reset()
            history.append({"rollout_id": rid, **{k: v for k, v in metrics.items()},
                            "mean_reward": data["mean_reward"], **ev})
    finally:
        pool.shutdown()
        mgr = getattr(engine, "_manager", None)
        if mgr is not None:
            mgr.stop_all()

    summary = {"engine": args.engine, "num_rollout": args.num_rollout,
               "final_weight_version": trainer.weight_version,
               "total_wall_s": round(time.perf_counter() - t_start, 2),
               "history": history}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] weight_version={trainer.weight_version} "
          f"wall={summary['total_wall_s']}s -> {out}/summary.json")
    return summary


if __name__ == "__main__":
    main()
