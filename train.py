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
from rl_sim.monitor import StageTimer, fmt_pct, fmt_stage_lines
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
        return LocalEngine(mgr.urls(), router=router)
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
                        gen_concurrency=args.gen_concurrency,
                        abort_after_groups=args.abort_after_groups)
    trainer = MockMegatronTrainer(monitor=monitor, eps_clip=args.eps_clip,
                                  eps_clip_high=args.eps_clip_high, use_tis=args.use_tis,
                                  sim_tokens_per_s=args.sim_tokens_per_s)
    updater = WeightUpdater(engine, monitor=monitor, real_sleep_cap_s=0.2)
    eval_tasks = [t for t in ALL_TASKS if t.split == "eval"]

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

            m = pool.metrics.snapshot()
            line = (f"[rollout {rid}] samples {data['num_groups']}g/{len(data['samples'])}s "
                    f"| mean_reward {data['mean_reward']:.3f} "
                    f"| zero_var {data['stats']['zero_var_groups']} "
                    f"| aborted {data['stats']['aborted']} "
                    f"| tis_masked {metrics['tis_masked_frac']:.2%} "
                    f"| wv {metrics['weight_version']}"
                    + (f" | eval {ev['eval_pass_rate']:.3f}" if ev else ""))
            print(line)
            print("  " + fmt_stage_lines(monitor.snapshot()).replace("\n", "\n  "))
            print(f"  [sandbox] busy_peak {m['peak_busy']}/{pool.workers} "
                  f"| q_peak {m['peak_queue']} | exec {fmt_pct(m['exec_walls'])}")
            history.append({"rollout_id": rid, **{k: v for k, v in metrics.items()},
                            "mean_reward": data["mean_reward"], **ev})
    finally:
        pool.shutdown()

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
