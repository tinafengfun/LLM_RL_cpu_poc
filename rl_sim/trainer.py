"""MockMegatronTrainer ← slime/ray/actor_group.py + backends/megatron_utils/loss.py.

GPU stand-in. REAL scalar CPU work (mirrors what the trainer host computes
outside the GEMMs): GRPO surrogate with PPO clip, sequence-level importance
ratio (GSPO-style; llama.cpp has no prompt-rescoring endpoint, so per-token
TIS is out of scope), IcePop-style masking, staleness stats.
FAKE part: no backward pass — simulated GPU time via token-count sleep, then
weight_version += 1.

Trainer-side logprob comes from `rescore_fn` (default: rollout logprob +
calibrated noise growing with staleness; the Q8 offline scorer plugs in here
on the node — design v3 section 5.4).
"""

from __future__ import annotations

import math
import random
import time

from rl_sim.types import Sample


class MockMegatronTrainer:
    def __init__(self, monitor=None, eps_clip: float = 0.2, eps_clip_high: float = 0.28,
                 use_tis: bool = True, tis_beta: float = 2.0,
                 sim_tokens_per_s: float = 50000.0, sim_max_s: float = 3.0,
                 mismatch_sigma: float = 0.05, seed: int = 0, rescore_fn=None) -> None:
        self.monitor = monitor
        self.eps_low = eps_clip
        self.eps_high = eps_clip_high
        self.use_tis = use_tis
        self.tis_beta = tis_beta
        self.sim_tokens_per_s = sim_tokens_per_s
        self.sim_max_s = sim_max_s
        self.mismatch_sigma = mismatch_sigma
        self._rng = random.Random(seed)
        self._rescore_fn = rescore_fn
        self.weight_version = 0

    # ------------------------------------------------------------------ core
    def rescore(self, sample: Sample, staleness: int) -> float:
        """Trainer-side sequence logprob. Default: calibrated-noise model."""
        if self._rescore_fn is not None:
            return self._rescore_fn(sample)
        sigma = self.mismatch_sigma * (1.0 + staleness)
        noise = self._rng.gauss(0.0, sigma * math.sqrt(max(sample.num_tokens, 1)))
        return sample.rollout_logprob + noise

    def async_train(self, rollout_id: int, data: dict) -> dict:
        t0 = time.perf_counter()
        samples: list[Sample] = data["samples"]
        n_clipped = n_masked = 0
        losses = []
        staleness_list = []
        total_tokens = 0

        for s in samples:
            staleness = max(0, self.weight_version - s.weight_version)
            staleness_list.append(staleness)
            s.train_logprob = self.rescore(s, staleness)
            ratio = math.exp(s.train_logprob - s.rollout_logprob)

            if self.use_tis:
                # IcePop-style: tokens/sequences outside the trust region are
                # masked to ZERO gradient (harder than PPO's one-sided clip)
                if not (1.0 / self.tis_beta <= ratio <= self.tis_beta):
                    n_masked += 1
                    continue

            clipped = min(max(ratio, 1 - self.eps_low), 1 + self.eps_high)
            if clipped != ratio:
                n_clipped += 1
            losses.append(-min(ratio * s.advantage, clipped * s.advantage))
            total_tokens += s.num_tokens

        # simulated GPU training time (fake compute, real accounting)
        sim_s = min(self.sim_max_s, total_tokens / self.sim_tokens_per_s) if total_tokens else 0.01
        time.sleep(sim_s)
        self.weight_version += 1

        metrics = {
            "rollout_id": rollout_id,
            "weight_version": self.weight_version,
            "num_samples": len(samples),
            "loss": (sum(losses) / len(losses)) if losses else 0.0,
            "frac_clipped": n_clipped / len(samples) if samples else 0.0,
            "tis_masked_frac": n_masked / len(samples) if samples else 0.0,
            "staleness_max": max(staleness_list) if staleness_list else 0,
            "total_tokens": total_tokens,
            "sim_train_s": sim_s,
            "cpu_wall_s": time.perf_counter() - t0,
        }
        if self.monitor is not None:
            self.monitor.record("train_mock_gpu", metrics["cpu_wall_s"], metrics["cpu_wall_s"])
        return metrics

    # ------------------------------------------------------------------ checkpoint
    def save_checkpoint(self, path: str, extra: dict | None = None) -> None:
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "weight_version": self.weight_version, **(extra or {})}, indent=2))
