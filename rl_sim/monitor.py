"""StageTimer / stats ← monitoring system (design doc v3.1 section 5.6).

Dual timing: every stage records BOTH wall time and driver-process CPU time.
Inference/network waits must show up as wall-only, never as CPU (the whole
point of the CPU-load profile). Child-process CPU (sandbox, llama-server) is
measured on the child side (rusage / engine metrics), not here.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager


class StageTimer:
    """Accumulates per-stage wall and process-CPU seconds. Thread-safe."""

    def __init__(self) -> None:
        self._wall: dict[str, float] = {}
        self._cpu: dict[str, float] = {}
        self._lock = threading.Lock()

    @contextmanager
    def stage(self, name: str):
        w0 = time.perf_counter()
        c0 = time.process_time()
        try:
            yield
        finally:
            self.record(
                name,
                time.perf_counter() - w0,
                time.process_time() - c0,
            )

    def record(self, name: str, wall_s: float, cpu_s: float) -> None:
        with self._lock:
            self._wall[name] = self._wall.get(name, 0.0) + wall_s
            self._cpu[name] = self._cpu.get(name, 0.0) + cpu_s

    def snapshot(self) -> dict[str, tuple[float, float]]:
        """{stage: (wall_s, cpu_s)} in insertion order."""
        with self._lock:
            return {k: (self._wall[k], self._cpu.get(k, 0.0)) for k in self._wall}

    def reset(self) -> None:
        with self._lock:
            self._wall.clear()
            self._cpu.clear()


def percentiles(values: list[float], ps: tuple[int, ...] = (50, 95, 99)) -> dict[int, float]:
    """Nearest-rank percentiles; empty input -> zeros."""
    if not values:
        return {p: 0.0 for p in ps}
    xs = sorted(values)
    out = {}
    for p in ps:
        rank = max(1, -(-p * len(xs) // 100))  # ceil without math import
        out[p] = xs[min(rank, len(xs)) - 1]
    return out


def fmt_stage_lines(snapshot: dict[str, tuple[float, float]]) -> str:
    """Two aligned lines, wall and cpu, e.g.:

    wall: orchestrate 0.4s | infer 38.3s | sandbox 11.8s
    cpu : orchestrate 0.3s | infer 36.1s | sandbox  9.2s
    """
    if not snapshot:
        return "wall: (no stages)\ncpu : (no stages)"
    wall = " | ".join(f"{k} {v[0]:.1f}s" for k, v in snapshot.items())
    cpu = " | ".join(f"{k} {v[1]:.1f}s" for k, v in snapshot.items())
    return f"wall: {wall}\ncpu : {cpu}"


def fmt_pct(values: list[float], unit: str = "s") -> str:
    """'p50 0.08s p95 0.5s p99 0.9s' for a list of samples."""
    p = percentiles(values)
    return " ".join(f"p{k} {v:.2f}{unit}" for k, v in p.items())


def rollout_report(rollout_id: int, stages: dict[str, tuple[float, float]],
                   sandbox_metrics: dict, sandbox_workers: int,
                   train_metrics: dict, engine_stats: dict | None = None,
                   eval_metrics: dict | None = None) -> str:
    """Full per-rollout report in design v3.1 section 5.6 format."""
    lines = [
        f"[rollout {rollout_id}]",
        "  " + fmt_stage_lines(stages).replace("\n", "\n  "),
    ]
    if engine_stats:
        lat = fmt_pct(engine_stats.get("lat", []))
        lines.append(f"  [engine] backends {engine_stats.get('backends', 0)} "
                     f"| reqs {engine_stats.get('requests', 0)} "
                     f"| tokens {engine_stats.get('tokens', 0)} | lat {lat}")
    m = sandbox_metrics
    lines.append(f"  [sandbox] busy_peak {m.get('peak_busy', 0)}/{sandbox_workers} "
                 f"| q_peak {m.get('peak_queue', 0)} "
                 f"| exec {fmt_pct(m.get('exec_walls', []))} "
                 f"| queue_wait {fmt_pct(m.get('queue_waits', []))}")
    lines.append(f"  [train] wv {train_metrics.get('weight_version')} "
                 f"| samples {train_metrics.get('num_samples')} "
                 f"| loss {train_metrics.get('loss', 0.0):.4f} "
                 f"| tis_masked {train_metrics.get('tis_masked_frac', 0.0):.2%} "
                 f"| staleness_max {train_metrics.get('staleness_max', 0)}")
    if eval_metrics:
        lines.append(f"  [eval] pass_rate {eval_metrics.get('eval_pass_rate', 0.0):.3f} "
                     f"(n={eval_metrics.get('eval_n', 0)})")
    return "\n".join(lines)
