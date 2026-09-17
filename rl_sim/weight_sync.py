"""WeightUpdater ← slime update_weight/UpdateWeightFromDistributed (TIMING simulation).

Simulates the pause -> transfer -> resume sequence and its scheduling effects.
Transfer time is modelled as param_gb / bandwidth_gbps (real sleep is capped);
with LocalEngine on the node, an optional real GGUF reload hook measures the
actual mmap/page-in window (design v3 section 5.5). No NCCL/RDMA fidelity
is claimed — this is control-flow + window measurement only.
"""

from __future__ import annotations

import time

from rl_sim.engine import Engine


class WeightUpdater:
    def __init__(self, engine: Engine, monitor=None,
                 sim_param_gb: float = 4.0, sim_bandwidth_gbps: float = 50.0,
                 real_sleep_cap_s: float = 0.5, reload_hook=None) -> None:
        self.engine = engine
        self.monitor = monitor
        self.sim_param_gb = sim_param_gb
        self.sim_bandwidth_gbps = sim_bandwidth_gbps
        self.real_sleep_cap_s = real_sleep_cap_s
        self.reload_hook = reload_hook  # e.g. LocalEngine GGUF reload (node)
        self.history: list[dict] = []

    def update_weights(self, new_version: int) -> dict:
        t0 = time.perf_counter()
        self.engine.pause_generation()

        sim_transfer_s = self.sim_param_gb / self.sim_bandwidth_gbps
        time.sleep(min(self.real_sleep_cap_s, sim_transfer_s))

        reload_s = 0.0
        if self.reload_hook is not None:
            r0 = time.perf_counter()
            self.reload_hook()
            reload_s = time.perf_counter() - r0

        self.engine.update_weights(new_version)
        self.engine.continue_generation()
        pause_window_s = time.perf_counter() - t0

        rec = {"version": new_version, "sim_transfer_s": sim_transfer_s,
               "pause_window_s": pause_window_s, "reload_s": reload_s}
        self.history.append(rec)
        if self.monitor is not None:
            self.monitor.record("weight_sync", pause_window_s, pause_window_s)
        return rec
