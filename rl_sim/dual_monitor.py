"""DualMonitor: aligned two-node timeline ← design 09 section 5.5.

node1 (driver side) samples its own /proc stats and pulls node2's
sandboxd /stats + /sysstats on the same tick, producing one aligned row per
sample. Clock skew between nodes is a declared distortion (design 09 F7);
rows are keyed by node1 wall clock.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request

from rl_sim.sandboxd import read_sys_stats


class DualMonitor:
    def __init__(self, node2_base_url: str, interval_s: float = 2.0,
                 timeout_s: float = 10.0) -> None:
        self.node2 = node2_base_url.rstrip("/")
        self.interval_s = interval_s
        self.timeout_s = timeout_s
        self.rows: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _pull(self, path: str) -> dict:
        with urllib.request.urlopen(self.node2 + path,
                                    timeout=self.timeout_s) as r:
            return json.loads(r.read())

    def sample_once(self) -> dict:
        row = {"ts": time.time(), "node1_sys": read_sys_stats(sample_s=0.05)}
        try:
            row["node2_sys"] = self._pull("/sysstats")
            row["node2_sandbox"] = self._pull("/stats")
        except Exception as e:  # noqa: BLE001 - node2 down must not kill the run
            row["node2_error"] = f"{type(e).__name__}: {e}"
        self.rows.append(row)
        return row

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def report(self) -> dict:
        """Aggregates over the aligned timeline (means/peaks per side)."""
        n1 = [r["node1_sys"]["cpu_busy_pct"] for r in self.rows if "node1_sys" in r]
        n2 = [r["node2_sys"]["cpu_busy_pct"] for r in self.rows if "node2_sys" in r]
        sb = [r["node2_sandbox"] for r in self.rows if "node2_sandbox" in r]
        return {
            "samples": len(self.rows),
            "node1_cpu_busy_mean": round(sum(n1) / len(n1), 2) if n1 else None,
            "node1_cpu_busy_peak": max(n1) if n1 else None,
            "node2_cpu_busy_mean": round(sum(n2) / len(n2), 2) if n2 else None,
            "node2_cpu_busy_peak": max(n2) if n2 else None,
            "node2_sandbox_peak_active": max(
                (s["active_sandboxes"] for s in sb), default=None),
            "node2_exec_total": sb[-1]["exec_total"] if sb else None,
            "node2_rejected_total": sb[-1]["rejected_total"] if sb else None,
            "node2_unreachable_samples": sum(1 for r in self.rows
                                             if "node2_error" in r),
        }
