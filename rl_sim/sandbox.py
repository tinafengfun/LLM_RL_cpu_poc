"""SandboxPool core: single-task sandboxed execution ← DeepSeek DSec (function/container tier).

Executes model-generated code + tests in a subprocess with hard resource
limits via prlimit(1) (thread-safe alternative to preexec_fn rlimits):
  - RLIMIT_CPU  : hard CPU seconds (SIGXCPU on expiry)
  - RLIMIT_AS   : address space (MemoryError on over-allocation)
  - RLIMIT_FSIZE / NOFILE

Protocol: runner.py writes one line `__RESULT__{json}` to stdout carrying
status / passed / cpu_s / maxrss_kb / io counters / boot_ts.
Network isolation (unshare -n) and uid drop are layered on in T06 via
`isolation_prefix()`.
"""

from __future__ import annotations

import json
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from rl_sim.types import Sample, SampleStatus

# Per-tier resource profiles (design v3.1 section 5.1): one-size limits are a
# distortion source — pytest+numpy cannot start under a 512MB A-tier cap.
PROFILES: dict[str, dict] = {
    "A": dict(cpu_s=5, mem_mb=512, wall_s=10, fsize_mb=16),
    "B": dict(cpu_s=120, mem_mb=4096, wall_s=300, fsize_mb=256),
    "C": dict(cpu_s=10, mem_mb=1024, wall_s=300, fsize_mb=64),
    "MOCK": dict(cpu_s=5, mem_mb=512, wall_s=10, fsize_mb=16),
}

_RUNNER = r'''
import json, resource, time
_boot_ts = time.time()
result = {"status": "error", "passed": 0, "total": __NTOTAL__, "detail": "", "boot_ts": _boot_ts}
try:
    from solution import *
    tests = __TESTS__
    npass, errs = 0, []
    for t in tests:
        try:
            exec(t, dict(globals()))
            npass += 1
        except BaseException as e:
            errs.append(("%s: %s" % (type(e).__name__, e))[:200])
    result["passed"] = npass
    result["status"] = "passed" if npass == len(tests) else "failed"
    if errs:
        result["detail"] = errs[0]
except MemoryError:
    result["status"] = "oom"
except BaseException as e:
    result["status"] = "error"
    result["detail"] = ("%s: %s" % (type(e).__name__, e))[:200]
ru = resource.getrusage(resource.RUSAGE_SELF)
result["cpu_s"] = round(ru.ru_utime + ru.ru_stime, 4)
result["maxrss_kb"] = ru.ru_maxrss
try:
    with open("/proc/self/io") as f:
        io = {l.split(":")[0].strip(): int(l.split(":")[1]) for l in f if ":" in l}
    result["io_read_bytes"] = io.get("read_bytes", 0)
    result["io_write_bytes"] = io.get("write_bytes", 0)
except Exception:
    pass
print("__RESULT__" + json.dumps(result))
'''


def isolation_prefix() -> list[str]:
    """Extra command prefix for isolation (T06: unshare -n / setpriv nobody). Empty for now."""
    return []


def _build_cmd(profile: dict) -> list[str]:
    mem_bytes = profile["mem_mb"] * 1024 * 1024
    fsize_bytes = profile["fsize_mb"] * 1024 * 1024
    return isolation_prefix() + [
        "prlimit",
        f"--as={mem_bytes}",
        f"--cpu={profile['cpu_s']}:{profile['cpu_s']}",
        f"--fsize={fsize_bytes}",
        "--nofile=64:64",
        sys.executable, "runner.py",
    ]


def run_sandbox_task(
    sample: Sample,
    code: str,
    tests: list[str],
    tier: str = "A",
    scratch_root: str | Path = "/tmp/rl_sim_scratch",
    overrides: dict | None = None,
) -> Sample:
    """Execute `code` + assert-style `tests` for one sample; fill status + metrics."""
    profile = dict(PROFILES[tier])
    if overrides:
        profile.update(overrides)

    submit_ts = time.time()
    Path(scratch_root).mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="rlsb_", dir=scratch_root))
    try:
        (tmpdir / "solution.py").write_text(code)
        runner = _RUNNER.replace("__NTOTAL__", str(len(tests))).replace("__TESTS__", repr(tests))
        (tmpdir / "runner.py").write_text(runner)

        spawn_ts = time.time()
        try:
            proc = subprocess.run(
                _build_cmd(profile),
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=profile["wall_s"],
            )
            end_ts = time.time()
            _map_result(sample, proc)
        except subprocess.TimeoutExpired:
            end_ts = time.time()
            sample.status = SampleStatus.SANDBOX_TIMEOUT
            sample.sandbox["detail"] = f"wall timeout {profile['wall_s']}s"

        boot_ts = sample.sandbox.get("boot_ts", spawn_ts)
        sample.sandbox.update(
            submit_ts=submit_ts,
            spawn_ts=spawn_ts,
            end_ts=end_ts,
            launch_overhead_s=round(boot_ts - spawn_ts, 4),
            exec_wall_s=round(end_ts - spawn_ts, 4),
            tier=tier,
        )
        return sample
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _map_result(sample: Sample, proc: subprocess.CompletedProcess) -> None:
    """Map process outcome + runner JSON onto sample."""
    if proc.returncode in (-signal.SIGXCPU, -signal.SIGKILL):
        # SIGXCPU at soft RLIMIT_CPU; SIGKILL at hard limit (observed on Linux 6.x:
        # the kernel goes straight to SIGKILL when soft==hard). RLIMIT_AS breaches
        # surface as MemoryError inside the runner, so SIGKILL here means CPU limit.
        sample.status = SampleStatus.SANDBOX_TIMEOUT
        sample.sandbox["detail"] = f"CPU limit exceeded (rc={proc.returncode})"
        return
    payload = _parse_result_line(proc.stdout)
    if payload is None:
        sample.status = SampleStatus.SANDBOX_ERROR
        tail = (proc.stderr or proc.stdout or "")[-200:]
        sample.sandbox["detail"] = f"rc={proc.returncode}: {tail}"
        return
    sample.sandbox.update(
        {k: payload[k] for k in ("cpu_s", "maxrss_kb", "io_read_bytes", "io_write_bytes", "boot_ts", "detail") if k in payload}
    )
    sample.tests_passed = payload.get("passed", 0)
    sample.tests_total = payload.get("total", 0)
    status = payload.get("status")
    if status in ("passed", "failed"):
        sample.status = SampleStatus.COMPLETED
    elif status == "oom":
        sample.status = SampleStatus.SANDBOX_OOM
    else:
        sample.status = SampleStatus.SANDBOX_ERROR


def _parse_result_line(stdout: str | None) -> dict | None:
    if not stdout:
        return None
    for line in reversed(stdout.splitlines()):
        if line.startswith("__RESULT__"):
            try:
                return json.loads(line[len("__RESULT__"):])
            except json.JSONDecodeError:
                return None
    return None


# ---------------------------------------------------------------------------
# SandboxPool: fixed worker pool with burst/poisson arrival + queue metrics.
# One worker = one serial executor, mirroring "one env instance per container".
# ---------------------------------------------------------------------------

import threading
from concurrent.futures import ThreadPoolExecutor


class PoolMetrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.submitted = 0
        self.completed = 0
        self.running = 0
        self.peak_queue = 0
        self.peak_busy = 0
        self.queue_waits: list[float] = []
        self.exec_walls: list[float] = []

    def snapshot(self) -> dict:
        with self.lock:
            return dict(
                submitted=self.submitted, completed=self.completed,
                running=self.running, peak_queue=self.peak_queue,
                peak_busy=self.peak_busy,
                queue_waits=list(self.queue_waits), exec_walls=list(self.exec_walls),
            )

    def reset(self) -> None:
        with self.lock:
            self.__init__()


class SandboxPool:
    """Fixed-size pool; `workers` bounds true concurrency so bursts build a real queue."""

    def __init__(self, workers: int = 8, scratch_root: str | Path = "/tmp/rl_sim_scratch") -> None:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        self.workers = workers
        self.scratch_root = scratch_root
        self.metrics = PoolMetrics()
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sandbox")

    def _run_one(self, sample: Sample, code: str, tests: list[str], tier: str, overrides: dict | None) -> Sample:
        m = self.metrics
        with m.lock:
            m.running += 1
            m.peak_busy = max(m.peak_busy, m.running)
        try:
            run_sandbox_task(sample, code, tests, tier=tier,
                             scratch_root=self.scratch_root, overrides=overrides)
        finally:
            with m.lock:
                m.running -= 1
                m.completed += 1
                sb = sample.sandbox
                if "pool_submit_ts" in sb and "submit_ts" in sb:
                    m.queue_waits.append(sb["submit_ts"] - sb["pool_submit_ts"])
                if "exec_wall_s" in sb:
                    m.exec_walls.append(sb["exec_wall_s"])
        return sample

    def submit(self, sample: Sample, code: str, tests: list[str], tier: str = "A",
               overrides: dict | None = None):
        m = self.metrics
        with m.lock:
            m.submitted += 1
            queued = m.submitted - m.completed - m.running
            m.peak_queue = max(m.peak_queue, queued)
        sample.sandbox["pool_submit_ts"] = time.time()
        return self._executor.submit(self._run_one, sample, code, tests, tier, overrides)

    def run_batch(self, tasks: list[tuple[Sample, str, list[str], str]],
                  overrides: dict | None = None) -> list[Sample]:
        """Burst arrival: submit everything at t=0, gather in submission order."""
        futures = [self.submit(s, c, t, tier, overrides) for (s, c, t, tier) in tasks]
        return [f.result() for f in futures]

    def arrive_poisson(self, tasks: list[tuple[Sample, str, list[str], str]],
                       rate_per_s: float) -> list[Sample]:
        """Poisson arrival process; submits as they arrive, gathers at the end."""
        import random
        futures = []
        for task in tasks:
            futures.append(self.submit(*task))
            time.sleep(random.expovariate(rate_per_s))
        return [f.result() for f in futures]

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
