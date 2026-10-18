"""sandboxd: node2 sandbox HTTP service ← DSec edge (design 09 section 5.2).

A stdlib-only daemon that turns rl_sim.sandbox's isolation primitives
(prlimit + unshare -n + setpriv) into a *service*: named, stateful sandboxes
with persistent workdirs, arbitrary shell exec, admission control, TTL reaping
and /stats. This is the execution substrate for agent_runner (T27) and
grader (T28) on node2.

API (JSON over HTTP):
  POST /sandbox/create  {profile, sandbox_id?} -> {sandbox_id, workdir}
  POST /sandbox/exec    {sandbox_id, cmd, timeout_s?, env?} ->
                        {rc, stdout, stderr, wall_s, cpu_s, maxrss_kb, timed_out}
  POST /sandbox/stop    {sandbox_id} -> {stopped}
  GET  /stats           -> counters
  GET  /healthz         -> "ok"

Concurrency: ThreadingHTTPServer; one in-flight exec per sandbox; admission
caps both concurrent execs and the summed memory budget of live sandboxes.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil

from rl_sim.sandbox import PROFILES, isolation_prefix

DEFAULT_SCRATCH = "/tmp/sandboxd_scratch"


class SandboxEntry:
    def __init__(self, sandbox_id: str, workdir: str, profile: str) -> None:
        self.sandbox_id = sandbox_id
        self.workdir = workdir
        self.profile = profile
        self.created_ts = time.time()
        self.last_used_ts = self.created_ts
        self.proc: subprocess.Popen | None = None
        self.exec_count = 0


class AdmissionError(Exception):
    pass


class SandboxRegistry:
    """Thread-safe lifecycle + admission; pure logic (unit-tested directly)."""

    def __init__(self, scratch_root: str = DEFAULT_SCRATCH,
                 max_concurrent_exec: int = 32,
                 mem_cap_mb: int = 60000,
                 ttl_s: float = 600.0,
                 self_url: str = "http://127.0.0.1:18101") -> None:
        self.scratch_root = scratch_root
        self.max_concurrent_exec = max_concurrent_exec
        self.mem_cap_mb = mem_cap_mb
        self.ttl_s = ttl_s
        self.self_url = self_url
        self.lock = threading.Lock()
        self.sandboxes: dict[str, SandboxEntry] = {}
        self.trajectories: dict[str, dict] = {}
        self.created_total = 0
        self.exec_total = 0
        self.rejected_total = 0
        self.exec_wall_total = 0.0
        Path(scratch_root).mkdir(parents=True, exist_ok=True)

    # -- admission ---------------------------------------------------------
    def _live_mem_mb(self) -> int:
        return sum(PROFILES[e.profile]["mem_mb"] for e in self.sandboxes.values())

    def create(self, profile: str = "B", sandbox_id: str | None = None) -> SandboxEntry:
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}")
        with self.lock:
            if self._live_mem_mb() + PROFILES[profile]["mem_mb"] > self.mem_cap_mb:
                self.rejected_total += 1
                raise AdmissionError("memory admission cap exceeded")
            sid = sandbox_id or f"sb-{uuid.uuid4().hex[:12]}"
            if sid in self.sandboxes:
                raise ValueError(f"sandbox {sid} already exists")
            workdir = os.path.join(self.scratch_root, sid)
            os.makedirs(workdir, exist_ok=False)
            os.chmod(workdir, 0o777)  # nobody must enter under setpriv
            entry = SandboxEntry(sid, workdir, profile)
            self.sandboxes[sid] = entry
            self.created_total += 1
            return entry

    def get(self, sandbox_id: str) -> SandboxEntry:
        with self.lock:
            if sandbox_id not in self.sandboxes:
                raise KeyError(sandbox_id)
            return self.sandboxes[sandbox_id]

    def stop(self, sandbox_id: str) -> bool:
        with self.lock:
            entry = self.sandboxes.pop(sandbox_id, None)
        if entry is None:
            return False
        proc = entry.proc
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        shutil.rmtree(entry.workdir, ignore_errors=True)
        return True

    def reap_idle(self) -> list[str]:
        """Stop sandboxes idle beyond ttl_s (no in-flight exec). Returns ids."""
        now = time.time()
        victims = []
        with self.lock:
            for sid, e in list(self.sandboxes.items()):
                if e.proc is None and now - e.last_used_ts > self.ttl_s:
                    victims.append(sid)
        return [sid for sid in victims if self.stop(sid)]

    def stats(self) -> dict:
        with self.lock:
            return {
                "active_sandboxes": len(self.sandboxes),
                "live_mem_mb": self._live_mem_mb(),
                "mem_cap_mb": self.mem_cap_mb,
                "created_total": self.created_total,
                "exec_total": self.exec_total,
                "rejected_total": self.rejected_total,
                "exec_wall_total_s": round(self.exec_wall_total, 3),
                "running_execs": sum(1 for e in self.sandboxes.values() if e.proc),
                "trajectories_active": sum(
                    1 for t in self.trajectories.values()
                    if t["proc"].poll() is None),
                "trajectories_total": len(self.trajectories),
            }

    # -- trajectories (agent_runner + grader subprocess, T29) --------------
    def start_trajectory(self, req: dict) -> str:
        traj_id = str(req["traj_id"])
        with self.lock:
            if traj_id in self.trajectories:
                raise ValueError(f"trajectory {traj_id} exists")
        run_root = req.get("run_root") or os.path.join(self.scratch_root, "runs")
        run_dir = os.path.join(run_root, traj_id)
        os.makedirs(run_dir, exist_ok=True)
        swe_file = os.path.join(run_dir, "swe.json")
        with open(swe_file, "w", encoding="utf-8") as f:
            json.dump(req["swe"], f)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cmd = [sys.executable, "-m", "rl_sim.trajectory_worker",
               "--adapter-url", req["adapter_url"],
               "--sandboxd-url", self.self_url,
               "--git-dir", req["git_dir"],
               "--venv", req.get("venv", ""),
               "--run-dir", run_dir,
               "--swe-file", swe_file,
               "--traj-id", traj_id,
               "--max-turns", str(req.get("max_turns", 40)),
               "--agent-timeout-s", str(req.get("agent_timeout_s", 1800)),
               "--grade-timeout-s", str(req.get("grade_timeout_s", 600))]
        log = open(os.path.join(run_dir, "worker.log"), "ab")
        proc = subprocess.Popen(cmd, cwd=repo_root, stdout=log,
                                stderr=subprocess.STDOUT,
                                start_new_session=True)
        with self.lock:
            self.trajectories[traj_id] = {
                "proc": proc, "run_dir": run_dir,
                "started_ts": time.time(), "log": log}
        return traj_id

    def trajectory_status(self, traj_id: str) -> dict:
        with self.lock:
            t = self.trajectories.get(traj_id)
        if t is None:
            raise KeyError(traj_id)
        rc = t["proc"].poll()
        result_file = os.path.join(t["run_dir"], "result.json")
        out = {"traj_id": traj_id,
               "started_ts": t["started_ts"],
               "elapsed_s": round(time.time() - t["started_ts"], 3)}
        if rc is None:
            out["state"] = "running"
        elif os.path.exists(result_file):
            out["state"] = "done"
            with open(result_file, encoding="utf-8") as f:
                out["result"] = json.load(f)
        else:
            out["state"] = "failed"
            out["rc"] = rc
            log_path = os.path.join(t["run_dir"], "worker.log")
            try:
                with open(log_path, encoding="utf-8") as f:
                    out["log_tail"] = f.read()[-2000:]
            except OSError:
                out["log_tail"] = ""
        return out

    def abort_trajectory(self, traj_id: str) -> bool:
        with self.lock:
            t = self.trajectories.get(traj_id)
        if t is None:
            return False
        proc = t["proc"]
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return True


# ---------------------------------------------------------------------------
# exec primitive: generic shell command under isolation + resource limits
# ---------------------------------------------------------------------------

def exec_in_sandbox(registry: SandboxRegistry, entry: SandboxEntry, cmd: str,
                    timeout_s: float | None = None,
                    env_extra: dict | None = None) -> dict:
    profile = PROFILES[entry.profile]
    mem_bytes = profile["mem_mb"] * 1024 * 1024
    fsize_bytes = profile["fsize_mb"] * 1024 * 1024
    wall = timeout_s if timeout_s is not None else profile["wall_s"]
    argv = isolation_prefix(True) + [
        "prlimit",
        f"--as={mem_bytes}",
        f"--cpu={profile['cpu_s']}:{profile['cpu_s']}",
        f"--fsize={fsize_bytes}",
        "--nofile=1024:1024",  # pytest needs more than the 64 used for A-tier
        "bash", "-c", cmd,
    ]
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": entry.workdir,
           "PYTHONDONTWRITEBYTECODE": "1"}
    if env_extra:
        env.update(env_extra)

    with registry.lock:
        running = sum(1 for e in registry.sandboxes.values() if e.proc)
        if running >= registry.max_concurrent_exec:
            registry.rejected_total += 1
            raise AdmissionError("exec concurrency cap exceeded")

    ru_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    spawn = time.time()
    proc = subprocess.Popen(argv, cwd=entry.workdir, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    entry.proc = proc
    timed_out = False
    try:
        out, err = proc.communicate(timeout=wall)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
    wall_s = time.time() - spawn
    entry.proc = None
    entry.last_used_ts = time.time()
    entry.exec_count += 1

    ru_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_s = round((ru_after.ru_utime - ru_before.ru_utime)
                  + (ru_after.ru_stime - ru_before.ru_stime), 4)

    with registry.lock:
        registry.exec_total += 1
        registry.exec_wall_total += wall_s

    def _s(x) -> str:
        if x is None:
            return ""
        return x if isinstance(x, str) else x.decode("utf-8", "replace")

    return {
        "rc": proc.returncode,
        "stdout": _s(out)[-65536:],   # cap payload; agent_runner truncates further
        "stderr": _s(err)[-16384:],
        "wall_s": round(wall_s, 4),
        "cpu_s": cpu_s,
        # RUSAGE_CHILDREN maxrss is a running max across all reaped children —
        # a lower-bound-free upper indicator, documented in design 09 F-list.
        "maxrss_kb": ru_after.ru_maxrss,
        "timed_out": timed_out,
    }


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

def _read_cpu_times() -> tuple[float, float]:
    """(busy, total) jiffies from /proc/stat."""
    with open("/proc/stat", encoding="utf-8") as f:
        parts = f.readline().split()[1:]
    vals = [float(x) for x in parts]
    idle = vals[3] + vals[4]  # idle + iowait
    total = sum(vals)
    return total - idle, total


def read_sys_stats(sample_s: float = 0.1) -> dict:
    """Node-level CPU busy% (short delta), load, memory."""
    b0, t0 = _read_cpu_times()
    time.sleep(sample_s)
    b1, t1 = _read_cpu_times()
    busy_pct = 100.0 * (b1 - b0) / max(1.0, t1 - t0)
    mem: dict[str, float] = {}
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            k, _, rest = line.partition(":")
            if k in ("MemTotal", "MemAvailable"):
                mem[k] = float(rest.split()[0]) / 1024.0  # kB -> MB
    with open("/proc/loadavg", encoding="utf-8") as f:
        load1 = float(f.read().split()[0])
    return {
        "ts": time.time(),
        "cpu_busy_pct": round(busy_pct, 2),
        "load1": load1,
        "mem_total_mb": mem.get("MemTotal", 0.0),
        "mem_used_mb": round(mem.get("MemTotal", 0.0) - mem.get("MemAvailable", 0.0), 1),
    }


def make_handler(registry: SandboxRegistry):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def log_message(self, *a):  # quiet; stats endpoint carries the numbers
            pass

        def do_GET(self):
            if self.path == "/healthz":
                return self._json(200, {"ok": True})
            if self.path == "/stats":
                return self._json(200, registry.stats())
            if self.path == "/sysstats":
                return self._json(200, read_sys_stats())
            if self.path.startswith("/trajectory/") and self.path.endswith("/status"):
                traj_id = self.path.split("/")[2]
                try:
                    return self._json(200, registry.trajectory_status(traj_id))
                except KeyError:
                    return self._json(404, {"error": f"unknown trajectory {traj_id}"})
            return self._json(404, {"error": "unknown path"})

        def do_POST(self):
            try:
                req = self._body()
            except json.JSONDecodeError:
                return self._json(400, {"error": "bad json"})
            try:
                if self.path == "/sandbox/create":
                    e = registry.create(req.get("profile", "B"),
                                        req.get("sandbox_id"))
                    return self._json(200, {"sandbox_id": e.sandbox_id,
                                            "workdir": e.workdir})
                if self.path == "/sandbox/exec":
                    e = registry.get(req["sandbox_id"])
                    return self._json(200, exec_in_sandbox(
                        registry, e, req["cmd"],
                        timeout_s=req.get("timeout_s"),
                        env_extra=req.get("env")))
                if self.path == "/sandbox/stop":
                    return self._json(200, {"stopped": registry.stop(req["sandbox_id"])})
                if self.path == "/trajectory/start":
                    traj_id = registry.start_trajectory(req)
                    return self._json(200, {"traj_id": traj_id, "state": "running"})
                if self.path == "/trajectory/abort":
                    return self._json(200,
                                      {"aborted": registry.abort_trajectory(req["traj_id"])})
                return self._json(404, {"error": "unknown path"})
            except KeyError as ke:
                return self._json(404, {"error": f"unknown sandbox {ke}"})
            except AdmissionError as ae:
                return self._json(429, {"error": str(ae)})
            except (ValueError, OSError) as ve:
                return self._json(400, {"error": str(ve)})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=18101)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--scratch", default=DEFAULT_SCRATCH)
    ap.add_argument("--max-concurrent-exec", type=int, default=32)
    ap.add_argument("--mem-cap-mb", type=int, default=60000)
    ap.add_argument("--ttl-s", type=float, default=600.0)
    args = ap.parse_args()

    registry = SandboxRegistry(args.scratch, args.max_concurrent_exec,
                               args.mem_cap_mb, args.ttl_s,
                               self_url=f"http://127.0.0.1:{args.port}")

    def reaper():
        while True:
            time.sleep(max(1.0, args.ttl_s / 4))
            for sid in registry.reap_idle():
                print(f"[sandboxd] ttl reaped {sid}", flush=True)

    threading.Thread(target=reaper, daemon=True).start()
    srv = ThreadingHTTPServer((args.bind, args.port), make_handler(registry))
    print(f"[sandboxd] listening on {args.bind}:{args.port} "
          f"scratch={args.scratch} cap={args.mem_cap_mb}MB", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
