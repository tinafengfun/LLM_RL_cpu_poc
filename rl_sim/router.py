"""HTTP router with session affinity ← SGLang router / GLM-5 TITO gateway / DP-aware routing.

Rendezvous (HRW) hashing: pick the alive backend with max hash(backend, session_key).
- same session key -> same backend (prefix-cache reuse, GLM DP-aware routing);
- backend death -> re-pick among alive ones only; affinity of the rest untouched.

Stdlib ThreadingHTTPServer proxy; forwarding + JSON (de)serialization cost is
real CPU load and measured by the driver-side monitor.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class BackendRing:
    """Alive-backend registry + rendezvous hashing + failure stats."""

    def __init__(self) -> None:
        self._alive: dict[str, str] = {}  # name -> base_url
        self.counts: dict[str, int] = {}
        self.failures: dict[str, int] = {}
        self._lock = threading.Lock()

    def add(self, name: str, base_url: str) -> None:
        with self._lock:
            self._alive[name] = base_url
            self.counts.setdefault(name, 0)
            self.failures.setdefault(name, 0)

    def remove(self, name: str) -> None:
        with self._lock:
            self._alive.pop(name, None)

    def alive(self) -> dict[str, str]:
        with self._lock:
            return dict(self._alive)

    @staticmethod
    def _h(*parts: str) -> int:
        return int(hashlib.md5(":".join(parts).encode()).hexdigest(), 16)

    def pick(self, session_key: str) -> tuple[str, str] | None:
        with self._lock:
            if not self._alive:
                return None
            name = max(self._alive, key=lambda b: self._h(b, session_key))
            self.counts[name] += 1
            return name, self._alive[name]

    def mark_failure(self, name: str) -> None:
        with self._lock:
            self.failures[name] = self.failures.get(name, 0) + 1
            self._alive.pop(name, None)


class Router:
    """In-process router (testable) + optional HTTP front."""

    def __init__(self, timeout_s: float = 60.0) -> None:
        self.ring = BackendRing()
        self.timeout_s = timeout_s
        self.forward_lat_s: list[float] = []
        self._lat_lock = threading.Lock()

    def forward(self, path: str, payload: dict, session_key: str) -> dict:
        """POST payload to the affinity backend; fail over once on error."""
        t0 = time.perf_counter()
        try:
            return self._forward_inner(path, payload, session_key)
        finally:
            with self._lat_lock:
                self.forward_lat_s.append(time.perf_counter() - t0)

    def _forward_inner(self, path: str, payload: dict, session_key: str) -> dict:
        last_err: Exception | None = None
        for _ in range(2):  # affinity backend, then one failover
            picked = self.ring.pick(session_key)
            if picked is None:
                raise RuntimeError("router: no alive backends")
            name, base = picked
            try:
                req = urllib.request.Request(
                    base + path, data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    out = json.loads(resp.read().decode())
                out["_backend"] = name
                return out
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_err = e
                self.ring.mark_failure(name)
        raise RuntimeError(f"router: all backends failed: {last_err}")


class _Handler(BaseHTTPRequestHandler):
    router: Router  # injected by serve()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            key = self.headers.get("X-Session-Key") or payload.get("session_key", "")
            out = self.router.forward(self.path, payload, session_key=key)
            body = json.dumps(out).encode()
            self.send_response(200)
        except Exception as e:  # noqa: BLE001 - router must not crash on bad requests
            body = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
            self.send_response(502)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def serve(router: Router, host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start router HTTP front in a daemon thread; returns (server, thread)."""
    handler = type("BoundHandler", (_Handler,), {"router": router})
    srv = ThreadingHTTPServer((host, port), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, t
