"""adapter service: node2 agent_runner ↔ node1 inference ← slime coding_agent_rl adapter.

The bridge that makes the dual-node split real: agent loops on node2 call
`POST /generate`; the adapter forwards to the router (session-affinity by
traj_id → prefix-cache reuse), and — the point of the whole exercise —
RETURNS ONLY TEXT to the caller while retaining token-level data on node1:
per-token logprobs + token strings land in adapter_log/<traj_id>.jsonl with
loss_mask semantics (generated tokens carry logprobs = trainable; tool
observations never pass through here, so they are implicitly mask=0).

API:
  POST /generate {traj_id, turn, messages, tools?, max_tokens?, temperature?}
      -> {text, tool_calls, finish_reason, usage, backend, adapter_lat_s}
  GET  /healthz -> {"ok": true}
  GET  /stats   -> {calls, tokens_out, lat_s: [...], backends: {...}}
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class TrajectoryLogger:
    """Append-only JSONL per trajectory; fsync-free (profiling, not WAL)."""

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def log_call(self, traj_id: str, record: dict) -> None:
        with self._guard:
            lock = self._locks.setdefault(traj_id, threading.Lock())
        with lock:
            with open(self.log_dir / f"{traj_id}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


class Adapter:
    def __init__(self, router, log_dir: str | Path = "adapter_log",
                 temperature: float = 1.0, max_tokens: int = 4096,
                 top_p: float | None = None) -> None:
        self.router = router
        self.logger = TrajectoryLogger(log_dir)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.calls = 0
        self.tokens_out = 0
        self.lat_s: list[float] = []
        self._lock = threading.Lock()

    def generate(self, req: dict) -> dict:
        traj_id = str(req["traj_id"])
        payload = {
            "model": "rl_sim",
            "messages": req["messages"],
            "temperature": req.get("temperature", self.temperature),
            "max_tokens": req.get("max_tokens", self.max_tokens),
            "logprobs": True,
            "top_logprobs": 1,
            "chat_template_kwargs": {"enable_thinking": False},
            "session_key": traj_id,
        }
        if req.get("tools"):
            payload["tools"] = req["tools"]
        if self.top_p is not None:
            payload["top_p"] = self.top_p

        t0 = time.perf_counter()
        out = self.router.forward("/v1/chat/completions", payload, traj_id)
        lat = time.perf_counter() - t0

        choice = out["choices"][0]
        msg = choice.get("message") or {}
        toks = (choice.get("logprobs") or {}).get("content") or []
        usage = out.get("usage") or {}
        backend = out.get("_backend", "")

        self.logger.log_call(traj_id, {
            "traj_id": traj_id,
            "turn": req.get("turn", -1),
            "ts": time.time(),
            "backend": backend,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
            # token-level training data (loss_mask=1 by construction)
            "out_tokens": [{"t": t.get("token", ""), "lp": t.get("logprob", 0.0)}
                           for t in toks],
        })

        with self._lock:
            self.calls += 1
            self.tokens_out += len(toks)
            self.lat_s.append(lat)

        return {
            "text": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or [],
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "backend": backend,
            "adapter_lat_s": round(lat, 4),
        }

    def stats(self) -> dict:
        with self._lock:
            return {"calls": self.calls, "tokens_out": self.tokens_out,
                    "lat_s": list(self.lat_s),
                    "backends": dict(self.router.ring.counts)}


def make_handler(adapter: Adapter):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/healthz":
                return self._json(200, {"ok": True})
            if self.path == "/stats":
                return self._json(200, adapter.stats())
            return self._json(404, {"error": "unknown path"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "bad json"})
            if self.path != "/generate":
                return self._json(404, {"error": "unknown path"})
            try:
                return self._json(200, adapter.generate(req))
            except KeyError as ke:
                return self._json(400, {"error": f"missing field {ke}"})
            except urllib.error.HTTPError as e:
                # surface the backend's own error body (llama.cpp puts the
                # cause there, e.g. "exceeds the available context size")
                body = e.read().decode("utf-8", "replace")[:300]
                return self._json(502, {"error": f"backend HTTP {e.code}: {body}"})
            except Exception as e:  # noqa: BLE001 - engine/router failure
                return self._json(502, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def main() -> None:
    from rl_sim.router import Router

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=18090)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--log-dir", default="adapter_log")
    ap.add_argument("--backend", action="append", default=[],
                    help="name=url of a llama-server instance; repeatable")
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    router = Router(timeout_s=600.0)
    for b in args.backend:
        name, _, url = b.partition("=")
        router.ring.add(name, url)
    adapter = Adapter(router, log_dir=args.log_dir, max_tokens=args.max_tokens)
    srv = ThreadingHTTPServer((args.bind, args.port), make_handler(adapter))
    print(f"[adapter] listening on {args.bind}:{args.port} "
          f"backends={len(args.backend)}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
