"""HTTP client for sandboxd ← libdsec SDK 微缩 (design 09 section 5.2).

Used by agent_runner/grader on node2 (localhost) and by rollout_manager on
node1 (cross-node, T29). Same code path both ways — only latency differs.
"""

from __future__ import annotations

import urllib.error

from rl_sim import netutil


class SandboxdError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"HTTP {code}: {message}")
        self.code = code


class SandboxHTTPClient:
    def __init__(self, base_url: str, timeout_s: float = 660.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _post(self, path: str, obj: dict, timeout_s: float | None = None) -> dict:
        try:
            return netutil.post_json(self.base_url + path, obj,
                                     timeout_s or self.timeout_s)
        except urllib.error.HTTPError as e:
            raise SandboxdError(e.code, e.read().decode("utf-8", "replace")) from e

    def create(self, profile: str = "B", sandbox_id: str | None = None,
               ttl_s: float | None = None) -> dict:
        return self._post("/sandbox/create",
                          {"profile": profile, "sandbox_id": sandbox_id,
                           "ttl_s": ttl_s})

    def exec(self, sandbox_id: str, cmd: str, timeout_s: float | None = None,
             env: dict | None = None) -> dict:
        # client timeout must exceed sandbox wall timeout
        return self._post("/sandbox/exec",
                          {"sandbox_id": sandbox_id, "cmd": cmd,
                           "timeout_s": timeout_s, "env": env},
                          timeout_s=(timeout_s or 300) + 60)

    def stop(self, sandbox_id: str) -> bool:
        return bool(self._post("/sandbox/stop", {"sandbox_id": sandbox_id})["stopped"])

    def stats(self) -> dict:
        return netutil.get_json(self.base_url + "/stats", self.timeout_s)

    def healthy(self) -> bool:
        try:
            r = netutil.get_json(self.base_url + "/healthz", 5)
            return bool(r.get("ok"))
        except Exception:
            return False
