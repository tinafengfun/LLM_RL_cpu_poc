"""HTTP client for sandboxd ← libdsec SDK 微缩 (design 09 section 5.2).

Used by agent_runner/grader on node2 (localhost) and by rollout_manager on
node1 (cross-node, T29). Same code path both ways — only latency differs.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class SandboxdError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"HTTP {code}: {message}")
        self.code = code


class SandboxHTTPClient:
    def __init__(self, base_url: str, timeout_s: float = 660.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _post(self, path: str, obj: dict, timeout_s: float | None = None) -> dict:
        req = urllib.request.Request(
            self.base_url + path, data=json.dumps(obj).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(
                    req, timeout=timeout_s or self.timeout_s) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise SandboxdError(e.code, e.read().decode("utf-8", "replace")) from e

    def create(self, profile: str = "B", sandbox_id: str | None = None) -> dict:
        return self._post("/sandbox/create",
                          {"profile": profile, "sandbox_id": sandbox_id})

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
        with urllib.request.urlopen(self.base_url + "/stats",
                                    timeout=self.timeout_s) as r:
            return json.loads(r.read())

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/healthz", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False
