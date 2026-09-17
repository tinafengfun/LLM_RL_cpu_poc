"""Tokenizer wrapper ← TITO-style server-side tokenization.

Real BPE happens inside llama.cpp (server-side, like TITO gateway: the trainer
sees exactly the tokens the policy sampled). Driver-side we call the engine's
/tokenize endpoint; a whitespace fallback exists ONLY for offline smoke tests
and is explicitly flagged as non-representative.
"""

from __future__ import annotations

import json
import time
import urllib.request


class Tokenizer:
    """Driver-side tokenizer client with per-call timing."""

    def __init__(self, base_url: str | None = None, timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout_s = timeout_s
        self.calls = 0
        self.spent_s = 0.0

    def tokenize(self, text: str) -> list[int]:
        if self.base_url is None:
            return self._fallback(text)
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                self.base_url + "/tokenize",
                data=json.dumps({"content": text}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode())
            return [t["id"] if isinstance(t, dict) else t for t in data["tokens"]]
        finally:
            self.calls += 1
            self.spent_s += time.perf_counter() - t0

    def detokenize(self, tokens: list[int]) -> str:
        if self.base_url is None:
            raise RuntimeError("fallback tokenizer cannot detokenize")
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                self.base_url + "/detokenize",
                data=json.dumps({"tokens": tokens}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode())["content"]
        finally:
            self.calls += 1
            self.spent_s += time.perf_counter() - t0

    @staticmethod
    def _fallback(text: str) -> list[int]:
        """Whitespace pseudo-tokens (NOT real BPE; smoke tests only)."""
        return [hash(w) % 50000 for w in text.split()]

    def count(self, text: str) -> int:
        return len(self.tokenize(text))

    def stats(self) -> dict:
        return {"calls": self.calls, "spent_s": round(self.spent_s, 4),
                "backend": "server" if self.base_url else "fallback"}
