"""LocalEngine ← slime/backends/sglang_utils/sglang_engine.py (SGLangEngine actor).

Manages llama.cpp `llama-server` instances as rollout DP ranks:
- N instances, each taskset-pinned to a core set, `--parallel` slots, `--jinja`
  chat templates (tool calling), per-token logprobs via OpenAI API.
- ServerManager: launch / health-check / restart / stop  ← RolloutHealthMonitor.
- Real per-token rollout logprobs (Q4). Trainer-side rescoring uses the
  sequence-level (GSPO-style) ratio via an offline Q8 scorer — llama.cpp has no
  prompt_logprobs endpoint, so per-token TIS rescoring is out of scope.
"""

from __future__ import annotations

import base64
import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from rl_sim.data_source import Task
from rl_sim.engine import Engine, extract_code
from rl_sim.types import Sample, SampleStatus


@dataclass
class InstanceSpec:
    name: str
    port: int
    cores: str = ""  # taskset list, e.g. "0-15"; empty = no pinning
    slots: int = 8
    model: str = ""
    mmproj: str = ""  # multimodal projection (VLM instances)
    extra: list[str] = field(default_factory=list)


class LlamaServerManager:
    """Lifecycle for llama-server processes (one per rollout DP rank)."""

    def __init__(self, binary: str, specs: list[InstanceSpec],
                 log_dir: str | Path = "logs/engines") -> None:
        self.binary = binary
        self.specs = {s.name: s for s in specs}
        self.log_dir = Path(log_dir)
        self._procs: dict[str, subprocess.Popen] = {}

    def build_cmd(self, spec: InstanceSpec) -> list[str]:
        cmd = []
        if spec.cores:
            cmd += ["taskset", "-c", spec.cores]
        cmd += [self.binary, "--model", spec.model, "--port", str(spec.port),
                "--parallel", str(spec.slots), "--jinja", "--log-disable"]
        if spec.mmproj:
            cmd += ["--mmproj", spec.mmproj]
        cmd += spec.extra
        return cmd

    def start(self, name: str) -> None:
        spec = self.specs[name]
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.log_dir / f"{name}.log", "ab")
        self._procs[name] = subprocess.Popen(self.build_cmd(spec),
                                             stdout=log, stderr=subprocess.STDOUT)

    def start_all(self) -> None:
        for name in self.specs:
            self.start(name)

    def health(self, name: str) -> bool:
        port = self.specs[name].port
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def wait_healthy(self, timeout_s: float = 300.0) -> bool:
        deadline = time.time() + timeout_s
        pending = set(self.specs)
        while pending and time.time() < deadline:
            for name in list(pending):
                proc = self._procs.get(name)
                if proc is not None and proc.poll() is not None:
                    raise RuntimeError(f"engine {name} exited rc={proc.returncode}, see log")
                if self.health(name):
                    pending.discard(name)
            if pending:
                time.sleep(1.0)
        return not pending

    def restart(self, name: str) -> None:
        self.stop(name)
        self.start(name)

    def stop(self, name: str) -> None:
        proc = self._procs.pop(name, None)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    def stop_all(self) -> None:
        for name in list(self._procs):
            self.stop(name)

    def urls(self) -> dict[str, str]:
        return {n: f"http://127.0.0.1:{s.port}" for n, s in self.specs.items()}


class LocalEngine(Engine):
    """Talks to llama-server instances (directly or via router)."""

    kind = "local"

    def __init__(self, backend_urls: dict[str, str], router=None,
                 timeout_s: float = 300.0, temperature: float = 1.0,
                 max_tokens: int = 2048) -> None:
        super().__init__()
        if not backend_urls:
            raise ValueError("backend_urls required")
        self.backends = backend_urls
        self.router = router
        for name, url in backend_urls.items():
            if router is not None:
                router.ring.add(name, url)
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        # client-side engine profile (server /metrics needs a flag; this is free)
        self.req_count = 0
        self.tok_generated = 0
        self.req_lat_s: list[float] = []

    def engine_stats(self) -> dict:
        return {"requests": self.req_count, "tokens": self.tok_generated,
                "lat": list(self.req_lat_s), "backends": len(self.backends)}

    def generate(self, sample: Sample, task: Task | None = None) -> Sample:
        self._pause.wait()
        session_key = sample.metadata.get("session_id") or f"g{sample.group_id}"
        t0 = time.perf_counter()
        try:
            content, logprob_sum, ntok = self._chat(sample, session_key)
        except Exception as e:  # noqa: BLE001
            sample.status = SampleStatus.GEN_ERROR
            sample.metadata["error"] = f"{type(e).__name__}: {e}"
            return sample
        finally:
            self.req_count += 1
            self.req_lat_s.append(time.perf_counter() - t0)
        self.tok_generated += ntok
        sample.response = content
        sample.code = extract_code(content)
        sample.num_tokens = ntok
        sample.rollout_logprob = logprob_sum  # REAL per-token logprobs
        sample.weight_version = self._weight_version
        sample.turns = 1
        sample.status = SampleStatus.PENDING
        return sample

    def _chat(self, sample: Sample, session_key: str) -> tuple[str, float, int]:
        content = sample.prompt
        if sample.image_path:
            b64 = base64.b64encode(Path(sample.image_path).read_bytes()).decode()
            content = [{"type": "text", "text": sample.prompt},
                       {"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}}]
        payload = {
            "model": "rl_sim",
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "logprobs": True,
            "top_logprobs": 1,
            # Qwen3 thinks by default and would burn the whole budget on
            # reasoning_content; disable for RL rollout-style short answers.
            "chat_template_kwargs": {"enable_thinking": False},
            "session_key": session_key,  # consumed by router for affinity
        }
        if self.router is not None:
            out = self.router.forward("/v1/chat/completions", payload, session_key)
        else:
            base = next(iter(self.backends.values()))
            req = urllib.request.Request(
                base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                out = json.loads(resp.read().decode())
        choice = out["choices"][0]
        text = choice["message"]["content"] or ""
        toks = (choice.get("logprobs") or {}).get("content") or []
        logprob_sum = sum(t.get("logprob", 0.0) for t in toks)
        return text, logprob_sum, len(toks) or len(text.split())


def aggregate_logprobs(content_logprobs: list[dict]) -> tuple[float, int]:
    """(sum of per-token logprobs, token count) from OpenAI-style logprobs.content."""
    return (sum(t.get("logprob", 0.0) for t in content_logprobs), len(content_logprobs))
