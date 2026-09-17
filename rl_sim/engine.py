"""RolloutEngine protocol + MockEngine + APIEngine ← slime/backends/sglang_utils/sglang_engine.py.

Three engines behind one protocol:
  - LocalEngine (T11, [node]): llama.cpp llama-server instances = rollout DP ranks (real logprobs)
  - APIEngine  (fallback):     remote OpenAI-compatible chat completions
  - MockEngine (smoke):        offline templates; correctness probability rises with
                               weight_version purely to drive control flow (filtering,
                               abort, version bumps) — reward VALUES are meaningless
                               by design (design v3.1: fake loop OK, fake load NOT OK).

Pause/continue semantics mirror slime's update_weights flow: generation calls
block while the engine is paused (weight sync window).
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
import urllib.request
import urllib.error

from rl_sim.data_source import Task
from rl_sim.types import Sample, SampleStatus

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

# simple corruption mutations for MockEngine's "wrong" samples
_MUTATIONS = [
    ("==", "!="), ("<", "<="), (" + ", " - "), ("[::-1]", "[::1]"),
    ("range(1, n + 1)", "range(1, n)"), ("s.lower()", "s"),
]


def mutate(code: str, rng: random.Random) -> str:
    rng = random.Random(rng.random())  # derive stream
    cands = [(p, r) for p, r in _MUTATIONS if p in code]
    if cands:
        p, r = cands[rng.randrange(len(cands))]
        return code.replace(p, r, 1)
    return "def " + code.split("def ", 1)[-1].split("(", 1)[0] + "(*a):\n    return None\n"


def extract_code(text: str) -> str:
    m = _CODE_BLOCK.search(text)
    return (m.group(1) if m else text).strip()


class EngineError(Exception):
    pass


class Engine:
    """Common engine state: weight version + pause gate."""

    kind = "abstract"

    def __init__(self) -> None:
        self._weight_version = 0
        self._pause = threading.Event()
        self._pause.set()  # set = generating allowed

    @property
    def weight_version(self) -> int:
        return self._weight_version

    def pause_generation(self) -> None:
        self._pause.clear()

    def continue_generation(self) -> None:
        self._pause.set()

    def update_weights(self, version: int) -> None:
        self._weight_version = version

    # subclass hook
    def generate(self, sample: Sample, task: Task) -> Sample:
        raise NotImplementedError


class MockEngine(Engine):
    """Offline engine: emits task.solution or a mutated variant."""

    kind = "mock"

    def __init__(self, tasks_by_id: dict[str, Task], seed: int = 0,
                 base_correct: float = 0.15, slope: float = 0.08,
                 latency_s: float = 0.0) -> None:
        super().__init__()
        self._tasks = tasks_by_id
        self._rng = random.Random(seed)
        self.base_correct = base_correct
        self.slope = slope
        self.latency_s = latency_s

    def correct_prob(self, version: int) -> float:
        return min(0.95, self.base_correct + self.slope * version)

    def generate(self, sample: Sample, task: Task | None = None) -> Sample:
        self._pause.wait()  # weight-sync window: generation blocks while paused
        task = task or self._tasks[sample.task_id]
        if self.latency_s:
            time.sleep(self.latency_s)
        if not task.solution:
            sample.status = SampleStatus.GEN_ERROR
            sample.metadata["error"] = "no solution template for mock engine"
            return sample
        if self._rng.random() < self.correct_prob(self._weight_version):
            code = task.solution
        else:
            code = mutate(task.solution, self._rng)
        sample.response = f"```python\n{code}```"
        sample.code = code
        sample.num_tokens = len(sample.response.split())
        # fake per-token logprobs: mean -0.5 with noise (see class docstring)
        sample.rollout_logprob = sum(self._rng.gauss(-0.5, 0.15) for _ in range(sample.num_tokens))
        sample.weight_version = self._weight_version
        sample.turns = 1
        sample.status = SampleStatus.PENDING  # sandbox will finalize
        return sample


class APIEngine(Engine):
    """Remote OpenAI-compatible chat completions (env-configured)."""

    kind = "api"

    def __init__(self, base_url: str, api_key: str = "", model: str = "",
                 timeout_s: float = 120.0, max_retries: int = 3,
                 latency_floor_s: float = 0.0) -> None:
        super().__init__()
        if not base_url:
            raise ValueError("base_url required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.latency_floor_s = latency_floor_s

    @classmethod
    def from_env(cls) -> "APIEngine":
        import os
        return cls(
            base_url=os.environ.get("RL_SIM_API_BASE", ""),
            api_key=os.environ.get("RL_SIM_API_KEY", ""),
            model=os.environ.get("RL_SIM_API_MODEL", ""),
        )

    def generate(self, sample: Sample, task: Task | None = None) -> Sample:
        self._pause.wait()
        try:
            content = self._chat(sample.prompt, sample.image_path)
        except Exception as e:  # noqa: BLE001 - engine boundary must not crash the loop
            sample.status = SampleStatus.GEN_ERROR
            sample.metadata["error"] = f"{type(e).__name__}: {e}"
            return sample
        sample.response = content
        sample.code = extract_code(content)
        sample.num_tokens = len(content.split())
        # NOTE: most chat APIs do not return per-token logprobs; slime gets real
        # ones from SGLang return_logprob. LocalEngine (T11) provides real values;
        # here we mark them as unavailable-ish mock values.
        sample.rollout_logprob = -0.5 * sample.num_tokens
        sample.weight_version = self._weight_version
        sample.turns = 1
        sample.status = SampleStatus.PENDING
        return sample

    def _chat(self, prompt: str, image_path: str | None = None) -> str:
        content: str | list = prompt
        if image_path:
            import base64
            b64 = base64.b64encode(open(image_path, "rb").read()).decode()
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 1.0,
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
        )
        delay = 1.0
        last_err: Exception | None = None
        for _ in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"]
            except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as e:
                last_err = e
                time.sleep(delay)
                delay *= 2
        raise EngineError(f"api failed after {self.max_retries} tries: {last_err}")
