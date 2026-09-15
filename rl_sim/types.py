"""Sample / SampleStatus / TaskLine ← slime/utils/types.py.

Core data unit flowing through the whole RL loop:
data_source -> rollout_manager -> engine -> sandbox -> reward -> trainer.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field


class SampleStatus(enum.Enum):
    """Mirrors slime's Sample status set, extended with sandbox outcomes."""

    PENDING = "pending"
    COMPLETED = "completed"  # rollout + reward finished (pass or fail)
    TRUNCATED = "truncated"  # hit max response length
    ABORTED = "aborted"  # aborted mid-generation (partial rollout candidate)
    GEN_ERROR = "gen_error"  # engine failure
    SANDBOX_ERROR = "sandbox_error"  # syntax error / crash before tests
    SANDBOX_TIMEOUT = "sandbox_timeout"
    SANDBOX_OOM = "sandbox_oom"


class TaskLine(enum.Enum):
    """Task lines from design doc v3.1 section 2."""

    A_CODE = "A"  # single-step code execution
    L1_SWE = "L1"  # multi-file repo repair (long-horizon)
    L2_TERM = "L2"  # terminal / bash pipeline
    L3_SEARCH = "L3"  # multi-hop QA over local corpus
    V1_CHART = "V1"  # chart / document QA (multimodal)
    V2_GEOM = "V2"  # visual geometry (multimodal)
    MOCK = "MOCK"  # synthetic fallback


@dataclass
class Sample:
    """One rollout sample; a group shares (task_id, group_id) with n samples per prompt."""

    prompt: str
    task_id: str
    task_line: str = TaskLine.A_CODE.value
    group_id: int = -1

    # generation (filled by engine)
    response: str = ""
    code: str = ""
    image_path: str | None = None  # multimodal input
    status: SampleStatus = SampleStatus.PENDING

    # reward (filled by reward.py)
    reward: float = 0.0
    advantage: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0

    # training data (logprobs are REAL from llama.cpp; train_logprob = Q8 rescore)
    num_tokens: int = 0
    rollout_logprob: float = 0.0
    train_logprob: float | None = None
    weight_version: int = 0  # policy version at generation time (staleness tracking)

    turns: int = 0  # agentic rounds used (L1/L2/L3 multi-turn)

    # sandbox profiling: submit_ts/start_ts/end_ts, queue_wait_s, exec_wall_s,
    # exec_cpu_s, launch_overhead_s, maxrss_kb, io_read_bytes, io_write_bytes
    sandbox: dict = field(default_factory=dict)

    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        d = dict(d)
        d["status"] = SampleStatus(d["status"])
        return cls(**d)
