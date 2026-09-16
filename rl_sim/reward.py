"""Rule-based reward functions ← slime/rollout/rm_hub (deepscaler/dapo style).

Three verifier types covering all task lines (design v3.1 section 2):
  - "tests":   partial credit = tests_passed / tests_total (A / L1 / L2)
  - "numeric": extracted numeric answer matches label (V1 / math-style)
  - "exact":   normalized string match (L3 answers)

Any non-COMPLETED sandbox/gen outcome scores 0.0 — failed trajectories still
flow into group advantage as zero-reward samples (mirrors slime behavior).
"""

from __future__ import annotations

import re

from rl_sim.types import Sample, SampleStatus

_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def reward_from_tests(sample: Sample) -> float:
    """Partial credit from sandbox test outcomes."""
    if sample.status is not SampleStatus.COMPLETED or sample.tests_total <= 0:
        return 0.0
    return sample.tests_passed / sample.tests_total


def extract_number(text: str) -> float | None:
    """Prefer \\boxed{...}; else last number in text. None if nothing found."""
    boxed = _BOXED.findall(text)
    if boxed:
        return _to_float(boxed[-1])
    nums = _NUMBER.findall(text)
    if not nums:
        return None
    return _to_float(nums[-1])


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", "").strip().rstrip("."))
    except ValueError:
        return None


def numeric_match(response: str, answer: float, rel_tol: float = 1e-4) -> bool:
    got = extract_number(response)
    if got is None:
        return False
    if answer == 0:
        return abs(got) <= rel_tol
    return abs(got - answer) / abs(answer) <= rel_tol


def normalize_text(s: str) -> str:
    return " ".join(s.strip().lower().split())


def exact_match(response: str, answer: str) -> bool:
    """Normalized containment: answer string appears in (normalized) response tail."""
    resp = normalize_text(response)
    ans = normalize_text(answer)
    return ans in resp[-max(len(ans) * 4, 64):] if ans else False


def compute_reward(sample: Sample, spec: dict) -> float:
    """Dispatch by spec['type']: 'tests' | 'numeric' | 'exact'."""
    rtype = spec.get("type", "tests")
    if rtype == "tests":
        return reward_from_tests(sample)
    if sample.status is SampleStatus.GEN_ERROR:
        return 0.0
    if rtype == "numeric":
        return 1.0 if numeric_match(sample.response, float(spec["answer"])) else 0.0
    if rtype == "exact":
        return 1.0 if exact_match(sample.response, str(spec["answer"])) else 0.0
    raise ValueError(f"unknown reward type: {rtype}")
