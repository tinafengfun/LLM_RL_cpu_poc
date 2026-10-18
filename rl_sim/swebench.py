"""SWE-bench task line ← design doc 09 section 6.

Loads SWE-bench(-Verified) style jsonl instances and maps them to
data_source Tasks (line=TaskLine.SWE). Execution happens on node2
(sandboxd / agent_runner / grader); this module is pure parsing and
mapping, so it runs identically on dev, node1 and node2.

Real SWE-bench rows carry FAIL_TO_PASS / PASS_TO_PASS as JSON-encoded
strings; we accept both that and plain lists. `patch` (the gold patch) is
kept for sanity checks only and is never shown to the model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from rl_sim.data_source import Task
from rl_sim.types import TaskLine


@dataclass
class SWEInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    version: str = ""
    gold_patch: str = ""  # sanity-check only, NOT shown to the model


def _as_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return []
        parsed = json.loads(v)
        if not isinstance(parsed, list):
            raise ValueError(f"expected JSON list, got {type(parsed).__name__}")
        return [str(x) for x in parsed]
    raise ValueError(f"cannot coerce {type(v).__name__} to list")


def parse_instance(row: dict) -> SWEInstance:
    return SWEInstance(
        instance_id=str(row["instance_id"]),
        repo=str(row["repo"]),
        base_commit=str(row["base_commit"]),
        problem_statement=str(row.get("problem_statement", "")),
        test_patch=str(row.get("test_patch", "")),
        fail_to_pass=_as_list(row.get("FAIL_TO_PASS")),
        pass_to_pass=_as_list(row.get("PASS_TO_PASS")),
        version=str(row.get("version", "")),
        gold_patch=str(row.get("patch", "")),
    )


def load_instances(path: str, repo: str | None = None,
                   limit: int | None = None) -> list[SWEInstance]:
    """Read a SWE-bench jsonl; optionally filter to one repo and cap count."""
    out: list[SWEInstance] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            inst = parse_instance(json.loads(line))
            if repo is not None and inst.repo != repo:
                continue
            out.append(inst)
            if limit is not None and len(out) >= limit:
                break
    if not out:
        raise ValueError(f"no instances loaded from {path} (repo={repo})")
    return out


def to_task(inst: SWEInstance, tier: str = "B", max_turns: int = 40,
            split: str = "train") -> Task:
    """Map an instance to a Task. `swe` payload travels via Sample.metadata
    to the node2 trajectory dispatcher; gold patch stays behind."""
    return Task(
        task_id=f"swe_{inst.instance_id}",
        line=TaskLine.SWE.value,
        tier=tier,
        prompt=inst.problem_statement,
        reward_spec={"type": "swebench"},
        max_turns=max_turns,
        split=split,
        swe={
            "instance_id": inst.instance_id,
            "repo": inst.repo,
            "base_commit": inst.base_commit,
            "test_patch": inst.test_patch,
            "fail_to_pass": inst.fail_to_pass,
            "pass_to_pass": inst.pass_to_pass,
            "version": inst.version,
        },
    )


def load_tasks(path: str, repo: str | None = None, limit: int | None = None,
               tier: str = "B", max_turns: int = 40,
               split: str = "train") -> list[Task]:
    return [to_task(i, tier=tier, max_turns=max_turns, split=split)
            for i in load_instances(path, repo=repo, limit=limit)]


# ---------------------------------------------------------------------------
# test-node resolution (shared by bake verify on node2 and grader/T28)
# ---------------------------------------------------------------------------

def resolve_test_nodes(swe: dict) -> list[str]:
    """Map FAIL_TO_PASS/PASS_TO_PASS entries to pytest node ids.

    Entries come in three shapes: full node ids (path::name), file paths,
    or bare test-function names (old sympy rows). Bare names are anchored to
    the test file that the official test_patch adds them to; tests not added
    by the patch fall back to the first touched test file.
    """
    patch = swe.get("test_patch", "") or ""
    file_tests: dict[str, set[str]] = {}
    current = None
    for line in patch.splitlines():
        m = re.match(r"^diff --git a/(\S+)", line)
        if m:
            current = m.group(1)
            file_tests.setdefault(current, set())
            continue
        if current and line.startswith("+"):
            t = re.search(r"def\s+(test\w+)", line)
            if t:
                file_tests[current].add(t.group(1))

    files = list(file_tests)
    nodes = []
    for name in swe.get("fail_to_pass", []) + swe.get("pass_to_pass", []):
        if "::" in name or "/" in name:
            nodes.append(name)
            continue
        hit = next((f for f in files if name in file_tests[f]), None)
        nodes.append(f"{hit or (files[0] if files else '')}::{name}"
                     if (hit or files) else name)
    return nodes
