"""Task schema + RolloutDataSource ← slime/rollout/data_source.py.

Task lines (design v3.1 section 2): A code exec, L1 SWE-mini, L2 terminal,
L3 multi-hop QA, V1/V2 multimodal (T08), MOCK fallback.

Mirrors slime's epoch-based group sampling: each prompt is expanded into
`n_samples_per_prompt` Samples sharing a group_id; epochs cycle without
intra-epoch repetition; a held-out split feeds eval.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from rl_sim.types import Sample, TaskLine


@dataclass
class Task:
    task_id: str
    line: str  # TaskLine value
    prompt: str
    reward_spec: dict  # {"type": "tests"|"numeric"|"exact", ...}
    tier: str = "A"  # sandbox resource profile
    tests: list[str] = field(default_factory=list)  # A/L1: assert statements
    answer: str | float | None = None  # L3/V*: expected answer
    entry: str = ""  # A: entry function name
    repo: dict[str, str] = field(default_factory=dict)  # L1: path -> content
    test_cmd: str = ""  # L1/L2: command to run inside sandbox
    corpus: dict[str, str] = field(default_factory=dict)  # L3: doc_id -> text
    max_turns: int = 1  # agentic lines allow multi-turn
    solution: str = ""  # MockEngine fallback reference (NOT shown to model)
    image_path: str | None = None  # V1/V2
    split: str = "train"  # train | eval


# ---------------------------------------------------------------------------
# Built-in task set. Kept small and self-contained; content is illustrative,
# load realism comes from execution, not from dataset size.
# ---------------------------------------------------------------------------

def _a(task_id, entry, doc, tests, solution, split="train") -> Task:
    prompt = (
        f"Implement the following Python function. Return ONLY a python code block.\n\n"
        f"```python\ndef {entry}{doc}\n```"
    )
    return Task(task_id=task_id, line=TaskLine.A_CODE.value, prompt=prompt,
                reward_spec={"type": "tests"}, tier="A", tests=tests,
                entry=entry, solution=solution, split=split)


A_TASKS = [
    _a("a_reverse_string", "reverse_string", "(s):\n    \"\"\"Return the reversed string.\"\"\"",
       ["assert reverse_string('abc') == 'cba'", "assert reverse_string('') == ''",
        "assert reverse_string('a') == 'a'", "assert reverse_string('ab cd') == 'dc ba'"],
       "def reverse_string(s):\n    return s[::-1]\n"),
    _a("a_is_palindrome", "is_palindrome", "(s):\n    \"\"\"True if s reads the same backwards.\"\"\"",
       ["assert is_palindrome('aba') is True", "assert is_palindrome('abc') is False",
        "assert is_palindrome('') is True", "assert is_palindrome('aa') is True"],
       "def is_palindrome(s):\n    return s == s[::-1]\n"),
    _a("a_max_subarray", "max_subarray", "(nums):\n    \"\"\"Max sum of any contiguous subarray (Kadane).\"\"\"",
       ["assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6", "assert max_subarray([1]) == 1",
        "assert max_subarray([-1,-2]) == -1", "assert max_subarray([5,4,-1,7,8]) == 23"],
       "def max_subarray(nums):\n    best = cur = nums[0]\n    for x in nums[1:]:\n        cur = max(x, cur + x)\n        best = max(best, cur)\n    return best\n"),
    _a("a_count_vowels", "count_vowels", "(s):\n    \"\"\"Count aeiou (case-insensitive).\"\"\"",
       ["assert count_vowels('hello') == 2", "assert count_vowels('AEIOU') == 5",
        "assert count_vowels('xyz') == 0", "assert count_vowels('') == 0"],
       "def count_vowels(s):\n    return sum(c in 'aeiou' for c in s.lower())\n"),
    _a("a_flatten", "flatten", "(xs):\n    \"\"\"Flatten one level of nesting: [[1,2],[3]] -> [1,2,3].\"\"\"",
       ["assert flatten([[1,2],[3]]) == [1,2,3]", "assert flatten([]) == []",
        "assert flatten([[],[]]) == []", "assert flatten([[1],[2],[3,4]]) == [1,2,3,4]"],
       "def flatten(xs):\n    return [x for sub in xs for x in sub]\n"),
    _a("a_fizzbuzz", "fizzbuzz", "(n):\n    \"\"\"List 1..n with Fizz/Buzz/FizzBuzz substitutions.\"\"\"",
       ["assert fizzbuzz(1) == ['1']", "assert fizzbuzz(3) == ['1','2','Fizz']",
        "assert fizzbuzz(5)[4] == 'Buzz'", "assert fizzbuzz(15)[14] == 'FizzBuzz'"],
       "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        s = ('Fizz' if i % 3 == 0 else '') + ('Buzz' if i % 5 == 0 else '')\n        out.append(s or str(i))\n    return out\n"),
    # held-out eval
    _a("a_second_max", "second_max", "(nums):\n    \"\"\"Second largest distinct value; None if <2 distinct.\"\"\"",
       ["assert second_max([1,2,3]) == 2", "assert second_max([5,5]) is None",
        "assert second_max([3,3,2,1]) == 2", "assert second_max([1]) is None"],
       "def second_max(nums):\n    u = sorted(set(nums), reverse=True)\n    return u[1] if len(u) > 1 else None\n",
       split="eval"),
    _a("a_word_freq", "top_word", "(text):\n    \"\"\"Most frequent word (lowercased); ties -> first seen.\"\"\"",
       ["assert top_word('a b a') == 'a'", "assert top_word('x Y y') == 'x'",
        "assert top_word('one') == 'one'"],
       "def top_word(text):\n    from collections import Counter\n    ws = text.lower().split()\n    return max(Counter(ws), key=lambda w: (Counter(ws)[w], -ws.index(w))) if ws else None\n",
       split="eval"),
]


def _l1_calc_repo() -> Task:
    repo = {
        "calc/__init__.py": "from .core import Calculator\n",
        "calc/core.py": (
            "class Calculator:\n"
            "    def add(self, a, b): return a + b\n"
            "    def div(self, a, b): return a / b  # BUG: no zero guard\n"
            "    def avg(self, xs): return sum(xs) / len(xs)  # BUG: empty list crash\n"
        ),
        "test_calc.py": (
            "from calc import Calculator\n"
            "c = Calculator()\n"
            "assert c.add(1, 2) == 3\n"
            "assert c.div(4, 2) == 2\n"
            "assert c.div(1, 0) is None, 'div by zero should return None'\n"
            "assert c.avg([2, 4]) == 3\n"
            "assert c.avg([]) is None, 'avg of empty should return None'\n"
            "print('ALL PASS')\n"
        ),
    }
    return Task(
        task_id="l1_calc_zero", line=TaskLine.L1_SWE.value, tier="B", max_turns=8,
        prompt=("Repo has a bug: Calculator.div crashes on zero and avg crashes on empty. "
                "Fix calc/core.py so div(...,0) and avg([]) return None. "
                "Use tools to read files, edit, and run `python test_calc.py` until ALL PASS."),
        reward_spec={"type": "tests"}, repo=repo, test_cmd="python test_calc.py",
        solution="return None guards", split="train",
    )


L1_TASKS = [_l1_calc_repo()]

L2_TASKS = [
    Task(
        task_id="l2_log_errors", line=TaskLine.L2_TERM.value, tier="C", max_turns=8,
        prompt=("access.log has lines 'HH:MM LEVEL msg'. Count ERROR lines per hour and "
                "write 'HH:count' lines sorted by hour to answer.txt."),
        reward_spec={"type": "exact"},
        answer="09:2\n10:1\n11:3",  # matches corpus generated in T07 fixtures
        solution="grep ERROR | cut -d: -f1 | sort | uniq -c",
        split="train",
    ),
]

_L3_CORPUS = {
    "doc_paris": "Paris is the capital of France. The Eiffel Tower is located in Paris.",
    "doc_eiffel": "The Eiffel Tower was completed in 1889. It was designed by Gustave Eiffel.",
    "doc_france": "France is a country in Europe. Its population is about 68 million.",
}

L3_TASKS = [
    Task(
        task_id="l3_eiffel_year", line=TaskLine.L3_SEARCH.value, tier="C", max_turns=8,
        prompt=("Answer using the search tool (2 hops needed): In what year was the tower "
                "located in the capital of France completed?"),
        reward_spec={"type": "numeric"}, answer=1889, corpus=_L3_CORPUS, split="train",
    ),
]

V_TASKS = [
    Task(task_id="v1_bars_tallest_1", line=TaskLine.V1_CHART.value, tier="A",
         prompt=("Look at the bar chart image. How many bars are there? "
                 "Answer with a single number."),
         reward_spec={"type": "numeric"}, answer=6,
         image_path="vendor/images/v1_bars_tallest_1.png", split="train"),
    Task(task_id="v1_bars_tallest_2", line=TaskLine.V1_CHART.value, tier="A",
         prompt=("Look at the bar chart image. How many bars are there? "
                 "Answer with a single number."),
         reward_spec={"type": "numeric"}, answer=6,
         image_path="vendor/images/v1_bars_tallest_2.png", split="train"),
    Task(task_id="v2_shapes_count_1", line=TaskLine.V2_GEOM.value, tier="A",
         prompt="Look at the image. How many RED CIRCLES are there? Answer with a single number.",
         reward_spec={"type": "numeric"}, answer=3,
         image_path="vendor/images/v2_shapes_1.png", split="train"),
    Task(task_id="v2_shapes_count_2", line=TaskLine.V2_GEOM.value, tier="A",
         prompt="Look at the image. How many RED CIRCLES are there? Answer with a single number.",
         reward_spec={"type": "numeric"}, answer=5,
         image_path="vendor/images/v2_shapes_2.png", split="train"),
    Task(task_id="v1_bars_tallest_eval", line=TaskLine.V1_CHART.value, tier="A",
         prompt=("Look at the bar chart image. How many bars are there? "
                 "Answer with a single number."),
         reward_spec={"type": "numeric"}, answer=6,
         image_path="vendor/images/v1_bars_tallest_eval.png", split="eval"),
]

MOCK_TASKS = A_TASKS[:3]  # reuse; MockEngine mutates their solutions

ALL_TASKS = A_TASKS + L1_TASKS + L2_TASKS + L3_TASKS + V_TASKS


class RolloutDataSource:
    """Epoch-based group sampler ← slime RolloutDataSource.

    get_groups(batch_size) -> list of groups; each group is
    n_samples_per_prompt Samples sharing one group_id.
    """

    def __init__(self, tasks: list[Task], n_samples_per_prompt: int = 8,
                 seed: int = 0, split: str = "train") -> None:
        self.tasks = [t for t in tasks if t.split == split]
        if not self.tasks:
            raise ValueError(f"no tasks in split={split}")
        if n_samples_per_prompt < 1:
            raise ValueError("n_samples_per_prompt must be >= 1")
        self.n = n_samples_per_prompt
        self._rng = random.Random(seed)
        self._epoch: list[Task] = []
        self._group_counter = 0

    def _next_task(self) -> Task:
        if not self._epoch:
            self._epoch = list(self.tasks)
            self._rng.shuffle(self._epoch)
        return self._epoch.pop()

    def get_groups(self, batch_size: int) -> list[list[Sample]]:
        groups = []
        for _ in range(batch_size):
            task = self._next_task()
            gid = self._group_counter
            self._group_counter += 1
            groups.append([
                Sample(prompt=task.prompt, task_id=task.task_id, task_line=task.line,
                       group_id=gid, image_path=task.image_path,
                       metadata={"tier": task.tier, "max_turns": task.max_turns})
                for _ in range(self.n)
            ])
        return groups

    def get_eval_tasks(self) -> list[Task]:
        return [t for t in ALL_TASKS if t.split == "eval"]


def task_by_id(task_id: str) -> Task:
    for t in ALL_TASKS:
        if t.task_id == task_id:
            return t
    raise KeyError(task_id)
