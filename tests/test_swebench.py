"""T23 acceptance: SWE-bench jsonl loader + Task mapping (fixture-based)."""

import json
import os
import tempfile
import unittest

from rl_sim.data_source import RolloutDataSource
from rl_sim.swebench import load_instances, load_tasks, parse_instance
from rl_sim.types import TaskLine


def _row(iid, repo="sympy/sympy", f2p='["test_a"]', p2p='["test_b", "test_c"]'):
    return {
        "instance_id": iid,
        "repo": repo,
        "base_commit": "abc123",
        "problem_statement": f"fix issue {iid}",
        "patch": "diff --git gold...",
        "test_patch": "diff --git test...",
        "version": "1.0",
        "FAIL_TO_PASS": f2p,
        "PASS_TO_PASS": p2p,
    }


class TestSWEBenchLoader(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        rows = [_row("sympy__sympy-1"), _row("sympy__sympy-2"),
                _row("django__django-9", repo="django/django")]
        with open(self.path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def tearDown(self):
        os.unlink(self.path)

    def test_parse_json_string_lists(self):
        inst = parse_instance(_row("x"))
        self.assertEqual(inst.fail_to_pass, ["test_a"])
        self.assertEqual(inst.pass_to_pass, ["test_b", "test_c"])
        self.assertEqual(inst.base_commit, "abc123")

    def test_parse_native_lists(self):
        inst = parse_instance(_row("x", f2p=["t1"], p2p=[]))
        self.assertEqual(inst.fail_to_pass, ["t1"])
        self.assertEqual(inst.pass_to_pass, [])

    def test_load_all_and_filter_repo(self):
        all_inst = load_instances(self.path)
        self.assertEqual(len(all_inst), 3)
        sym = load_instances(self.path, repo="sympy/sympy")
        self.assertEqual(len(sym), 2)
        self.assertTrue(all(i.repo == "sympy/sympy" for i in sym))

    def test_limit(self):
        self.assertEqual(len(load_instances(self.path, limit=1)), 1)

    def test_empty_filter_raises(self):
        with self.assertRaises(ValueError):
            load_instances(self.path, repo="no/such")

    def test_to_task_mapping(self):
        tasks = load_tasks(self.path, repo="sympy/sympy", max_turns=25)
        self.assertEqual(len(tasks), 2)
        t = tasks[0]
        self.assertEqual(t.line, TaskLine.SWE.value)
        self.assertEqual(t.reward_spec["type"], "swebench")
        self.assertEqual(t.tier, "B")
        self.assertEqual(t.max_turns, 25)
        self.assertEqual(t.prompt, f"fix issue {t.swe['instance_id']}")
        self.assertIn("test_patch", t.swe)
        self.assertNotIn("patch", t.swe)  # gold patch must not ship
        self.assertNotIn("gold_patch", t.swe)

    def test_datasource_propagates_swe_payload(self):
        tasks = load_tasks(self.path, repo="sympy/sympy")
        ds = RolloutDataSource(tasks, n_samples_per_prompt=3, seed=0)
        groups = ds.get_groups(1)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 3)
        s = groups[0][0]
        self.assertEqual(s.task_line, TaskLine.SWE.value)
        self.assertIn("swe", s.metadata)
        self.assertEqual(s.metadata["swe"]["repo"], "sympy/sympy")
        self.assertEqual({x.group_id for x in groups[0]}, {0})


_TEST_PATCH = """\
diff --git a/sympy/core/tests/test_arit.py b/sympy/core/tests/test_arit.py
index aaa..bbb 100644
--- a/sympy/core/tests/test_arit.py
+++ b/sympy/core/tests/test_arit.py
@@ -1,2 +1,6 @@
+def test_issue_14941():
+    assert True
+
 def test_old():
     pass
"""


class TestResolveTestNodes(unittest.TestCase):
    def test_bare_names_anchor_to_patch_file(self):
        from rl_sim.swebench import resolve_test_nodes
        swe = {"fail_to_pass": ["test_issue_14941"],
               "pass_to_pass": ["test_old"],
               "test_patch": _TEST_PATCH}
        nodes = resolve_test_nodes(swe)
        self.assertEqual(nodes[0], "sympy/core/tests/test_arit.py::test_issue_14941")
        self.assertEqual(nodes[1], "sympy/core/tests/test_arit.py::test_old")

    def test_full_node_ids_pass_through(self):
        from rl_sim.swebench import resolve_test_nodes
        swe = {"fail_to_pass": ["a/b.py::test_x"], "pass_to_pass": [],
               "test_patch": ""}
        self.assertEqual(resolve_test_nodes(swe), ["a/b.py::test_x"])

    def test_no_patch_falls_back_to_bare(self):
        from rl_sim.swebench import resolve_test_nodes
        swe = {"fail_to_pass": ["test_x"], "pass_to_pass": [], "test_patch": ""}
        self.assertEqual(resolve_test_nodes(swe), ["test_x"])


if __name__ == "__main__":
    unittest.main()
