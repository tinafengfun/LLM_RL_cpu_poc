"""T24 acceptance: pure helpers of tools/bake_swe_env.py (no network)."""

import importlib.util
import json
import os
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "bake_swe_env",
    os.path.join(os.path.dirname(__file__), "..", "tools", "bake_swe_env.py"))
bake = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bake)


def _hf_payload(rows):
    return {"num_rows_total": len(rows),
            "rows": [{"row": r} for r in rows]}


class TestBakeHelpers(unittest.TestCase):
    def test_page_url(self):
        u = bake.page_url(100, 50)
        self.assertIn("offset=100", u)
        self.assertIn("length=50", u)
        self.assertIn("princeton-nlp%2FSWE-bench_Verified", u)

    def test_rows_to_instances(self):
        rows = [{"instance_id": "a"}, {"instance_id": "b"}]
        got = bake.rows_to_instances(_hf_payload(rows))
        self.assertEqual([r["instance_id"] for r in got], ["a", "b"])
        self.assertEqual(bake.rows_to_instances({"rows": []}), [])

    def test_select_subset_filters_and_sorts(self):
        rows = [
            {"repo": "sympy/sympy", "instance_id": "sympy__sympy-3"},
            {"repo": "django/django", "instance_id": "django__django-1"},
            {"repo": "sympy/sympy", "instance_id": "sympy__sympy-1"},
        ]
        got = bake.select_subset(rows, "sympy/sympy")
        self.assertEqual([r["instance_id"] for r in got],
                         ["sympy__sympy-1", "sympy__sympy-3"])
        one = bake.select_subset(rows, "sympy/sympy", limit=1)
        self.assertEqual(len(one), 1)

    def test_select_subset_empty_raises(self):
        with self.assertRaises(ValueError):
            bake.select_subset([{"repo": "x/y"}], "sympy/sympy")

    def test_build_manifest(self):
        m = bake.build_manifest("/workspace/swebench", "sympy/sympy",
                                "/workspace/swebench/repo/sympy.git",
                                "/workspace/swebench/envs/py311.tar.gz",
                                "py311", "/workspace/swebench/instances.jsonl", 50)
        self.assertEqual(m["num_instances"], 50)
        self.assertEqual(m["repo"], "sympy/sympy")
        json.dumps(m)  # must be serializable


if __name__ == "__main__":
    unittest.main()
