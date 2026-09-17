"""T07 acceptance: data_source task schema + group sampling."""

import unittest

from rl_sim.data_source import (
    ALL_TASKS, A_TASKS, L1_TASKS, RolloutDataSource, Task, task_by_id,
)
from rl_sim.types import TaskLine


class TestTaskSet(unittest.TestCase):
    def test_lines_covered(self):
        lines = {t.line for t in ALL_TASKS}
        for line in (TaskLine.A_CODE.value, TaskLine.L1_SWE.value,
                     TaskLine.L2_TERM.value, TaskLine.L3_SEARCH.value):
            self.assertIn(line, lines)

    def test_unique_ids(self):
        ids = [t.task_id for t in ALL_TASKS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_a_tasks_wellformed(self):
        for t in A_TASKS:
            self.assertEqual(t.reward_spec["type"], "tests")
            self.assertTrue(t.tests, t.task_id)
            self.assertTrue(t.solution, t.task_id)  # MockEngine needs it
            self.assertTrue(t.entry, t.task_id)

    def test_l1_repo_has_test_cmd(self):
        for t in L1_TASKS:
            self.assertIn("test_calc.py", t.repo)
            self.assertEqual(t.test_cmd, "python test_calc.py")
            self.assertGreater(t.max_turns, 1)

    def test_l3_corpus_attached(self):
        t = task_by_id("l3_eiffel_year")
        self.assertTrue(t.corpus)
        self.assertEqual(t.reward_spec["type"], "numeric")

    def test_eval_split_disjoint(self):
        train = {t.task_id for t in ALL_TASKS if t.split == "train"}
        ev = {t.task_id for t in ALL_TASKS if t.split == "eval"}
        self.assertTrue(train.isdisjoint(ev))
        self.assertTrue(ev)


class TestRolloutDataSource(unittest.TestCase):
    def test_group_expansion(self):
        ds = RolloutDataSource(ALL_TASKS, n_samples_per_prompt=8)
        groups = ds.get_groups(4)
        self.assertEqual(len(groups), 4)
        self.assertTrue(all(len(g) == 8 for g in groups))
        for g in groups:
            gids = {s.group_id for s in g}
            tids = {s.task_id for s in g}
            self.assertEqual(len(gids), 1)
            self.assertEqual(len(tids), 1)

    def test_no_intra_epoch_repeat(self):
        ds = RolloutDataSource(ALL_TASKS, n_samples_per_prompt=1)
        n_train = len([t for t in ALL_TASKS if t.split == "train"])
        seen = [g[0].task_id for g in ds.get_groups(n_train)]
        self.assertEqual(len(seen), len(set(seen)))  # one epoch, no repeat

    def test_eval_tasks(self):
        ds = RolloutDataSource(ALL_TASKS)
        ev = ds.get_eval_tasks()
        self.assertTrue(all(t.split == "eval" for t in ev))

    def test_empty_split_raises(self):
        with self.assertRaises(ValueError):
            RolloutDataSource([t for t in ALL_TASKS if t.split == "train"], split="nope")


if __name__ == "__main__":
    unittest.main()
