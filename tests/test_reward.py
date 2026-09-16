"""T03 acceptance: rl_sim/reward.py contracts."""

import unittest

from rl_sim.reward import (
    compute_reward, exact_match, extract_number, normalize_text,
    numeric_match, reward_from_tests,
)
from rl_sim.types import Sample, SampleStatus


def _completed(passed: int, total: int) -> Sample:
    s = Sample(prompt="p", task_id="t")
    s.status = SampleStatus.COMPLETED
    s.tests_passed, s.tests_total = passed, total
    return s


class TestRewardFromTests(unittest.TestCase):
    def test_partial_credit(self):
        self.assertAlmostEqual(reward_from_tests(_completed(3, 4)), 0.75)

    def test_full_and_zero(self):
        self.assertEqual(reward_from_tests(_completed(4, 4)), 1.0)
        self.assertEqual(reward_from_tests(_completed(0, 4)), 0.0)

    def test_failed_sandbox_scores_zero(self):
        for st in (SampleStatus.SANDBOX_TIMEOUT, SampleStatus.SANDBOX_OOM,
                   SampleStatus.SANDBOX_ERROR, SampleStatus.PENDING):
            s = Sample(prompt="p", task_id="t", status=st)
            s.tests_total = 4
            self.assertEqual(reward_from_tests(s), 0.0, st)


class TestExtractNumber(unittest.TestCase):
    def test_boxed_preferred(self):
        self.assertEqual(extract_number(r"so 99 is wrong, thus $\boxed{42}$"), 42.0)

    def test_last_number_fallback(self):
        self.assertEqual(extract_number("1 + 2 = 3, answer is 4"), 4.0)

    def test_commas_and_none(self):
        self.assertEqual(extract_number("about 1,024 tokens"), 1024.0)
        self.assertIsNone(extract_number("no numbers here"))


class TestNumericMatch(unittest.TestCase):
    def test_match_and_tolerance(self):
        self.assertTrue(numeric_match("answer: 42", 42))
        self.assertTrue(numeric_match("41.999", 42, rel_tol=1e-3))
        self.assertFalse(numeric_match("43", 42))

    def test_zero_answer(self):
        self.assertTrue(numeric_match("0", 0))
        self.assertFalse(numeric_match("0.5", 0))


class TestExactMatch(unittest.TestCase):
    def test_normalized(self):
        self.assertEqual(normalize_text("  Hello\n World "), "hello world")
        self.assertTrue(exact_match("The answer is Paris.", "paris"))
        self.assertFalse(exact_match("I think London", "paris"))


class TestDispatch(unittest.TestCase):
    def test_tests_type(self):
        s = _completed(2, 4)
        self.assertEqual(compute_reward(s, {"type": "tests"}), 0.5)

    def test_numeric_and_exact(self):
        s = _completed(0, 0)
        s.response = "the total is 1,000"
        self.assertEqual(compute_reward(s, {"type": "numeric", "answer": 1000}), 1.0)
        s.response = "capital: Oslo"
        self.assertEqual(compute_reward(s, {"type": "exact", "answer": "oslo"}), 1.0)

    def test_gen_error_zero(self):
        s = Sample(prompt="p", task_id="t", status=SampleStatus.GEN_ERROR)
        s.response = "42"
        self.assertEqual(compute_reward(s, {"type": "numeric", "answer": 42}), 0.0)

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            compute_reward(Sample(prompt="p", task_id="t"), {"type": "nope"})


if __name__ == "__main__":
    unittest.main()
