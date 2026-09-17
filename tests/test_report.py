"""T18 acceptance: full-format rollout report fields and self-consistency."""

import unittest

from rl_sim.monitor import rollout_report


class TestRolloutReport(unittest.TestCase):
    def test_all_sections_present(self):
        out = rollout_report(
            3,
            {"generate": (12.3, 1.1), "train_mock_gpu": (2.0, 2.0), "weight_sync": (0.1, 0.1)},
            {"peak_busy": 8, "peak_queue": 14,
             "exec_walls": [0.1, 0.2, 0.3], "queue_waits": [0.01, 0.02]},
            8,
            {"weight_version": 4, "num_samples": 128, "loss": -0.12,
             "tis_masked_frac": 0.032, "staleness_max": 1},
            {"requests": 128, "tokens": 4096, "lat": [0.5, 0.7], "backends": 2},
            {"eval_pass_rate": 0.5, "eval_n": 4},
        )
        for frag in ("[rollout 3]", "wall:", "cpu :", "[engine]", "[sandbox]",
                     "[train]", "[eval]", "tis_masked", "q_peak 14", "wv 4"):
            self.assertIn(frag, out)

    def test_optional_sections_omitted(self):
        out = rollout_report(0, {"a": (1.0, 0.5)},
                             {"peak_busy": 1, "peak_queue": 0,
                              "exec_walls": [], "queue_waits": []},
                             1, {"weight_version": 1})
        self.assertNotIn("[engine]", out)
        self.assertNotIn("[eval]", out)
        self.assertIn("[sandbox]", out)


if __name__ == "__main__":
    unittest.main()
