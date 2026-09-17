"""T16 acceptance: train.py sync closed loop smoke (mock engine)."""

import json
import unittest
from pathlib import Path
import tempfile

import train


class TestTrainSmoke(unittest.TestCase):
    def test_sync_loop_mock_engine(self):
        with tempfile.TemporaryDirectory() as d:
            summary = train.main([
                "--engine", "mock",
                "--num-rollout", "3",
                "--rollout-batch-size", "2",
                "--n-samples-per-prompt", "4",
                "--sandbox-workers", "2",
                "--save-interval", "2",
                "--eval-interval", "3",
                "--sim-tokens-per-s", "1e18",
                "--out", d,
            ])
            self.assertEqual(summary["final_weight_version"], 3)
            self.assertEqual(len(summary["history"]), 3)
            self.assertTrue((Path(d) / "summary.json").exists())
            ckpts = list(Path(d).glob("ckpt_v*.json"))
            self.assertTrue(ckpts)  # checkpoint written
            last = json.loads(sorted(ckpts)[-1].read_text())
            self.assertEqual(last["weight_version"], 3)
            # eval ran at final rollout
            self.assertIn("eval_pass_rate", summary["history"][-1])

    def test_abort_path_runs(self):
        with tempfile.TemporaryDirectory() as d:
            summary = train.main([
                "--engine", "mock", "--num-rollout", "2",
                "--rollout-batch-size", "3", "--n-samples-per-prompt", "2",
                "--sandbox-workers", "2", "--abort-after-groups", "2",
                "--sim-tokens-per-s", "1e18", "--out", d,
            ])
            self.assertEqual(summary["final_weight_version"], 2)


if __name__ == "__main__":
    unittest.main()
