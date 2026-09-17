"""T17 acceptance: train_async.py lookahead + weight sync interval + staleness."""

import json
import tempfile
import unittest
from pathlib import Path

import train_async


class TestTrainAsync(unittest.TestCase):
    def test_lookahead_loop(self):
        with tempfile.TemporaryDirectory() as d:
            summary = train_async.main([
                "--engine", "mock", "--num-rollout", "4",
                "--rollout-batch-size", "2", "--n-samples-per-prompt", "2",
                "--sandbox-workers", "2", "--update-weights-interval", "2",
                "--sim-tokens-per-s", "1e18", "--out", d,
            ])
            # 4 rollouts, sync every 2 -> engine sees version 2 after rid 1, 4 after rid 3
            self.assertEqual(len(summary["history"]), 4)
            self.assertEqual(summary["final_weight_version"], 4)
            self.assertTrue((Path(d) / "summary.json").exists())
            saved = json.loads((Path(d) / "summary.json").read_text())
            self.assertEqual(saved["mode"], "async_lookahead")

    def test_staleness_drop(self):
        with tempfile.TemporaryDirectory() as d:
            summary = train_async.main([
                "--engine", "mock", "--num-rollout", "4",
                "--rollout-batch-size", "2", "--n-samples-per-prompt", "2",
                "--sandbox-workers", "2", "--update-weights-interval", "1",
                "--max-staleness", "0",
                "--sim-tokens-per-s", "1e18", "--out", d,
            ])
            # with sync every step and max_staleness=0, lookahead samples (generated
            # before the last sync) may be dropped
            self.assertIn("dropped_stale_total", summary)
            self.assertEqual(summary["final_weight_version"], 4)


if __name__ == "__main__":
    unittest.main()
