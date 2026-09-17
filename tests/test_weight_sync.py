"""T15 acceptance: WeightUpdater sequencing (pause -> transfer -> version -> resume)."""

import unittest

from rl_sim.engine import Engine
from rl_sim.weight_sync import WeightUpdater


class RecordingEngine(Engine):
    def __init__(self):
        super().__init__()
        self.events = []

    def pause_generation(self):
        self.events.append("pause")
        super().pause_generation()

    def update_weights(self, version):
        self.events.append(("update", version))
        super().update_weights(version)

    def continue_generation(self):
        self.events.append("continue")
        super().continue_generation()


class TestWeightUpdater(unittest.TestCase):
    def test_sequence_and_version(self):
        eng = RecordingEngine()
        up = WeightUpdater(eng, sim_param_gb=4.0, sim_bandwidth_gbps=50.0,
                           real_sleep_cap_s=0.01)
        rec = up.update_weights(3)
        self.assertEqual(eng.events, ["pause", ("update", 3), "continue"])
        self.assertEqual(eng.weight_version, 3)
        self.assertEqual(rec["version"], 3)
        self.assertAlmostEqual(rec["sim_transfer_s"], 0.08)
        self.assertGreaterEqual(rec["pause_window_s"], 0.01)

    def test_reload_hook_measured(self):
        eng = RecordingEngine()
        up = WeightUpdater(eng, real_sleep_cap_s=0.0,
                           reload_hook=lambda: None)
        rec = up.update_weights(1)
        self.assertIn("reload_s", rec)
        self.assertGreaterEqual(rec["reload_s"], 0.0)

    def test_history_accumulates(self):
        eng = RecordingEngine()
        up = WeightUpdater(eng, real_sleep_cap_s=0.0)
        up.update_weights(1)
        up.update_weights(2)
        self.assertEqual([r["version"] for r in up.history], [1, 2])


if __name__ == "__main__":
    unittest.main()
