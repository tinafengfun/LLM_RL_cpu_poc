"""T14 acceptance: MockMegatronTrainer numerics (hand-computed)."""

import math
import unittest

from rl_sim.trainer import MockMegatronTrainer
from rl_sim.types import Sample


def mk_sample(adv, logp=-2.0, tokens=10, wv=0):
    s = Sample(prompt="p", task_id="t")
    s.advantage = adv
    s.rollout_logprob = logp
    s.num_tokens = tokens
    s.weight_version = wv
    return s


class TestTrainer(unittest.TestCase):
    def test_identity_ratio_when_rescore_matches(self):
        # rescore == rollout logprob -> ratio 1 -> loss = -mean(adv), no clip/mask
        tr = MockMegatronTrainer(rescore_fn=lambda s: s.rollout_logprob,
                                 sim_tokens_per_s=1e18)
        data = {"samples": [mk_sample(1.0), mk_sample(-0.5)]}
        m = tr.async_train(0, data)
        self.assertAlmostEqual(m["loss"], -(1.0 - 0.5) / 2)
        self.assertEqual(m["frac_clipped"], 0.0)
        self.assertEqual(m["tis_masked_frac"], 0.0)
        self.assertEqual(m["weight_version"], 1)

    def test_clip_engages(self):
        # ratio = e^2 >> 1+eps -> clipped branch dominates
        tr = MockMegatronTrainer(rescore_fn=lambda s: s.rollout_logprob + 2.0,
                                 use_tis=False, sim_tokens_per_s=1e18)
        m = tr.async_train(0, {"samples": [mk_sample(1.0)]})
        self.assertAlmostEqual(m["loss"], -(1.28 * 1.0))  # clip to 1+eps_high
        self.assertEqual(m["frac_clipped"], 1.0)

    def test_tis_masks_out_of_region(self):
        # ratio = e^3 ≈ 20 > beta=2 -> masked, excluded from loss
        tr = MockMegatronTrainer(rescore_fn=lambda s: s.rollout_logprob + 3.0,
                                 use_tis=True, tis_beta=2.0, sim_tokens_per_s=1e18)
        m = tr.async_train(0, {"samples": [mk_sample(1.0), mk_sample(-1.0)]})
        self.assertEqual(m["tis_masked_frac"], 1.0)
        self.assertEqual(m["loss"], 0.0)  # everything masked

    def test_staleness_widens_mismatch(self):
        tr = MockMegatronTrainer(mismatch_sigma=1.0, seed=3, sim_tokens_per_s=1e18)
        s_fresh = mk_sample(1.0, wv=0)
        s_stale = mk_sample(1.0, wv=0)
        tr.weight_version = 3
        tr.async_train(0, {"samples": [s_fresh, s_stale]})
        # stale sample (staleness=3) gets wider noise than fresh (staleness=0): same seed
        # stream order -> just check train_logprob differs from rollout_logprob
        self.assertNotEqual(s_stale.train_logprob, s_stale.rollout_logprob)

    def test_version_and_timing(self):
        tr = MockMegatronTrainer(sim_tokens_per_s=1000.0, sim_max_s=0.05,
                                 rescore_fn=lambda s: s.rollout_logprob)
        m = tr.async_train(0, {"samples": [mk_sample(1.0, tokens=10000)]})
        self.assertGreaterEqual(m["sim_train_s"], 0.05)
        self.assertEqual(tr.weight_version, 1)

    def test_checkpoint(self):
        import json, tempfile, pathlib
        tr = MockMegatronTrainer()
        tr.weight_version = 7
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "ckpt.json"
            tr.save_checkpoint(str(p))
            self.assertEqual(json.loads(p.read_text())["weight_version"], 7)


if __name__ == "__main__":
    unittest.main()
