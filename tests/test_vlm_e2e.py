"""T20 acceptance [node]: VLM end-to-end (Qwen2.5-VL + mmproj, image input, real logprobs)."""

import os
import unittest

from rl_sim.data_source import task_by_id
from rl_sim.reward import compute_reward
from rl_sim.types import Sample


@unittest.skipUnless(os.environ.get("RL_SIM_NODE") == "1", "node-only integration")
class TestVlmE2E(unittest.TestCase):
    def test_vlm_chart_qa(self):
        from rl_sim.engine_local import InstanceSpec, LlamaServerManager, LocalEngine
        model = os.environ.get("RL_SIM_VLM", "/mnt/nvme0/models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf")
        mmproj = os.environ.get("RL_SIM_MMPROJ", "/mnt/nvme0/models/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf")
        mgr = LlamaServerManager(
            "/mnt/nvme0/models/llama.cpp/build/bin/llama-server",
            [InstanceSpec(name="vlm0", port=19320, cores="64-79", slots=2,
                          model=model, mmproj=mmproj)])
        mgr.start("vlm0")
        try:
            self.assertTrue(mgr.wait_healthy(600))
            eng = LocalEngine(mgr.urls(), max_tokens=128)
            task = task_by_id("v1_bars_tallest_1")
            s = Sample(prompt=task.prompt, task_id=task.task_id,
                       task_line=task.line, image_path=task.image_path)
            s = eng.generate(s)
            # contract: real content + real logprobs + numeric answer parseable
            self.assertEqual(s.status.value, "pending")
            self.assertTrue(s.response.strip())
            self.assertLess(s.rollout_logprob, 0)
            self.assertGreater(s.num_tokens, 0)
            r = compute_reward(s, task.reward_spec)
            self.assertIn(r, (0.0, 1.0))
            print(f"\nVLM response: {s.response[:200]!r} reward={r} logp={s.rollout_logprob:.2f}")
        finally:
            mgr.stop_all()


if __name__ == "__main__":
    unittest.main()
