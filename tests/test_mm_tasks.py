"""T08 acceptance: multimodal image generation + V1/V2 task registration."""

import tempfile
import unittest
from pathlib import Path

from rl_sim.data_source import ALL_TASKS
from rl_sim.mm_images import render_bar_chart, render_shapes, write_png
from rl_sim.reward import numeric_match


class TestPngWriter(unittest.TestCase):
    def test_valid_png_signature_and_dims(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.png"
            write_png(p, [[(255, 0, 0)] * 10 for _ in range(7)])
            data = p.read_bytes()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            self.assertIn(b"IHDR", data)
            self.assertIn(b"IEND", data)


class TestRenderers(unittest.TestCase):
    def test_bar_chart_pixels(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.png"
            render_bar_chart([3, 7, 5, 9, 2, 6], p)
            self.assertGreater(p.stat().st_size, 100)

    def test_shapes_pixels(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.png"
            render_shapes(3, 2, p, seed=1)
            self.assertGreater(p.stat().st_size, 100)


class TestVTasks(unittest.TestCase):
    def test_v_tasks_registered(self):
        v = [t for t in ALL_TASKS if t.line in ("V1", "V2")]
        self.assertGreaterEqual(len(v), 4)
        for t in v:
            self.assertTrue(t.image_path and t.image_path.endswith(".png"), t.task_id)
            self.assertEqual(t.reward_spec["type"], "numeric")
            self.assertIsNotNone(t.answer)

    def test_ground_truth_answers_consistent(self):
        by_id = {t.task_id: t for t in ALL_TASKS}
        # bar values [3,7,5,9,2,6] -> 6 bars (question asks count, answerable from image)
        self.assertEqual(by_id["v1_bars_tallest_1"].answer, 6)
        # shapes spec: 3 circles
        self.assertEqual(by_id["v2_shapes_count_1"].answer, 3)

    def test_answer_verifiable(self):
        self.assertTrue(numeric_match("the tallest bar is 9", 9))
        self.assertFalse(numeric_match("the tallest bar is 7", 9))


if __name__ == "__main__":
    unittest.main()
