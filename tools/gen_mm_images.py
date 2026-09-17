#!/usr/bin/env python3
"""Generate vendored V1/V2 task images into vendor/images/ (node-offline friendly)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_sim.mm_images import render_bar_chart, render_shapes

OUT = Path(__file__).resolve().parent.parent / "vendor" / "images"

SPECS = [
    ("v1_bars_tallest_1.png", "bar", [3, 7, 5, 9, 2, 6]),
    ("v1_bars_tallest_2.png", "bar", [8, 4, 6, 5, 7, 3]),
    ("v1_bars_tallest_eval.png", "bar", [2, 9, 4, 6, 5, 7]),
    ("v2_shapes_1.png", "shapes", (3, 2, 11)),
    ("v2_shapes_2.png", "shapes", (5, 1, 12)),
    ("v2_shapes_eval.png", "shapes", (2, 4, 13)),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, kind, spec in SPECS:
        path = OUT / name
        if kind == "bar":
            render_bar_chart(spec, path)
        else:
            n_c, n_r, seed = spec
            render_shapes(n_c, n_r, path, seed=seed)
        print("wrote", path)


if __name__ == "__main__":
    main()
