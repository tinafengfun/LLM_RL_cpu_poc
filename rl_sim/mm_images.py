"""Stdlib-only image generation for V1/V2 multimodal tasks (design v3.1 section 2.2).

No matplotlib on the node (stdlib-only rule) -> tiny PNG writer + procedural
chart/shape renderers. Ground-truth answers are known by construction.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (70, 110, 200)
RED = (200, 70, 70)
GREEN = (70, 160, 90)
GRAY = (200, 200, 200)


def write_png(path: str | Path, pixels: list[list[tuple[int, int, int]]]) -> None:
    h = len(pixels)
    w = len(pixels[0])
    raw = b"".join(b"\x00" + bytes(c for px in row for c in px) for row in pixels)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def _canvas(w: int, h: int, color=WHITE) -> list[list[tuple[int, int, int]]]:
    return [[color] * w for _ in range(h)]


def _fill_rect(px, x0, y0, x1, y1, color):
    for y in range(max(0, y0), min(len(px), y1)):
        row = px[y]
        for x in range(max(0, x0), min(len(row), x1)):
            row[x] = color


def _fill_circle(px, cx, cy, r, color):
    for y in range(max(0, cy - r), min(len(px), cy + r + 1)):
        for x in range(max(0, cx - r), min(len(px[0]), cx + r + 1)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                px[y][x] = color


def render_bar_chart(values: list[int], path: str | Path, w: int = 400, h: int = 300) -> None:
    """V1: vertical bar chart; tallest bar is identifiable by construction."""
    px = _canvas(w, h)
    _fill_rect(px, 39, 10, 41, h - 30, BLACK)  # y axis
    _fill_rect(px, 40, h - 31, w - 10, h - 29, BLACK)  # x axis
    n = len(values)
    slot = (w - 60) // n
    vmax = max(values)
    for i, v in enumerate(values):
        bh = int((h - 60) * v / vmax)
        x0 = 45 + i * slot
        _fill_rect(px, x0 + 3, h - 31 - bh, x0 + slot - 3, h - 31, BLUE)
    write_png(path, px)


def render_shapes(n_circles: int, n_rects: int, path: str | Path,
                  w: int = 320, h: int = 240, seed: int = 0) -> None:
    """V2: circles (RED) + rectangles (GREEN) on white canvas; count circles."""
    import random
    rng = random.Random(seed)
    px = _canvas(w, h)
    for _ in range(n_rects):
        rw, rh = rng.randint(20, 50), rng.randint(20, 50)
        x, y = rng.randint(5, w - rw - 5), rng.randint(5, h - rh - 5)
        _fill_rect(px, x, y, x + rw, y + rh, GREEN)
    for _ in range(n_circles):
        r = rng.randint(10, 25)
        _fill_circle(px, rng.randint(r + 5, w - r - 5), rng.randint(r + 5, h - r - 5), r, RED)
    write_png(path, px)
