"""Step 1: find text lines and group them into speech-bubble regions.

Uses EasyOCR's CRAFT text detector. Nearby line boxes are merged into one region,
so a bubble with three vertical columns becomes one TextRegion.
"""
import threading

import numpy as np

from .. import config
from .models import Box, TextRegion, union

_readers: dict[str, object] = {}
_lock = threading.Lock()

# EasyOCR language codes for each source language
EASYOCR_LANGS = {"ja": ["ja", "en"], "zh": ["ch_sim", "en"], "ko": ["ko", "en"]}


def get_reader(source_lang: str):
    """Load EasyOCR once per language (model load takes a few seconds)."""
    import easyocr

    with _lock:
        if source_lang not in _readers:
            _readers[source_lang] = easyocr.Reader(
                EASYOCR_LANGS[source_lang], gpu=config.USE_GPU, verbose=False
            )
        return _readers[source_lang]


def detect_lines(img: np.ndarray, source_lang: str) -> list[Box]:
    reader = get_reader(source_lang)
    horizontal, free = reader.detect(img, min_size=8, text_threshold=0.6, low_text=0.35,
                                     width_ths=0.3, height_ths=0.5, ycenter_ths=0.5)
    boxes: list[Box] = []
    for x1, x2, y1, y2 in horizontal[0]:
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
    for quad in free[0]:  # rotated text comes back as 4 points
        xs, ys = [p[0] for p in quad], [p[1] for p in quad]
        boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
    h, w = img.shape[:2]
    return [(max(0, a), max(0, b), min(w, c), min(h, d)) for a, b, c, d in boxes
            if c - a > 4 and d - b > 4]


def _close(a: Box, b: Box, gap: int) -> bool:
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])


def group_lines(lines: list[Box], gap_ratio: float = 0.6) -> list[list[Box]]:
    """Merge line boxes that sit close together (union-find on proximity).

    The allowed gap scales with the text size so small and large fonts both work.
    """
    n = len(lines)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = lines[i], lines[j]
            size = min(min(a[2] - a[0], a[3] - a[1]), min(b[2] - b[0], b[3] - b[1]))
            if _close(a, b, int(size * gap_ratio) + 2):
                parent[find(i)] = find(j)

    groups: dict[int, list[Box]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(lines[i])
    return list(groups.values())


def reading_order(regions: list[TextRegion], img_h: int, rtl: bool = True) -> list[TextRegion]:
    """Manga reads top-to-bottom in bands, right-to-left within a band."""
    band = max(1, img_h // 8)

    def key(r: TextRegion):
        x_key = -r.box[2] if rtl else r.box[0]
        return (r.box[1] // band, x_key)

    return sorted(regions, key=key)


def ink_runs(profile: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """Split a 1-D ink projection into runs, merging runs separated by < min_gap blanks."""
    runs, start, last = [], None, -10**9
    for i, v in enumerate(profile):
        if v:
            if start is None:
                start = i
            elif i - last > min_gap:
                runs.append((start, last + 1))
                start = i
            last = i
    if start is not None:
        runs.append((start, last + 1))
    return runs


def split_columns(img: np.ndarray, box: Box) -> list[Box] | None:
    """If the region holds vertical text, return one box per column (right to left).

    Uses a vertical projection of dark pixels: columns of text are separated by blank strips.
    """
    import cv2

    x1, y1, x2, y2 = box
    gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
    ink = gray < 110
    if ink.sum() < 20:
        return None
    est = max(4, min(x2 - x1, y2 - y1) // 12)
    cols = ink_runs(ink.any(axis=0), min_gap=est)
    boxes = []
    for a, b in cols:
        rows = np.where(ink[:, a:b].any(axis=1))[0]
        if len(rows) == 0:
            continue
        boxes.append((x1 + a, y1 + int(rows[0]), x1 + b, y1 + int(rows[-1]) + 1))
    if not boxes:
        return None
    tall = all((bb[3] - bb[1]) > 1.6 * (bb[2] - bb[0]) for bb in boxes)
    return sorted(boxes, key=lambda bb: -bb[2]) if tall else None


def detect_regions(img: np.ndarray, source_lang: str) -> list[TextRegion]:
    lines = detect_lines(img, source_lang)
    regions = []
    for group in group_lines(lines):
        box = union(group)
        vertical = False
        # Japanese/Chinese bubbles are usually vertical columns read right-to-left
        if source_lang in ("ja", "zh"):
            cols = split_columns(img, box)
            if cols:
                group, vertical = cols, True
        regions.append(TextRegion(id=0, box=union(group), line_boxes=group, vertical=vertical))
    ordered = reading_order(regions, img.shape[0], rtl=source_lang in ("ja", "zh"))
    for i, r in enumerate(ordered):
        r.id = i
    return ordered
