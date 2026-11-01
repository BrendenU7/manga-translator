"""Step 1: find the text regions (speech bubbles, captions) on a page.

Two detectors:
- "ctd" (default): comic-text-detector, trained on manga. Finds whole text blocks
  and also gives a pixel mask of the lettering. See ctd.py.
- "easyocr" (fallback): EasyOCR's generic CRAFT detector. Finds lines, which we merge
  into bubbles by proximity. Misses a lot of manga lettering.
"""
import logging
import threading

import numpy as np

from .. import config
from .models import Box, TextRegion, union

log = logging.getLogger(__name__)
_readers: dict[str, object] = {}
_lock = threading.Lock()
# EasyOCR / manga-ocr models aren't documented as thread-safe, and several pages run at
# once, so calls into them take this lock. (The ONNX text detector is thread-safe.)
MODEL_LOCK = threading.Lock()

# EasyOCR language codes for each source language
EASYOCR_LANGS = {"auto": ["ja", "en"], "ja": ["ja", "en"], "zh-Hans": ["ch_sim", "en"],
                 "zh-Hant": ["ch_tra", "en"], "ko": ["ko", "en"], "en": ["en"]}


def may_be_vertical(source_lang: str) -> bool:
    return source_lang in config.VERTICAL_SOURCES or source_lang == "auto"


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
    with MODEL_LOCK:
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


def split_columns(ink: np.ndarray, box: Box) -> list[Box] | None:
    """If a region holds vertical text, return one box per column (right to left).

    `ink` is a boolean "text pixel" crop of the region. Columns of text are separated
    by blank vertical strips, so a vertical projection splits them.
    """
    x1, y1, x2, y2 = box
    if ink.sum() < 20:
        return None
    est = max(4, min(x2 - x1, y2 - y1) // 12)
    boxes = []
    for a, b in ink_runs(ink.any(axis=0), min_gap=est):
        rows = np.where(ink[:, a:b].any(axis=1))[0]
        if len(rows):
            boxes.append((x1 + a, y1 + int(rows[0]), x1 + b, y1 + int(rows[-1]) + 1))
    if not boxes:
        return None
    tall = all((bb[3] - bb[1]) > 1.6 * (bb[2] - bb[0]) for bb in boxes)
    return sorted(boxes, key=lambda bb: -bb[2]) if tall else None


def _dark_ink(img: np.ndarray, box: Box) -> np.ndarray:
    import cv2
    x1, y1, x2, y2 = box
    return cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY) < 110


def _regions_easyocr(img: np.ndarray, source_lang: str) -> list[TextRegion]:
    """Generic detector: find text lines, then merge nearby lines into bubbles."""
    regions = []
    for group in group_lines(detect_lines(img, source_lang)):
        box = union(group)
        vertical = False
        if may_be_vertical(source_lang):
            cols = split_columns(_dark_ink(img, box), box)
            if cols:
                group, vertical = cols, True
        regions.append(TextRegion(id=0, box=union(group), line_boxes=group, vertical=vertical))
    return regions


def _regions_ctd(img: np.ndarray, source_lang: str) -> tuple[list[TextRegion], np.ndarray]:
    """Manga-trained detector: it already returns one box per text block."""
    from . import ctd

    boxes, mask = ctd.detect(img)
    regions = []
    for box in boxes:
        x1, y1, x2, y2 = box
        ink = mask[y1:y2, x1:x2] > 0
        cols = split_columns(ink, box) if may_be_vertical(source_lang) else None
        # Japanese/Chinese blocks are vertical unless clearly wider than tall. With auto-detect
        # we only call it vertical when it really looks like columns.
        tall_ratio = 1.3 if source_lang == "auto" else 0.8
        vertical = may_be_vertical(source_lang) and (cols is not None or (y2 - y1) >= tall_ratio * (x2 - x1))
        regions.append(TextRegion(id=0, box=box, line_boxes=cols or [box], vertical=vertical))
    return regions, mask


def _overlaps(a: Box, b: Box, frac: float = 0.3) -> bool:
    """True if the overlap covers more than `frac` of the smaller box."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return smaller > 0 and ix * iy > frac * smaller


def detect_regions(img: np.ndarray, source_lang: str) -> tuple[list[TextRegion], np.ndarray | None]:
    """Find text regions in reading order.

    Returns (regions, text_mask). text_mask is None when the fallback detector is used.
    """
    mask = None
    if config.DETECTOR == "ctd":
        try:
            regions, mask = _regions_ctd(img, source_lang)
            # Safety net: the manga detector sometimes skips small captions. Add anything
            # the generic detector finds that doesn't overlap a block we already have.
            # (Non-text hits are dropped later: OCR finds nothing / Claude returns "".)
            extra = [r for r in _regions_easyocr(img, source_lang)
                     if not any(_overlaps(r.box, c.box) for c in regions)]
            regions += extra
        except Exception as e:  # model download blocked, onnxruntime missing, ...
            log.warning("Manga text detector unavailable (%s); falling back to EasyOCR", e)
            regions = _regions_easyocr(img, source_lang)
    else:
        regions = _regions_easyocr(img, source_lang)

    # Manga/manhua read right-to-left. With auto-detect, decide from the layout:
    # mostly vertical text means a right-to-left page.
    if source_lang == "auto":
        rtl = sum(r.vertical for r in regions) * 2 > len(regions)
    else:
        rtl = source_lang in config.VERTICAL_SOURCES
    ordered = reading_order(regions, img.shape[0], rtl=rtl)
    for i, r in enumerate(ordered):
        r.id = i
    return ordered, mask
