"""Step 2: read the text inside each region.

- Japanese: uses manga-ocr if it's installed (much better on vertical manga text),
  otherwise falls back to EasyOCR.
- Chinese / Korean: EasyOCR.
"""
import logging
import threading

import numpy as np
from PIL import Image

from .detect import get_reader
from .models import TextRegion

log = logging.getLogger(__name__)
_manga_ocr = None
_manga_ocr_failed = False
_lock = threading.Lock()


def get_manga_ocr():
    global _manga_ocr, _manga_ocr_failed
    with _lock:
        if _manga_ocr is None and not _manga_ocr_failed:
            try:
                from manga_ocr import MangaOcr
                _manga_ocr = MangaOcr()
            except Exception as e:  # not installed, or model download blocked
                log.info("manga-ocr unavailable (%s); using EasyOCR for Japanese", e)
                _manga_ocr_failed = True
        return _manga_ocr


def _crop(img: np.ndarray, box, pad: int = 4) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    return img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]


def column_to_strip(col: np.ndarray) -> np.ndarray:
    """Turn a vertical text column into a horizontal line EasyOCR can read.

    Cuts the column into characters at blank rows, then lays them out left to right.
    """
    import cv2

    from .detect import ink_runs

    gray = cv2.cvtColor(col, cv2.COLOR_RGB2GRAY)
    w = col.shape[1]
    chars = ink_runs((gray < 110).any(axis=1), min_gap=max(2, w // 8))
    # Re-join pieces of one character that got split (e.g. 三, 二): keep cells near-square
    cells: list[list[int]] = []
    for a, b in chars:
        if cells and (b - cells[-1][0]) <= w * 1.05:
            cells[-1][1] = b
        else:
            cells.append([a, b])
    if len(cells) <= 1:
        return col
    gap = max(4, w // 5)
    pieces = []
    for a, b in cells:
        cell = col[max(0, a - 2):b + 2]
        canvas = np.full((w + 4, cell.shape[1] + gap), 255, np.uint8)
        canvas = np.stack([canvas] * 3, axis=-1)
        top = (canvas.shape[0] - cell.shape[0]) // 2
        canvas[max(0, top):max(0, top) + min(cell.shape[0], canvas.shape[0]), gap // 2:gap // 2 + cell.shape[1]] = \
            cell[:canvas.shape[0]]
        pieces.append(canvas)
    return np.concatenate(pieces, axis=1)


def _easyocr_lines(img: np.ndarray, region: TextRegion, source_lang: str) -> str:
    reader = get_reader(source_lang)
    # Order lines: vertical text reads right-to-left by column, horizontal top-to-bottom
    if region.vertical:
        lines = sorted(region.line_boxes, key=lambda b: -b[2])
    else:
        lines = sorted(region.line_boxes, key=lambda b: (b[1], b[0]))
    parts = []
    for box in lines:
        crop = _crop(img, box)
        if region.vertical:
            crop = column_to_strip(crop)
        res = reader.recognize(crop, detail=0, paragraph=False)
        parts.append("".join(res) if region.vertical else " ".join(res))
    sep = "" if source_lang in ("ja", "zh") else " "
    return sep.join(p.strip() for p in parts if p.strip())


def read_region(img: np.ndarray, region: TextRegion, source_lang: str) -> str:
    if source_lang == "ja":
        mocr = get_manga_ocr()
        if mocr is not None:
            return mocr(Image.fromarray(_crop(img, region.box, pad=6))).strip()
    return _easyocr_lines(img, region, source_lang)


def read_regions(img: np.ndarray, regions: list[TextRegion], source_lang: str,
                 keep_empty: bool = False) -> list[TextRegion]:
    """OCR every region.

    Regions where OCR found nothing are usually false detections on artwork, so they're
    dropped, unless keep_empty is set (Claude will look at the image and decide).
    """
    for r in regions:
        r.source_text = read_region(img, r, source_lang)
    return regions if keep_empty else [r for r in regions if r.source_text]
