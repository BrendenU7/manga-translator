"""Step 4: erase the original text.

If the detector gave us a text mask (comic-text-detector does), we erase exactly the
masked lettering pixels inside each region. Otherwise we guess the strokes from
pixel brightness inside each line box.

- Text inside a white speech bubble: paint over it with the bubble's color.
- Text over artwork: build a mask of just the text strokes and inpaint it, rebuilding the
  drawing behind it (LaMa model, see inpaint.py).
"""
import cv2
import numpy as np

from .inpaint import inpaint
from .models import Box, TextRegion


def pad_box(box: Box, p: int, w: int, h: int) -> Box:
    return (max(0, box[0] - p), max(0, box[1] - p), min(w, box[2] + p), min(h, box[3] + p))


def ring_pixels(img: np.ndarray, box: Box, width: int = 6) -> np.ndarray:
    """Pixels in a thin ring just outside the box: tells us the background color.

    Works on grayscale (returns 1-D) or RGB (returns N x 3).
    """
    h, w = img.shape[:2]
    outer = pad_box(box, width, w, h)
    patch = img[outer[1]:outer[3], outer[0]:outer[2]]
    mask = np.ones(patch.shape[:2], bool)
    mask[box[1] - outer[1]:box[3] - outer[1], box[0] - outer[0]:box[2] - outer[0]] = False
    return patch[mask]


def is_light_background(gray: np.ndarray, box: Box) -> bool:
    ring = ring_pixels(gray, box)
    return bool(ring.size > 0 and np.median(ring) > 200 and (ring > 180).mean() > 0.75)


def bubble_color(img: np.ndarray, box: Box) -> np.ndarray:
    h, w = img.shape[:2]
    ring = ring_pixels(img, pad_box(box, 3, w, h))
    return np.median(ring, axis=0).astype(np.uint8) if len(ring) else np.uint8([255] * 3)


def clean_regions(img: np.ndarray, regions: list[TextRegion],
                  text_mask: np.ndarray | None = None) -> np.ndarray:
    out = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    inpaint_mask = np.zeros((h, w), np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    for r in regions:
        r.on_light_bg = is_light_background(gray, r.box)
        color = bubble_color(img, r.box) if r.on_light_bg else None

        if text_mask is not None:
            # Use the detector's lettering mask, limited to this region (plus a little slack)
            size = min(r.box[2] - r.box[0], r.box[3] - r.box[1])
            box = pad_box(r.box, max(3, size // 10), w, h)
            m = text_mask[box[1]:box[3], box[0]:box[2]]
            m = cv2.dilate(m, kernel, iterations=3)
            if r.on_light_bg:
                # On a white background, the detector's box holds only lettering, so erase
                # every non-white pixel inside it; outside the box, trust the mask only.
                patch = gray[box[1]:box[3], box[0]:box[2]]
                strokes = (m > 0) | ((patch < 215) & _inside(r.box, box, 2))
                out[box[1]:box[3], box[0]:box[2]][strokes] = color
            else:
                inpaint_mask[box[1]:box[3], box[0]:box[2]] |= m
            continue

        for lb in r.line_boxes:
            size = min(lb[2] - lb[0], lb[3] - lb[1])
            box = pad_box(lb, max(2, size // 6), w, h)
            patch = gray[box[1]:box[3], box[0]:box[2]]
            if r.on_light_bg:
                # Repaint only the text strokes (plus a small halo), not the whole rectangle,
                # so we never paint over the bubble outline or artwork next to it
                strokes = (patch < 200).astype(np.uint8)
                strokes = cv2.dilate(strokes, kernel, iterations=2).astype(bool)
                strokes &= _inside(lb, box, max(2, size // 8))
                out[box[1]:box[3], box[0]:box[2]][strokes] = color
            else:
                # Mask only pixels that look like text strokes (very dark or very light)
                strokes = ((patch < 80) | (patch > 230)).astype(np.uint8) * 255
                strokes = cv2.dilate(strokes, kernel, iterations=2)
                inpaint_mask[box[1]:box[3], box[0]:box[2]] |= strokes

    if inpaint_mask.any():
        out = inpaint(out, inpaint_mask)  # LaMa (or OpenCV fallback), see inpaint.py
    return out


def _inside(line: Box, box: Box, pad: int) -> np.ndarray:
    """Boolean mask (in box coordinates) of the line box grown by `pad` pixels."""
    m = np.zeros((box[3] - box[1], box[2] - box[0]), bool)
    m[max(0, line[1] - pad - box[1]):line[3] + pad - box[1],
      max(0, line[0] - pad - box[0]):line[2] + pad - box[0]] = True
    return m
