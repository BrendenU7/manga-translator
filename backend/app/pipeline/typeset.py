"""Step 5: draw the translated text back into each bubble.

1. Work out how much room we have: for white bubbles, flood-fill from the text area
   to find the bubble's full interior (translations are usually wider than the
   vertical Japanese columns they replace).
2. Find the largest font size where the word-wrapped text fits that box.
3. Draw it centered. Text over artwork gets a white outline so it stays readable.
"""
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import config
from .models import Box, TextRegion, area

MIN_FONT, MAX_FONT = 9, 64


@lru_cache(maxsize=64)
def get_font(size: int) -> ImageFont.FreeTypeFont:
    if config.FONT_PATH:
        return ImageFont.truetype(config.FONT_PATH, size)
    return ImageFont.load_default(size)


def bubble_box(clean_gray: np.ndarray, region: TextRegion) -> Box:
    """Bounding box of the white bubble around a region, shrunk to stay off the outline."""
    x1, y1, x2, y2 = region.box
    if not region.on_light_bg:
        return region.box
    h, w = clean_gray.shape
    binary = (clean_gray > 200).astype(np.uint8)
    seed = ((x1 + x2) // 2, (y1 + y2) // 2)
    if binary[seed[1], seed[0]] == 0:
        return region.box
    mask = np.zeros((h + 2, w + 2), np.uint8)
    _, _, _, rect = cv2.floodFill(binary.copy(), mask, seed, 2, flags=4 | (255 << 8))
    bx, by, bw, bh = rect
    touches_edge = bx <= 0 or by <= 0 or bx + bw >= w or by + bh >= h
    if touches_edge or bw * bh > 15 * area(region.box):
        return region.box  # leaked into the page background; don't trust it
    # A rectangle inscribed in an ellipse covers ~70% of its width/height
    sx, sy = int(bw * 0.15), int(bh * 0.15)
    inner = (bx + sx, by + sy, bx + bw - sx, by + bh - sy)
    # Never shrink below the original text area
    return (min(inner[0], x1), min(inner[1], y1), max(inner[2], x2), max(inner[3], y2))


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words, line = paragraph.split(), ""
        for word in words:
            trial = f"{line} {word}".strip()
            if font.getlength(trial) <= max_width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
    return lines


def fit_text(text: str, box: Box) -> tuple[ImageFont.FreeTypeFont, list[str], int]:
    """Largest font size whose wrapped text fits inside box. Returns (font, lines, line_height)."""
    bw, bh = box[2] - box[0], box[3] - box[1]
    best = None
    lo, hi = MIN_FONT, MAX_FONT
    while lo <= hi:
        size = (lo + hi) // 2
        font = get_font(size)
        lines = wrap_text(text, font, bw)
        line_h = int(size * 1.15)
        widest = max((font.getlength(line) for line in lines), default=0)
        if widest <= bw and line_h * len(lines) <= bh:
            best = (font, lines, line_h)
            lo = size + 1
        else:
            hi = size - 1
    if best is None:  # doesn't fit even at the minimum size; draw small anyway
        font = get_font(MIN_FONT)
        best = (font, wrap_text(text, font, bw), int(MIN_FONT * 1.15))
    return best


def typeset(clean_img: np.ndarray, regions: list[TextRegion]) -> np.ndarray:
    pil = Image.fromarray(clean_img)
    draw = ImageDraw.Draw(pil)
    gray = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)

    for r in regions:
        text = r.translated_text.strip()
        if not text:
            continue
        if r.render_box is None:
            r.render_box = bubble_box(gray, r)
        box = r.render_box
        font, lines, line_h = fit_text(text, box)
        total_h = line_h * len(lines)
        y = box[1] + ((box[3] - box[1]) - total_h) // 2
        cx = (box[0] + box[2]) / 2
        stroke = 0 if r.on_light_bg else max(2, font.size // 8)
        for line in lines:
            draw.text((cx, y), line, font=font, fill=(0, 0, 0), anchor="ma",
                      stroke_width=stroke, stroke_fill=(255, 255, 255))
            y += line_h
    return np.array(pil)
