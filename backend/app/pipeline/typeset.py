"""Step 5: draw the translated text back into each bubble.

1. Work out how much room we have: for white bubbles, flood-fill from the text area
   to find the bubble's full interior (translations are usually wider than the
   vertical Japanese columns they replace).
2. Pick a font that has every character of the text (see fonts.py), then find the
   largest size where the wrapped text fits that box. Languages written without spaces
   (Chinese, Japanese, Thai) wrap per character; Arabic/Persian are shaped right-to-left.
3. Draw it centered. Text over artwork gets a white outline so it stays readable.
"""
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import config
from . import fonts
from .models import Box, TextRegion, area

MIN_FONT, MAX_FONT = 9, 64

# Characters that shouldn't start a line in CJK text (closing punctuation, small kana...)
NO_LINE_START = set("、。，．・：；？！…‥ー）」』】〉》〕！？ぁぃぅぇぉっゃゅょァィゥェォッャュョ")


def get_font(size: int, text: str = "A", style: str = "manga") -> ImageFont.FreeTypeFont:
    return fonts.load(fonts.font_path_for(text, style), size)


def shape(line: str, target_lang: str) -> str:
    """Arabic/Persian: join letters into their connected forms and order them for display."""
    if target_lang not in config.RTL_TARGETS:
        return line
    import arabic_reshaper
    from bidi.algorithm import get_display
    return get_display(arabic_reshaper.reshape(line))


def prepare(text: str, target_lang: str) -> str:
    """Comic-style ALL CAPS for scripts that have case."""
    if not config.UPPERCASE_TEXT or target_lang in config.NO_SPACE_TARGETS | config.RTL_TARGETS:
        return text
    if target_lang == "tr":  # Turkish dotted/dotless i
        text = text.replace("i", "İ").replace("ı", "I")
    return text.upper()


def widen_box(region: TextRegion, others: list[TextRegion], img_w: int) -> Box:
    """Give text without a closed bubble more width.

    Japanese is set in tall, narrow columns; English needs a wider box. Grow the box
    sideways (centered) until it's about as wide as it is tall, without running into
    other text regions or off the page.
    """
    x1, y1, x2, y2 = region.box
    w, h = x2 - x1, y2 - y1
    target = max(w, int(h * 0.9))
    if target <= w:
        return region.box
    cx = (x1 + x2) // 2
    nx1, nx2 = max(0, cx - target // 2), min(img_w, cx + target // 2)
    for o in others:
        if o is region:
            continue
        ox1, oy1, ox2, oy2 = o.box
        if oy2 <= y1 or oy1 >= y2:  # no vertical overlap, can't collide
            continue
        if ox2 <= x1:
            nx1 = max(nx1, ox2 + 4)
        elif ox1 >= x2:
            nx2 = min(nx2, ox1 - 4)
    return (min(nx1, x1), y1, max(nx2, x2), y2)


def bubble_box(clean_gray: np.ndarray, region: TextRegion,
               others: list[TextRegion] | None = None) -> Box:
    """Where to draw the translation for a region.

    For text in a white bubble: flood-fill the bubble to find its interior, shrunk to
    stay off the outline. If there's no closed bubble, widen the text's own box.
    """
    x1, y1, x2, y2 = region.box
    others = others or []
    fallback = widen_box(region, others, clean_gray.shape[1])
    if not region.on_light_bg:
        return fallback
    h, w = clean_gray.shape
    binary = (clean_gray > 200).astype(np.uint8)
    seed = ((x1 + x2) // 2, (y1 + y2) // 2)
    if binary[seed[1], seed[0]] == 0:
        return fallback
    mask = np.zeros((h + 2, w + 2), np.uint8)
    _, _, _, rect = cv2.floodFill(binary.copy(), mask, seed, 2, flags=4 | (255 << 8))
    bx, by, bw, bh = rect
    touches_edge = bx <= 0 or by <= 0 or bx + bw >= w or by + bh >= h
    if touches_edge or bw * bh > 15 * area(region.box):
        return fallback  # leaked into the page background; don't trust it
    # A rectangle inscribed in an ellipse covers ~70% of its width/height
    sx, sy = int(bw * 0.15), int(bh * 0.15)
    inner = (min(bx + sx, x1), min(by + sy, y1), max(bx + bw - sx, x2), max(by + bh - sy, y2))
    # Still too narrow for English? Widen toward the fallback, but stay inside the bubble
    if fallback[2] - fallback[0] > inner[2] - inner[0]:
        inner = (max(min(inner[0], fallback[0]), bx + 2), inner[1],
                 min(max(inner[2], fallback[2]), bx + bw - 2), inner[3])
    return inner


def _units(paragraph: str, per_char: bool) -> list[tuple[str, str]]:
    """Split a paragraph into (piece, joiner) units that lines may break between.

    Words for spaced languages; single characters for CJK/Thai, keeping combining marks
    (Thai vowels and tones) and no-line-start punctuation attached to the previous unit.
    """
    if not per_char:
        return [(w, " ") for w in paragraph.split()]
    units: list[str] = []
    for ch in paragraph:
        if ch.isspace():
            units.append(" ")
        elif units and (unicodedata.combining(ch) or unicodedata.category(ch) == "Mn"
                        or ch in NO_LINE_START):
            units[-1] += ch
        else:
            units.append(ch)
    return [(u, "") for u in units]


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int,
              target_lang: str = "en") -> list[str]:
    """Greedy line wrapping by width. Measures the shaped (display) form of each line."""
    per_char = target_lang in config.CHAR_WRAP_TARGETS
    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for piece, joiner in _units(paragraph, per_char):
            trial = f"{line}{joiner}{piece}" if line else piece
            if not line or font.getlength(shape(trial.strip(), target_lang)) <= max_width:
                line = trial
            else:
                lines.append(line.strip())
                line = piece.strip()
        if line.strip():
            lines.append(line.strip())
    return lines


def fit_text(text: str, box: Box, target_lang: str = "en",
             font_style: str = "manga") -> tuple[ImageFont.FreeTypeFont, list[str], int]:
    """Largest font size whose wrapped text fits inside box.

    Returns (font, display lines, line_height). Lines are already shaped for drawing.
    """
    bw, bh = box[2] - box[0], box[3] - box[1]
    path = fonts.font_path_for(text, font_style)
    complex_layout = target_lang == "th"
    best = None
    lo, hi = MIN_FONT, MAX_FONT
    while lo <= hi:
        size = (lo + hi) // 2
        font = fonts.load(path, size, complex_layout)
        lines = [shape(line, target_lang) for line in wrap_text(text, font, bw, target_lang)]
        line_h = int(size * 1.15)
        widest = max((font.getlength(line) for line in lines), default=0)
        if widest <= bw and line_h * len(lines) <= bh:
            best = (font, lines, line_h)
            lo = size + 1
        else:
            hi = size - 1
    if best is None:  # doesn't fit even at the minimum size; draw small anyway
        font = fonts.load(path, MIN_FONT, complex_layout)
        lines = [shape(line, target_lang) for line in wrap_text(text, font, bw, target_lang)]
        best = (font, lines, int(MIN_FONT * 1.15))
    return best


def typeset(clean_img: np.ndarray, regions: list[TextRegion], target_lang: str = "en",
            font_style: str = "manga") -> np.ndarray:
    pil = Image.fromarray(clean_img)
    draw = ImageDraw.Draw(pil)
    gray = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)

    for r in regions:
        text = r.translated_text.strip()
        if not text:
            continue
        text = prepare(text, target_lang)
        if r.render_box is None:
            r.render_box = bubble_box(gray, r, regions)
        box = r.render_box
        font, lines, line_h = fit_text(text, box, target_lang, font_style)
        total_h = line_h * len(lines)
        y = box[1] + ((box[3] - box[1]) - total_h) // 2
        cx = (box[0] + box[2]) / 2
        stroke = 0 if r.on_light_bg else max(2, font.size // 8)
        for line in lines:
            draw.text((cx, y), line, font=font, fill=(0, 0, 0), anchor="ma",
                      stroke_width=stroke, stroke_fill=(255, 255, 255))
            y += line_h
    return np.array(pil)
