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


def _is_blank_column(gray: np.ndarray | None, x: int, y1: int, y2: int) -> bool:
    """True if this column of the (cleaned) page is almost all paper, not artwork."""
    if gray is None:
        return True
    col = gray[y1:y2, x]
    return col.size == 0 or np.mean(col > 200) >= 0.97


def widen_box(region: TextRegion, others: list[TextRegion], img_w: int,
              gray: np.ndarray | None = None) -> Box:
    """Give text without a closed bubble more width.

    Japanese is set in tall, narrow columns; English needs a wider box. Grow the box
    sideways (centered) until it's about as wide as it is tall, without running into
    other text regions, off the page, or over artwork: on white paper we only grow
    into blank columns. Text that sits on artwork only grows a little (its outline
    would cover the drawing).
    """
    x1, y1, x2, y2 = region.box
    w, h = x2 - x1, y2 - y1
    target = max(w, int(h * 0.9))
    if not region.on_light_bg:
        target = min(target, int(w * 1.3))
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
    if region.on_light_bg and gray is not None:
        # Walk outward one column at a time and stop at the first one with artwork in it
        left = x1
        while left - 1 >= nx1 and _is_blank_column(gray, left - 1, y1, y2):
            left -= 1
        right = x2
        while right < nx2 and _is_blank_column(gray, right, y1, y2):
            right += 1
        nx1, nx2 = left, right
    return (min(nx1, x1), y1, max(nx2, x2), y2)


def source_font_cap(region: TextRegion) -> int:
    """Largest sensible font size: about the size of the original lettering.

    A translation drawn much bigger than the text it replaces spills over the art, so
    use the original character size (column width for vertical text, line height for
    horizontal) as a ceiling.
    """
    if not region.line_boxes:
        return MAX_FONT
    sizes = sorted((b[2] - b[0]) if region.vertical else (b[3] - b[1])
                   for b in region.line_boxes)
    char = sizes[len(sizes) // 2]  # median, so one odd line can't skew it
    return int(min(MAX_FONT, max(MIN_FONT + 3, char * 1.2)))


def bubble_box(clean_gray: np.ndarray, region: TextRegion,
               others: list[TextRegion] | None = None) -> Box:
    """Where to draw the translation for a region.

    For text in a white bubble: flood-fill the bubble to find its interior, shrunk to
    stay off the outline. If there's no closed bubble, widen the text's own box.
    """
    x1, y1, x2, y2 = region.box
    others = others or []
    fallback = widen_box(region, others, clean_gray.shape[1], clean_gray)
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


def fit_text(text: str, box: Box, target_lang: str = "en", font_style: str = "manga",
             max_size: int = MAX_FONT) -> tuple[ImageFont.FreeTypeFont, list[str], int]:
    """Largest font size (up to max_size) whose wrapped text fits inside box.

    Returns (font, display lines, line_height). Lines are already shaped for drawing.
    """
    bw, bh = box[2] - box[0], box[3] - box[1]
    path = fonts.font_path_for(text, font_style)
    complex_layout = target_lang == "th"
    best = None
    lo, hi = MIN_FONT, max(MIN_FONT, min(max_size, MAX_FONT))
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


def _overlaps(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def separate_boxes(regions: list[TextRegion]) -> None:
    """Make sure no two drawing boxes overlap.

    Two text blocks in the same white area (or one bubble shared by two lines of
    dialogue) each find the same empty space and would draw on top of each other.
    Split the space between them: the border goes halfway across the gap between
    their original text, so each keeps the side it started on.
    """
    for i, a in enumerate(regions):
        for b in regions[i + 1:]:
            if a.render_box is None or b.render_box is None:
                continue
            if not _overlaps(a.render_box, b.render_box):
                continue
            ra, rb = list(a.render_box), list(b.render_box)
            oa, ob = a.box, b.box
            gap_x = max(ob[0] - oa[2], oa[0] - ob[2])   # space between the original texts
            gap_y = max(ob[1] - oa[3], oa[1] - ob[3])
            if gap_x >= 0 and gap_x >= gap_y:            # side by side: split left/right
                left, right = (ra, rb) if oa[0] < ob[0] else (rb, ra)
                lo, ro = (oa, ob) if oa[0] < ob[0] else (ob, oa)
                mid = (lo[2] + ro[0]) // 2
                left[2] = min(left[2], max(lo[2], mid - 2))
                right[0] = max(right[0], min(ro[0], mid + 2))
            elif gap_y >= 0:                             # stacked: split top/bottom
                top, bottom = (ra, rb) if oa[1] < ob[1] else (rb, ra)
                to, bo = (oa, ob) if oa[1] < ob[1] else (ob, oa)
                mid = (to[3] + bo[1]) // 2
                top[3] = min(top[3], max(to[3], mid - 2))
                bottom[1] = max(bottom[1], min(bo[1], mid + 2))
            else:                                        # original texts overlap too
                ra, rb = list(oa), list(ob)
            a.render_box, b.render_box = tuple(ra), tuple(rb)


def typeset(clean_img: np.ndarray, regions: list[TextRegion], target_lang: str = "en",
            font_style: str = "manga") -> np.ndarray:
    pil = Image.fromarray(clean_img)
    draw = ImageDraw.Draw(pil)
    gray = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)

    # Work out every drawing box first, so neighbours can be kept apart
    drawn = [r for r in regions if r.translated_text.strip()]
    for r in drawn:
        if r.render_box is None:
            r.render_box = bubble_box(gray, r, regions)
    separate_boxes(drawn)

    for r in drawn:
        text = prepare(r.translated_text.strip(), target_lang)
        box = r.render_box
        font, lines, line_h = fit_text(text, box, target_lang, font_style,
                                       max_size=source_font_cap(r))
        total_h = line_h * len(lines)
        y = box[1] + ((box[3] - box[1]) - total_h) // 2
        cx = (box[0] + box[2]) / 2
        stroke = 0 if r.on_light_bg else max(2, font.size // 8)
        for line in lines:
            draw.text((cx, y), line, font=font, fill=(0, 0, 0), anchor="ma",
                      stroke_width=stroke, stroke_fill=(255, 255, 255))
            y += line_h
    return np.array(pil)
