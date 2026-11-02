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
# Standard lettering size, as a fraction of the page size. Professional scanlations (and
# ai-manga-translator) letter every bubble on a page at about the same size, shrinking only
# bubbles that are too small; filling each bubble with the biggest text that fits looks
# uneven and spills over the art. ~1/50 of the page width = ~20px on a 1024px page.
PAGE_FONT_RATIO = 1 / 50

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
    the original character size is used as a ceiling. Two estimates, take the smaller:
    - line size: column width (vertical) or line height (horizontal), when the detector
      split the block into lines. If it didn't, the one "line" is the whole block.
    - area per character: a block of N characters of size c covers about N * c^2,
      so c ~ sqrt(area / N). Works even when the lines weren't split.
    """
    estimates = []
    real_lines = [b for b in region.line_boxes if b != region.box] or (
        region.line_boxes if len(region.line_boxes) > 1 else [])
    if real_lines:
        sizes = sorted((b[2] - b[0]) if region.vertical else (b[3] - b[1]) for b in real_lines)
        estimates.append(sizes[len(sizes) // 2])  # median, so one odd line can't skew it
    n_chars = sum(1 for ch in region.source_text if not ch.isspace())
    if n_chars:
        estimates.append((area(region.box) / n_chars) ** 0.5)
    if not estimates:
        return MAX_FONT
    return int(min(MAX_FONT, max(MIN_FONT + 3, min(estimates) * 1.2)))


def _grow_inside(mask: np.ndarray, box: Box, max_w: int, max_h: int) -> Box:
    """Grow box one pixel at a time on each side while the new edge stays inside mask
    (the bubble's white interior), up to max_w x max_h. Widening goes first, because
    translations need width more than height. Result: a rectangle that fits inside the
    bubble's real shape, not the rectangle around it (which would include artwork)."""
    h, w = mask.shape
    x1, y1, x2, y2 = box

    def ok(line):  # a new edge may touch a few stray dark pixels, not more
        return line.size > 0 and np.mean(line) >= 0.97

    for horizontal in (True, False):
        moved = True
        while moved:
            moved = False
            if horizontal and x2 - x1 < max_w:
                if x1 > 0 and ok(mask[y1:y2, x1 - 1]):
                    x1 -= 1; moved = True
                if x2 < w and x2 - x1 < max_w and ok(mask[y1:y2, x2]):
                    x2 += 1; moved = True
            if not horizontal and y2 - y1 < max_h:
                if y1 > 0 and ok(mask[y1 - 1, x1:x2]):
                    y1 -= 1; moved = True
                if y2 < h and y2 - y1 < max_h and ok(mask[y2, x1:x2]):
                    y2 += 1; moved = True
    return (x1, y1, x2, y2)


def paper_mask(gray: np.ndarray) -> np.ndarray:
    """1 where the page is blank paper. Tiny dark specks (dust, leftovers of the erased
    text) count as paper: they aren't artwork."""
    paper = (gray > 200).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(1 - paper, connectivity=8)
    small = np.nonzero(stats[:, cv2.CC_STAT_AREA] < 40)[0]
    small = small[small != 0]  # label 0 is the paper itself
    paper[np.isin(labels, small)] = 1
    return paper


def bubble_box(clean_gray: np.ndarray, region: TextRegion,
               others: list[TextRegion] | None = None, text: str | None = None,
               target_lang: str = "en", font_style: str = "manga",
               paper: np.ndarray | None = None) -> Box:
    """Where to draw the translation for a region.

    Text on white (a bubble, or open paper): find the white area the text sat in and
    grow the text's box inside it, so the result never covers outlines or artwork, even
    when the bubble isn't fully closed. Text on artwork keeps (about) its own box.
    """
    others = others or []
    if not region.on_light_bg:
        return widen_box(region, others, clean_gray.shape[1], clean_gray)
    h, w = clean_gray.shape
    x1, y1, x2, y2 = region.box
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return region.box
    if paper is None:  # typeset() computes it once per page and passes it in
        paper = paper_mask(clean_gray)
    # The erased text area is white now; flood from it to get the connected white space
    seed = ((x1 + x2) // 2, (y1 + y2) // 2)
    if not paper[seed[1], seed[0]]:
        ys, xs = np.nonzero(paper[y1:y2, x1:x2])
        if not len(xs):
            return region.box
        seed = (x1 + int(xs[len(xs) // 2]), y1 + int(ys[len(ys) // 2]))
    fill = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(paper.copy(), fill, seed, 2, flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
    inside = fill[1:-1, 1:-1] > 0
    # Keep a little air between the text and the bubble outline / artwork
    inside = cv2.erode(inside.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    # Other text blocks are off limits (they get their own space)
    for o in others:
        if o is not region:
            ox1, oy1, ox2, oy2 = o.box
            inside[max(0, oy1 - 4):oy2 + 4, max(0, ox1 - 4):ox2 + 4] = False
    bw, bh = x2 - x1, y2 - y1
    # English wants a wider shape than tall Japanese columns; don't drift far though
    max_w = max(int(bw * 1.8), int(bh * 1.1))
    max_h = int(bh * 1.4)
    # Round bubbles are narrow at the top and bottom, so a box as tall as the original
    # text often can't widen at all. Try shorter (centered) starting boxes too, and keep
    # the one where the translation fits at the largest size.
    cy = (y1 + y2) // 2
    candidates = []
    for keep in (1.0, 0.85, 0.7, 0.55):
        half = max(1, int(bh * keep) // 2)
        start = (x1, max(y1, cy - half), x2, min(y2, cy + half))
        gx1, gy1, gx2, gy2 = _grow_inside(inside, start, max_w, max_h)
        # Keep it centered on the original text (it may have grown more on one side)
        cx = (x1 + x2) // 2
        half_w = max(1, min(cx - gx1, gx2 - cx))
        candidates.append((cx - half_w, gy1, cx + half_w, gy2))
    if not text:
        return max(candidates, key=area)
    cap = min(source_font_cap(region), page_font_size(clean_gray.shape))

    def score(b):
        font, _, _ = fit_text(text, b, target_lang, font_style, max_size=cap)
        return (font.size, area(b))
    return max(candidates, key=score)


def page_font_size(shape: tuple) -> int:
    """The page's standard lettering size. Uses the shorter of width and height/1.45
    (a normal page is ~1.45x taller than wide), so double-page spreads and long webtoon
    strips get the same size as a single page of the same width."""
    h, w = shape[:2]
    ref = min(w, h / 1.45)
    return int(min(MAX_FONT, max(MIN_FONT + 3, round(ref * PAGE_FONT_RATIO))))


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
        return font, lines, int(MIN_FONT * 1.15)
    return _balance(text, best, bw, bh, target_lang)


def _balance(text: str, fitted, bw: int, bh: int, target_lang: str):
    """Rewrap into even lines, the way letterers do.

    With a standard font size the text is often smaller than the bubble, and wrapping at
    the full width gives one long line and a short leftover ("MOCK TRANSLATION FOR /
    BUBBLE"). Instead: use the fewest lines that fill about 3/4 of the width, then make
    them equally long. Only used if it still fits the box.
    """
    font, lines, line_h = fitted
    total = font.getlength(shape(" ".join(text.split()), target_lang))
    n = max(1, int(np.ceil(total / (0.75 * bw))))
    width = total / n * 1.08  # a little slack so words don't spill onto an extra line
    for _ in range(12):  # widen step by step until it fits
        wrapped = [shape(line, target_lang) for line in wrap_text(text, font, int(width), target_lang)]
        widest = max((font.getlength(line) for line in wrapped), default=0)
        if widest <= bw and line_h * len(wrapped) <= bh:
            return font, wrapped, line_h
        if width >= bw:
            break
        width = min(bw, width * 1.12)
    return fitted


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

    page_size = page_font_size(clean_img.shape)
    # Work out every drawing box first, so neighbours can be kept apart
    drawn = [r for r in regions if r.translated_text.strip()]
    paper = paper_mask(gray) if any(r.render_box is None for r in drawn) else None
    for r in drawn:
        if r.render_box is None:
            r.render_box = bubble_box(gray, r, regions, prepare(r.translated_text.strip(), target_lang),
                                      target_lang, font_style, paper)
    separate_boxes(drawn)

    for r in drawn:
        text = prepare(r.translated_text.strip(), target_lang)
        box = r.render_box
        font, lines, line_h = fit_text(text, box, target_lang, font_style,
                                       max_size=min(source_font_cap(r), page_size))
        total_h = line_h * len(lines)
        y = box[1] + ((box[3] - box[1]) - total_h) // 2
        cx = (box[0] + box[2]) / 2
        stroke = 0 if r.on_light_bg else max(2, font.size // 8)
        for line in lines:
            draw.text((cx, y), line, font=font, fill=(0, 0, 0), anchor="ma",
                      stroke_width=stroke, stroke_fill=(255, 255, 255))
            y += line_h
    return np.array(pil)
