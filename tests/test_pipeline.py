"""Fast unit tests (no OCR models needed) + one slow end-to-end test.

Run fast tests only:  pytest -m "not slow"
"""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from backend.app.pipeline import translate
from backend.app.pipeline.clean import clean_regions, is_light_background
from backend.app.pipeline.detect import group_lines, ink_runs, reading_order
from backend.app.pipeline.models import TextRegion
from backend.app.pipeline.typeset import fit_text, wrap_text, get_font

SAMPLE = Path(__file__).parents[1] / "samples" / "sample_page.png"


def test_group_lines_merges_close_boxes_only():
    lines = [(100, 10, 120, 100), (70, 10, 90, 100),   # two columns of one bubble
             (400, 300, 420, 380)]                      # a separate bubble
    groups = group_lines(lines)
    assert sorted(len(g) for g in groups) == [1, 2]


def test_reading_order_is_right_to_left_for_manga():
    left = TextRegion(id=0, box=(10, 10, 50, 50))
    right = TextRegion(id=0, box=(300, 12, 340, 50))
    lower = TextRegion(id=0, box=(300, 500, 340, 540))
    ordered = reading_order([left, lower, right], img_h=800, rtl=True)
    assert ordered == [right, left, lower]


def test_ink_runs_merges_small_gaps():
    profile = np.array([0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0], bool)
    assert ink_runs(profile, min_gap=2) == [(1, 5), (9, 11)]


def test_wrap_and_fit_text_stay_inside_box():
    box = (0, 0, 120, 90)
    font, lines, line_h = fit_text("Where do you think you're going?", box)
    assert line_h * len(lines) <= 90
    assert all(font.getlength(line) <= 120 for line in lines)
    assert wrap_text("one two three", get_font(12), 10000) == ["one two three"]


def test_clean_erases_text_in_white_bubble():
    img = Image.new("RGB", (200, 200), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((90, 40, 110, 160), fill="black")  # fake text column
    arr = np.array(img)
    region = TextRegion(id=0, box=(90, 40, 110, 160), line_boxes=[(90, 40, 110, 160)])
    gray = arr.mean(axis=2).astype(np.uint8)
    assert is_light_background(gray, region.box)
    out = clean_regions(arr, [region])
    assert out[40:160, 90:110].min() > 200


def test_translate_parses_model_json(monkeypatch):
    regions = [TextRegion(id=0, box=(0, 0, 1, 1), source_text="おい"),
               TextRegion(id=1, box=(0, 0, 1, 1), source_text="待そ")]  # OCR typo
    reply = ('Sure! {"translations": [{"id": 0, "source": "おい", "text": "Hey"}, '
             '{"id": 1, "source": "待て", "text": "Wait!"}]}')
    monkeypatch.setattr(translate, "translate_claude", lambda *a, **k: translate.parse_response(reply))
    monkeypatch.setattr(translate.config, "TRANSLATOR", "claude")
    out = translate.translate_regions(regions, "ja", "en")
    assert [r.translated_text for r in out] == ["Hey", "Wait!"]
    assert out[1].source_text == "待て"  # Claude's reading replaces the OCR guess


def test_text_prompt_contains_all_bubbles_in_order():
    regions = [TextRegion(id=i, box=(0, 0, 1, 1), source_text=t) for i, t in enumerate(["一", "二"])]
    system, messages = translate.build_messages(regions, "ja", "en", glossary="悟飯 = Gohan")
    assert "English" in system and "悟飯 = Gohan" in system
    payload = json.loads(messages[0]["content"].split("\n", 1)[1])
    assert [b["ocr"] for b in payload] == ["一", "二"]


def test_vision_prompt_sends_page_and_one_image_per_bubble(monkeypatch):
    monkeypatch.setattr(translate.config, "CLAUDE_VISION", True)
    page = np.full((300, 200, 3), 255, np.uint8)
    regions = [TextRegion(id=0, box=(10, 10, 40, 90), source_text="a"),
               TextRegion(id=1, box=(100, 100, 130, 200), source_text="b")]
    system, messages = translate.build_messages(regions, "ja", "en", page=page, style="faithful")
    content = messages[0]["content"]
    images = [c for c in content if c["type"] == "image"]
    assert len(images) == 3  # full page + 2 bubbles
    assert images[0]["source"]["media_type"] == "image/png"
    assert "Scanlation" in system


@pytest.mark.slow
def test_end_to_end_on_sample_page(monkeypatch):
    from backend.app.pipeline import run_pipeline
    monkeypatch.setattr(translate.config, "TRANSLATOR", "mock")
    img = np.array(Image.open(SAMPLE).convert("RGB"))
    regions, cleaned, result = run_pipeline(img, "ja", "en")
    assert len(regions) >= 4
    assert sum(r.vertical for r in regions) >= 3
    assert all(r.translated_text for r in regions)
    assert result.shape == img.shape


# --------------------------------------------------------------------- languages

import importlib

from backend.app.pipeline import fonts

ts = importlib.import_module("backend.app.pipeline.typeset")  # module, not the typeset() function


def _covers(path, text):
    return {ord(c) for c in text if not c.isspace()} <= fonts._charset(path)


def test_font_picker_covers_each_script():
    for text in ["Hey, wait up!", "Эй, подожди!", "喂，等等我！"]:
        path = fonts.font_path_for(text)
        assert path and _covers(path, text), text


def test_chinese_wraps_per_character_without_breaking_before_punctuation():
    font = ts.get_font(20, "今天真的好热啊。")
    lines = ts.wrap_text("今天真的好热啊。", font, 70, "zh-Hans")
    assert len(lines) > 1
    assert all(not line.startswith("。") for line in lines)
    assert "".join(lines) == "今天真的好热啊。"


def test_thai_wraps_at_spaces_only():
    font = ts.get_font(20, "วันนี้ร้อนจริงๆ เลยนะ")
    lines = ts.wrap_text("วันนี้ร้อนจริงๆ เลยนะ", font, 60, "th")
    assert lines == ["วันนี้ร้อนจริงๆ", "เลยนะ"]


def test_arabic_is_shaped_for_display():
    shaped = ts.shape("مرحبا", "ar")
    assert shaped != "مرحبا"            # joined letter forms, display order
    assert ts.shape("hello", "en") == "hello"


def test_caps_rules():
    assert ts.prepare("istanbul", "tr") == "İSTANBUL"
    assert ts.prepare("hey", "en") == "HEY"
    assert ts.prepare("今天", "zh-Hans") == "今天"


def test_every_target_language_has_a_prompt_name():
    from backend.app import config
    assert set(config.TARGET_LANGS) <= set(config.LANG_NAMES_EN)
    assert set(config.SOURCE_LANGS) <= set(config.LANG_NAMES_EN)
    assert len(config.TARGET_LANGS) == 27


# --------------------------------------------------------------------- webtoon strips

from backend.app.pipeline import detect as det


def _strip(gutters_at, h=4000, w=800):
    """Grey 'artwork' with white gutters at the given rows."""
    img = np.full((h, w, 3), 120, np.uint8)
    img[:, ::7] = 60  # texture so non-gutter rows aren't uniform
    for y in gutters_at:
        img[y:y + 80] = 255
    return img


def test_strip_is_cut_in_the_gutters():
    img = _strip([1200, 2500])
    slices = det.slice_strip(img)
    assert slices[0][0] == 0 and slices[-1][1] == 4000
    cuts = [b for _, b in slices[:-1]]
    assert all(any(g <= c <= g + 80 for g in (1200, 2500)) for c in cuts), cuts


def test_strip_without_gutters_overlaps_slices():
    slices = det.slice_strip(_strip([]))
    assert all(nxt[0] < cur[1] for cur, nxt in zip(slices, slices[1:]))  # overlapping
    assert slices[-1][1] == 4000


def test_normal_pages_are_not_strips():
    assert not det.is_long_strip(np.zeros((1200, 850, 3), np.uint8))
    assert det.is_long_strip(np.zeros((6000, 800, 3), np.uint8))


def test_duplicate_detections_from_overlap_are_merged():
    a = TextRegion(id=0, box=(100, 1000, 300, 1100), line_boxes=[(100, 1000, 300, 1100)])
    b = TextRegion(id=0, box=(100, 1020, 300, 1140), line_boxes=[(100, 1020, 300, 1140)])
    c = TextRegion(id=0, box=(100, 2000, 300, 2100), line_boxes=[(100, 2000, 300, 2100)])
    merged = det._merge_regions([a, b, c])
    assert len(merged) == 2
    assert (100, 1000, 300, 1140) in [m.box for m in merged]


def test_strip_reads_top_to_bottom_bubble_by_bubble():
    # Two bubbles 300px apart on a 6000px strip must not land in the same reading band
    upper_left = TextRegion(id=0, box=(50, 1000, 200, 1100))
    lower_right = TextRegion(id=0, box=(500, 1300, 700, 1400))
    ordered = det.reading_order([lower_right, upper_left], img_h=6000, rtl=True, img_w=800)
    assert ordered == [upper_left, lower_right]


def test_long_pages_translate_in_chunks_with_context(monkeypatch):
    calls = []

    def fake(chunk, s, t, page, style, glossary, previous, **kw):
        calls.append(([r.id for r in chunk], len(previous)))
        return {r.id: {"source": "", "text": f"t{r.id}"} for r in chunk}

    monkeypatch.setattr(translate, "translate_mock", fake)
    monkeypatch.setattr(translate.config, "TRANSLATOR", "mock")
    regions = [TextRegion(id=i, box=(0, 0, 1, 1), source_text="x") for i in range(65)]
    out = translate.translate_regions(regions, "ko", "en")
    assert [len(ids) for ids, _ in calls] == [30, 30, 5]
    assert [ctx for _, ctx in calls] == [0, 30, 60]  # each chunk sees the earlier ones
    assert out[64].translated_text == "t64"


def test_strip_prompt_skips_the_full_page_image(monkeypatch):
    monkeypatch.setattr(translate.config, "CLAUDE_VISION", True)
    strip = np.full((5000, 800, 3), 255, np.uint8)
    regions = [TextRegion(id=0, box=(10, 10, 200, 90), source_text="a")]
    _, messages = translate.build_messages(regions, "ko", "en", page=strip)
    images = [c for c in messages[0]["content"] if c["type"] == "image"]
    assert len(images) == 1  # just the bubble


def test_long_strip_can_stop_between_chunks(monkeypatch):
    """A stop request takes effect between the Claude calls of a long strip."""
    regions = [TextRegion(id=i, box=(0, i * 10, 10, i * 10 + 5), source_text="x")
               for i in range(translate.MAX_BUBBLES_PER_CALL * 3)]
    calls = []

    def before_call():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("stopped")

    try:
        translate.translate_regions(regions, "ko", "en", before_call=before_call)
    except RuntimeError:
        pass
    assert len(calls) == 2  # the third chunk was never sent


def test_text_without_bubble_does_not_spread_over_artwork():
    """Open-paper text next to artwork: the drawing box widens only into blank paper,
    and the font is capped near the size of the original lettering."""
    import cv2
    import importlib
    ts = importlib.import_module("backend.app.pipeline.typeset")
    img = np.full((330, 300, 3), 255, np.uint8)
    for x in range(215, 300, 4):
        cv2.line(img, (x, 40), (x, 330), (40, 40, 40), 2)   # artwork on the right
    cols = [(95 + i * 26, 60, 117 + i * 26, 270) for i in range(4)]
    r = TextRegion(id=0, box=(95, 60, 199, 270), line_boxes=cols, vertical=True,
                   translated_text="Doesn't that mean even you do it all the time?")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    box = ts.bubble_box(gray, r, [r])
    assert box[2] <= 214                    # stopped before the artwork
    assert ts.source_font_cap(r) <= 27      # ~1.2x the 22px original characters


def test_neighbouring_text_blocks_never_share_space():
    """Two text blocks in one big white area must not be drawn on top of each other."""
    import importlib
    ts = importlib.import_module("backend.app.pipeline.typeset")
    img = np.full((300, 400, 3), 255, np.uint8)
    img[0:4, :] = img[-4:, :] = 0       # panel border so the flood fill stays inside
    img[:, 0:4] = img[:, -4:] = 0
    a = TextRegion(id=0, box=(120, 60, 170, 240), line_boxes=[(120, 60, 170, 240)], vertical=True,
                   translated_text="A fairly long first line of dialogue goes here.")
    b = TextRegion(id=1, box=(220, 60, 270, 240), line_boxes=[(220, 60, 270, 240)], vertical=True,
                   translated_text="And a second, equally long line right next to it.")
    ts.typeset(img, [a, b])
    assert not ts._overlaps(a.render_box, b.render_box)
    assert a.render_box[2] <= b.render_box[0]


def test_sound_effects_are_left_untouched(monkeypatch):
    """Claude returns "" for sound effects: those regions are neither erased nor redrawn."""
    from backend.app import pipeline
    from backend.app.pipeline import translate
    assert "Do NOT translate sound effects" in translate.SYSTEM_PROMPT
    img = np.full((200, 300, 3), 255, np.uint8)
    img[40:60, 20:80] = 0          # "dialogue" lettering
    img[120:180, 180:280] = 30     # big drawn sound effect
    regions = [TextRegion(id=0, box=(20, 40, 80, 60), line_boxes=[(20, 40, 80, 60)]),
               TextRegion(id=1, box=(180, 120, 280, 180), line_boxes=[(180, 120, 280, 180)],
                          on_light_bg=False)]

    def fake_translate(regs, *a, **kw):
        for r in regs:
            r.translated_text = "Hi!" if r.id == 0 else ""
        return regs

    monkeypatch.setattr(pipeline, "translate_regions", fake_translate)
    kept = pipeline.translate_page(img, regions, "ja", "en", "standard", "", [], lambda s: None)
    assert [r.translated_text for r in kept] == ["Hi!"]
    _, out = pipeline.finish_page(img, kept, None, "en", lambda s: None)
    assert np.array_equal(out[120:180, 180:280], img[120:180, 180:280])  # SFX unchanged
    assert not np.array_equal(out[40:60, 20:80], img[40:60, 20:80])      # dialogue replaced


def test_font_cap_works_when_columns_were_not_split():
    """If the detector returns one box for a whole block, the cap uses area per character."""
    import importlib
    ts = importlib.import_module("backend.app.pipeline.typeset")
    box = (90, 90, 190, 230)
    r = TextRegion(id=0, box=box, line_boxes=[box], vertical=True, source_text="あ" * 60)
    assert ts.source_font_cap(r) <= 20   # ~15px lettering, not the 100px-wide block


def test_open_paper_text_stays_off_artwork_inside_a_panel():
    """Inside a panel border the flood fill covers the whole panel: the box must still
    stop at the artwork rather than use the rectangle around the white area."""
    import importlib, cv2
    ts = importlib.import_module("backend.app.pipeline.typeset")
    img = np.full((340, 300, 3), 255, np.uint8)
    cv2.rectangle(img, (2, 2), (297, 337), (0, 0, 0), 3)
    for x in range(215, 300, 4):
        cv2.line(img, (x, 40), (x, 340), (40, 40, 40), 2)
    cols = [(95 + i * 26, 60, 117 + i * 26, 270) for i in range(4)]
    r = TextRegion(id=0, box=(95, 60, 199, 270), line_boxes=cols, vertical=True,
                   source_text="あ" * 30, translated_text="Wait, you knew about this the whole time?")
    ts.typeset(img, [r])
    assert r.render_box[2] <= 214


def test_cleaning_keeps_bubble_outline_that_touches_the_text_box():
    import cv2
    from backend.app.pipeline.clean import clean_regions
    img = np.full((200, 200, 3), 255, np.uint8)
    cv2.ellipse(img, (100, 100), (60, 80), 0, 0, 360, (0, 0, 0), 3)   # bubble outline
    img[60:140, 85:115] = 0                                            # "lettering" block
    for y in range(62, 140, 8):
        img[y:y + 3, 85:115] = 255                                     # make it stroke-like
    box = (82, 18, 118, 150)               # detector box reaching the top of the outline
    mask = np.zeros((200, 200), np.uint8)
    mask[60:140, 85:115] = 255
    r = TextRegion(id=0, box=box, line_boxes=[box])
    out = clean_regions(img, [r], mask)
    assert out[60:140, 85:115].min() > 200            # lettering erased
    assert out[18:24, 95:105].min() < 60               # outline at the top still there


def test_page_lettering_size_is_consistent():
    """Every bubble on a page uses the same standard size unless it's too small for it."""
    import importlib
    ts = importlib.import_module("backend.app.pipeline.typeset")
    assert ts.page_font_size((1536, 1024)) == 20
    assert ts.page_font_size((20000, 800)) == 16          # long strip: by width
    img = np.full((1536, 1024, 3), 255, np.uint8)
    big = TextRegion(id=0, box=(100, 100, 400, 500), line_boxes=[(100, 100, 400, 500)],
                     translated_text="Short line.")
    small = TextRegion(id=1, box=(600, 100, 700, 300), line_boxes=[(600, 100, 700, 300)],
                       translated_text="Another short line.")
    ts.typeset(img, [big, small])
    sizes = [ts.fit_text(ts.prepare(r.translated_text, "en"), r.render_box,
                         max_size=min(ts.source_font_cap(r), 20))[0].size for r in (big, small)]
    assert sizes == [20, 20]
