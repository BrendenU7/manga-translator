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
