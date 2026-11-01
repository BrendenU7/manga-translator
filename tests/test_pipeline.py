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
               TextRegion(id=1, box=(0, 0, 1, 1), source_text="待て")]
    reply = 'Sure! {"translations": [{"id": 0, "text": "Hey"}, {"id": 1, "text": "Wait!"}]}'
    monkeypatch.setattr(translate, "translate_claude",
                        lambda *a: {int(t["id"]): t["text"]
                                    for t in translate._extract_json(reply)["translations"]})
    monkeypatch.setattr(translate.config, "TRANSLATOR", "claude")
    out = translate.translate_regions(regions, "ja", "en")
    assert [r.translated_text for r in out] == ["Hey", "Wait!"]


def test_prompt_contains_all_bubbles_in_order():
    regions = [TextRegion(id=i, box=(0, 0, 1, 1), source_text=t) for i, t in enumerate(["一", "二"])]
    system, messages = translate.build_messages(regions, "ja", "en")
    assert "English" in system
    payload = json.loads(messages[0]["content"].split("Bubbles:\n", 1)[1])
    assert [b["text"] for b in payload] == ["一", "二"]


@pytest.mark.slow
def test_end_to_end_on_sample_page(monkeypatch):
    from backend.app.pipeline import run_pipeline
    monkeypatch.setattr(translate.config, "TRANSLATOR", "mock")
    img = np.array(Image.open(SAMPLE).convert("RGB"))
    regions, cleaned, result = run_pipeline(img, "ja", "en")
    assert len(regions) == 4
    assert sum(r.vertical for r in regions) == 3
    assert all(r.translated_text for r in regions)
    assert result.shape == img.shape
