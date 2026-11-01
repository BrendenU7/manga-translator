"""Manga translation pipeline: detect -> OCR -> translate -> clean -> typeset.

A page goes through three phases:
  1. prepare_page:   detect text regions + OCR           (CPU heavy)
  2. translate_page: one Claude call for the whole page   (waits on the network)
  3. finish_page:    erase the original text + letter     (CPU)

`run_pipeline` runs them back to back. For a chapter, jobs.py runs several pages at once
but lets each page translate only after the previous one, passing that page's dialogue
along as context (the `context` hooks below), so names and voices stay consistent.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .. import config
from .clean import clean_regions
from .detect import detect_regions
from .models import TextRegion
from .ocr import read_regions
from .translate import translate_regions
from .typeset import typeset

__all__ = ["run_pipeline", "rerender", "TextRegion", "PageContext"]

ProgressFn = Callable[[str], None]


@dataclass
class PageContext:
    """Hooks that link a page to its neighbours in a chapter.

    wait_for_previous(): blocks until earlier pages are translated, then returns their
        dialogue as [{"source": ..., "text": ...}, ...] (oldest first).
    publish(lines): shares this page's dialogue with the next page. Must always be
        called (even with []), or the next page waits until its timeout.
    """
    wait_for_previous: Callable[[], list[dict]]
    publish: Callable[[list[dict]], None]


def prepare_page(img: np.ndarray, source_lang: str, progress: ProgressFn):
    progress("detecting")
    regions, text_mask = detect_regions(img, source_lang)
    progress("reading")
    claude_reads_images = config.TRANSLATOR == "claude" and config.CLAUDE_VISION
    regions = read_regions(img, regions, source_lang, keep_empty=claude_reads_images)
    return regions, text_mask


def translate_page(img, regions, source_lang, target_lang, style, glossary,
                   previous: list[dict], progress: ProgressFn,
                   quality: str = "economy", usage: dict | None = None) -> list[TextRegion]:
    progress("translating")
    regions = translate_regions(regions, source_lang, target_lang, page=img, style=style,
                                glossary=glossary, previous=previous, quality=quality, usage=usage)
    # Claude returns "" for detections that aren't really text: leave those untouched
    regions = [r for r in regions if r.translated_text.strip()]
    for i, r in enumerate(regions):
        r.id = i
    return regions


def finish_page(img, regions, text_mask, target_lang, progress: ProgressFn, font_style: str = "manga"):
    progress("cleaning")
    cleaned = clean_regions(img, regions, text_mask)
    progress("typesetting")
    return cleaned, typeset(cleaned, regions, target_lang, font_style)


def run_pipeline(img: np.ndarray, source_lang: str, target_lang: str,
                 progress: ProgressFn = lambda s: None,
                 style: str = "standard", glossary: str = "",
                 context: PageContext | None = None, font_style: str = "manga",
                 quality: str = "economy", usage: dict | None = None):
    """Translate one page. Returns (regions, cleaned_image, translated_image).

    usage: optional dict; filled with the Claude tokens used and their estimated cost.
    """
    published = False
    try:
        regions, text_mask = prepare_page(img, source_lang, progress)
        previous = []
        if context:
            progress("waiting")  # for the previous page's translation
            previous = context.wait_for_previous()
        regions = translate_page(img, regions, source_lang, target_lang, style, glossary,
                                 previous, progress, quality, usage)
        if context:
            context.publish([{"source": r.source_text, "text": r.translated_text} for r in regions])
            published = True
        cleaned, result = finish_page(img, regions, text_mask, target_lang, progress, font_style)
        return regions, cleaned, result
    finally:
        if context and not published:
            context.publish([])  # failed before translating: don't leave the next page waiting


def rerender(cleaned: np.ndarray, regions: list[TextRegion], target_lang: str = "en",
             font_style: str = "manga") -> np.ndarray:
    """Redraw text after the user edits translations or changes font (no OCR/LLM calls)."""
    return typeset(cleaned, regions, target_lang, font_style)
