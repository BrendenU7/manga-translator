"""Manga translation pipeline: detect -> OCR -> translate -> clean -> typeset."""
from typing import Callable

import numpy as np

from .. import config

from .clean import clean_regions
from .detect import detect_regions
from .models import TextRegion
from .ocr import read_regions
from .translate import translate_regions
from .typeset import typeset

__all__ = ["run_pipeline", "rerender", "TextRegion"]

ProgressFn = Callable[[str], None]


def run_pipeline(img: np.ndarray, source_lang: str, target_lang: str,
                 progress: ProgressFn = lambda s: None,
                 style: str = "standard", glossary: str = ""):
    """Translate one page. Returns (regions, cleaned_image, translated_image)."""
    progress("detecting")
    regions, text_mask = detect_regions(img, source_lang)
    progress("reading")
    claude_reads_images = config.TRANSLATOR == "claude" and config.CLAUDE_VISION
    regions = read_regions(img, regions, source_lang, keep_empty=claude_reads_images)
    progress("translating")
    regions = translate_regions(regions, source_lang, target_lang,
                                page=img, style=style, glossary=glossary)
    # Claude returns "" for detections that aren't really text: leave those untouched
    regions = [r for r in regions if r.translated_text.strip()]
    for i, r in enumerate(regions):
        r.id = i
    progress("cleaning")
    cleaned = clean_regions(img, regions, text_mask)
    progress("typesetting")
    result = typeset(cleaned, regions, target_lang)
    return regions, cleaned, result


def rerender(cleaned: np.ndarray, regions: list[TextRegion], target_lang: str = "en") -> np.ndarray:
    """Redraw text after the user edits translations (no OCR/LLM calls needed)."""
    return typeset(cleaned, regions, target_lang)
