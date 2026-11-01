"""Manga translation pipeline: detect -> OCR -> translate -> clean -> typeset."""
from typing import Callable

import numpy as np

from .clean import clean_regions
from .detect import detect_regions
from .models import TextRegion
from .ocr import read_regions
from .translate import translate_regions
from .typeset import typeset

__all__ = ["run_pipeline", "rerender", "TextRegion"]

ProgressFn = Callable[[str], None]


def run_pipeline(img: np.ndarray, source_lang: str, target_lang: str,
                 progress: ProgressFn = lambda s: None):
    """Translate one page. Returns (regions, cleaned_image, translated_image)."""
    progress("detecting")
    regions = detect_regions(img, source_lang)
    progress("reading")
    regions = read_regions(img, regions, source_lang)
    progress("translating")
    regions = translate_regions(regions, source_lang, target_lang)
    progress("cleaning")
    cleaned = clean_regions(img, regions)
    progress("typesetting")
    result = typeset(cleaned, regions)
    return regions, cleaned, result


def rerender(cleaned: np.ndarray, regions: list[TextRegion]) -> np.ndarray:
    """Redraw text after the user edits translations (no OCR/LLM calls needed)."""
    return typeset(cleaned, regions)
