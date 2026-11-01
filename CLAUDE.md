# Manga Translator: notes for Claude Code

Web app that translates manga pages: detect → OCR → translate (Claude API) → clean → typeset.
Portfolio project; keep code readable and well commented over clever.

## Layout
- `backend/app/main.py`: FastAPI routes; also serves `frontend/` as static files
- `backend/app/jobs.py`: in-memory job store + single worker thread; files in `data/<job_id>/`.
  A `Batch` is a list of page job ids (saved in `data/batches/`).
- `backend/app/archives.py`: uploads (images, ZIP/CBZ/EPUB, RAR/CBR, PDF) -> ordered pages
- `backend/app/pipeline/`: one module per step; `__init__.run_pipeline` wires them together
  - `ctd.py`: comic-text-detector ONNX model (downloads to `models/` on first use); gives boxes + text mask
  - `translate.py`: one Claude call per page; sends page + bubble crops (vision), styles, glossary
  - `fonts.py`: picks a font whose cmap covers the text (any script); `typeset.py` wraps CJK per
    character and shapes Arabic/Persian (RTL)
- Language codes live in `config.py` (SOURCE_LANGS incl. "auto", TARGET_LANGS, LANG_NAMES_EN,
  VERTICAL_SOURCES, CHAR_WRAP_TARGETS, RTL_TARGETS)
  - `models.TextRegion` is the shared type (box, line_boxes, source/translated text, vertical, on_light_bg, render_box)
- `frontend/`: plain HTML/CSS/JS, no build step. Colors are CSS variables with a
  `[data-theme="dark"]` override; never hard-code colors.
- `samples/make_sample.py`: generates a synthetic test page
- `tests/`: pytest; `slow` marker = loads OCR models

## Commands
- Run: `uvicorn backend.app.main:app --reload` (from repo root)
- Fast tests: `pytest -m "not slow"`; all: `pytest`
- Visual check: run the pipeline on `samples/sample_page.png` and look at the output image

## Conventions
- Boxes are `(x1, y1, x2, y2)` ints in image pixels. Images are RGB numpy arrays.
- Cast numpy scalars to Python types before they reach job JSON (FastAPI can't serialize `np.bool_`).
- Translation is one LLM call per page with JSON in/out; keep `translate_mock` working so the app runs without a key.
- Config comes from env vars in `backend/app/config.py`; never hardcode keys.

- Settings live in `config.py`: DETECTOR (ctd|easyocr), CLAUDE_VISION, UPPERCASE_TEXT, FONT_PATH.
- Don't commit copyrighted manga pages; use `samples/make_sample.py` for fixtures.

## Next up (see README roadmap)
Context across pages in a batch, then LaMa inpainting.
