# Manga Translator

Upload a manga page, or a whole chapter as a ZIP, CBZ, EPUB, RAR or PDF. The app finds the speech bubbles, reads the Japanese, Chinese or Korean
text, translates it with Claude, erases the original lettering and typesets the
translation back into the bubbles. You can then fix any line by hand and re-render.

![screenshot](docs/screenshot.png)

## How it works

```
upload ─▶ detect ─▶ OCR ─▶ translate ─▶ clean ─▶ typeset ─▶ review/edit
```

| Step | File | Approach |
|---|---|---|
| Detect | `pipeline/detect.py`, `pipeline/ctd.py` | [comic-text-detector](https://github.com/dmMaze/comic-text-detector), a model trained on manga pages, run with ONNX Runtime. It finds whole text blocks (vertical Japanese, hand-lettering next to screentone) and outputs a per-pixel text mask. Regions are sorted in manga reading order (right-to-left). EasyOCR's generic detector is the fallback. |
| OCR | `pipeline/ocr.py` | [manga-ocr](https://github.com/kha-white/manga-ocr) for Japanese if installed; otherwise EasyOCR. Vertical columns are cut into characters and laid out horizontally so EasyOCR can read them. |
| Translate | `pipeline/translate.py` | One Claude request per page. Claude gets the full page plus an image of every bubble with the OCR guess, reads the text itself (OCR on manga lettering is unreliable), and returns its reading plus the translation as JSON. Supports translation styles (standard / keep honorifics / faithful) and a glossary for names and terms. |
| Clean | `pipeline/clean.py` | Erases the masked lettering pixels. On white backgrounds they're repainted with the bubble color; over artwork they're filled in with OpenCV inpainting. |
| Typeset | `pipeline/typeset.py` | Flood-fills the bubble to find its real size (or widens the text box when there's no closed bubble), binary-searches the largest font that fits after word wrapping, and draws centered ALL-CAPS text in Comic Neue (with an outline over artwork). |

The backend is FastAPI. Each page is a background job, and the frontend polls for
progress. Edits only re-run the typesetting step, using the saved cleaned image.

**Batch upload** (`archives.py`): several images and/or documents at once. Archives are
unpacked and PDFs rendered into pages in natural order (page2 before page10); each page
becomes a job, and the UI shows overall progress, a page strip to review/edit any page,
and a "Download all" ZIP.

Supported uploads: JPG, PNG, WebP, GIF, BMP, AVIF, HEIC · ZIP, CBZ, EPUB · RAR, CBR
(needs [7-Zip](https://www.7-zip.org/) or UnRAR installed) · PDF.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your ANTHROPIC_API_KEY
uvicorn backend.app.main:app --reload
```

Open http://localhost:8000. The first run downloads the manga text detector (~95 MB, into
`models/`) and the EasyOCR models (~100 MB).
Without an API key the app runs in demo mode: detection, OCR and typesetting are real,
and translations are placeholders.

Better Japanese OCR: `pip install manga-ocr` (it's picked up automatically).

Lettering uses the bundled [Comic Neue](https://comicneue.com/) Bold (OFL licensed, see
`fonts/OFL.txt`). To use another font, put it at `fonts/comic.ttf` or set `FONT_PATH`.

Test page: `python samples/make_sample.py` generates `samples/sample_page.png`
(synthetic, so there's no copyright issue for demos).

## Tests

```bash
pytest -m "not slow"   # fast unit tests
pytest                 # includes an end-to-end run on the sample page
```

## API

| Method | Path | |
|---|---|---|
| `GET` | `/api/config` | Supported languages, translator mode |
| `POST` | `/api/jobs` | multipart: `file`, `source_lang`, `target_lang`, `style`, `glossary` → job |
| `POST` | `/api/batches` | multipart: `files` (repeat), same options → batch of page jobs |
| `GET` | `/api/batches/{id}` | Progress (`done` / `total`) and each page's status |
| `GET` | `/api/batches/{id}/download` | ZIP of all translated pages |
| `GET` | `/api/jobs/{id}` | Status, current stage, regions (source and translated text, boxes) |
| `GET` | `/api/jobs/{id}/image/{original,translated,cleaned}` | PNG |
| `PUT` | `/api/jobs/{id}/translations` | `{"translations":[{"id":0,"text":"..."}]}` → re-render |

## Roadmap

- [x] Batch: ZIP/CBZ/EPUB/RAR/PDF upload, split into pages, download as ZIP
- [ ] LaMa inpainting for text over artwork
- [x] Manga-specific text detector (comic-text-detector)
- [x] Claude reads bubble images (vision) instead of trusting OCR
- [x] Translation styles + glossary
- [ ] Context across pages in a batch (previous page's dialogue in the prompt)
- [ ] Font style choices
- [ ] Drag/resize text boxes in the editor
- [ ] Docker + deploy (backend on a GPU host, frontend static)

## Known limitations

- Without an API key, translations are placeholders and the source text is EasyOCR's
  guess, which is weak on manga lettering. With a key, Claude reads the bubble images.
- OpenCV inpainting smears textured backgrounds. LaMa is the fix.
- Only use pages you have rights to in public demos.
