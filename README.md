# Manga Translator

Upload a manga page. The app finds the speech bubbles, reads the Japanese, Chinese or Korean
text, translates it with Claude, erases the original lettering and typesets the
translation back into the bubbles. You can then fix any line by hand and re-render.

![screenshot](docs/screenshot.png)

## How it works

```
upload ─▶ detect ─▶ OCR ─▶ translate ─▶ clean ─▶ typeset ─▶ review/edit
```

| Step | File | Approach |
|---|---|---|
| Detect | `pipeline/detect.py` | EasyOCR's CRAFT detector finds text lines; nearby lines are merged into bubbles (union-find); a projection profile splits vertical text into columns; regions are sorted in manga reading order (right-to-left). |
| OCR | `pipeline/ocr.py` | [manga-ocr](https://github.com/kha-white/manga-ocr) for Japanese if installed; otherwise EasyOCR. Vertical columns are cut into characters and laid out horizontally so EasyOCR can read them. |
| Translate | `pipeline/translate.py` | The whole page is sent to Claude in one request, as JSON in reading order, so the model has context and can correct OCR mistakes. Returns JSON. |
| Clean | `pipeline/clean.py` | Text in white bubbles: strokes repainted with the bubble color. Text over artwork: stroke mask + OpenCV inpainting. |
| Typeset | `pipeline/typeset.py` | Flood-fills the bubble to find its real size, binary-searches the largest font that fits after word wrapping, and draws the text centered (with an outline over artwork). |

The backend is FastAPI. Each page is a background job, and the frontend polls for
progress. Edits only re-run the typesetting step, using the saved cleaned image.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your ANTHROPIC_API_KEY
uvicorn backend.app.main:app --reload
```

Open http://localhost:8000. The first run downloads the EasyOCR models (~100 MB).
Without an API key the app runs in demo mode: detection, OCR and typesetting are real,
and translations are placeholders.

Better Japanese OCR: `pip install manga-ocr` (it's picked up automatically).

Better lettering: put a comic font at `fonts/comic.ttf` or set `FONT_PATH`.
[Comic Neue](https://comicneue.com/) is free (OFL).

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
| `POST` | `/api/jobs` | multipart: `file`, `source_lang`, `target_lang` → job |
| `GET` | `/api/jobs/{id}` | Status, current stage, regions (source and translated text, boxes) |
| `GET` | `/api/jobs/{id}/image/{original,translated,cleaned}` | PNG |
| `PUT` | `/api/jobs/{id}/translations` | `{"translations":[{"id":0,"text":"..."}]}` → re-render |

## Roadmap

- [ ] Batch: CBZ/ZIP/PDF upload, split into pages, download as ZIP
- [ ] LaMa inpainting for text over artwork
- [ ] Manga-specific bubble detector (comic-text-detector / fine-tuned YOLO)
- [ ] Context across pages (previous page's dialogue in the prompt)
- [ ] Drag/resize text boxes in the editor
- [ ] Docker + deploy (backend on a GPU host, frontend static)

## Known limitations

- EasyOCR is weak on stylized or vertical Japanese. Install manga-ocr for real pages.
- OpenCV inpainting smears textured backgrounds. LaMa is the fix.
- Only use pages you have rights to in public demos.
