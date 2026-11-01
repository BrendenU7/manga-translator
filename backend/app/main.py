"""FastAPI app: REST API + serves the frontend.

Run:  uvicorn backend.app.main:app --reload
"""
import logging
from dataclasses import asdict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, jobs
from .pipeline.translate import STYLES

logging.basicConfig(level=logging.INFO)
config.DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Manga Translator")

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp"}


def _job_or_404(job_id: str) -> jobs.Job:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


def _job_json(job: jobs.Job) -> dict:
    data = asdict(job)
    data["original_url"] = f"/api/jobs/{job.id}/image/original"
    if job.status == "done":
        data["translated_url"] = f"/api/jobs/{job.id}/image/translated"
    return data


@app.get("/api/config")
def get_config():
    return {
        "source_langs": config.SOURCE_LANGS,
        "target_langs": config.TARGET_LANGS,
        "translator": config.TRANSLATOR,
        "styles": list(STYLES),
    }


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...),
                     source_lang: str = Form("ja"),
                     target_lang: str = Form("en"),
                     style: str = Form("standard"),
                     glossary: str = Form("")):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")
    if source_lang not in config.SOURCE_LANGS or target_lang not in config.TARGET_LANGS:
        raise HTTPException(400, "Unsupported language")
    if style not in STYLES:
        raise HTTPException(400, "Unsupported style")
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {config.MAX_UPLOAD_MB} MB")
    job = jobs.create_job(data, file.filename or "page.png", source_lang, target_lang,
                          style=style, glossary=glossary[:2000])
    return _job_json(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _job_json(_job_or_404(job_id))


@app.get("/api/jobs/{job_id}/image/{kind}")
def get_image(job_id: str, kind: str):
    if kind not in ("original", "translated", "cleaned"):
        raise HTTPException(404)
    path = _job_or_404(job_id).dir / f"{kind}.png"
    if not path.exists():
        raise HTTPException(404, "Image not ready")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})


class Edit(BaseModel):
    id: int
    text: str


class EditRequest(BaseModel):
    translations: list[Edit]


@app.put("/api/jobs/{job_id}/translations")
def edit_translations(job_id: str, body: EditRequest):
    job = _job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(409, "Job is not finished yet")
    job = jobs.update_translations(job, {e.id: e.text for e in body.translations})
    return _job_json(job)


# Frontend (mounted last so /api routes take priority)
app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
