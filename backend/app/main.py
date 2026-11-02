"""FastAPI app: REST API + serves the frontend.

Run:  uvicorn backend.app.main:app --reload

Every translation endpoint needs a logged-in user (session cookie), and users only ever
see their own uploads: anything owned by someone else answers 404, as if it didn't exist.
"""
import logging
import re
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import archives, auth, config, jobs, pricing
from .auth import User, current_user
from .pipeline.translate import STYLES

logging.basicConfig(level=logging.INFO)
config.DATA_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    jobs.start_cleanup_thread()  # auto-delete uploads older than RETENTION_DAYS
    yield
    jobs.shutdown()  # Ctrl+C: stop translating too, not just the web server


app = FastAPI(title="Manga Translator", lifespan=lifespan)

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp",
                 "image/avif", "image/heic", "image/heif"}


# ---------------------------------------------------------------- auth

class Credentials(BaseModel):
    username: str
    password: str


def _set_session(response: Response, user: User):
    response.set_cookie(auth.COOKIE_NAME, auth.create_session(user), httponly=True,
                        samesite="lax", max_age=auth.SESSION_DAYS * 86400)


@app.post("/api/auth/signup")
def signup(body: Credentials, response: Response):
    try:
        user = auth.create_user(body.username, body.password)
    except ValueError as e:
        raise HTTPException(400, str(e))
    _set_session(response, user)
    return {"username": user.username}


@app.post("/api/auth/login")
def login(body: Credentials, response: Response):
    user = auth.authenticate(body.username, body.password)
    if user is None:
        raise HTTPException(401, "Wrong username or password")
    _set_session(response, user)
    return {"username": user.username}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(auth.COOKIE_NAME)
    if token:
        auth.delete_session(token)
    response.delete_cookie(auth.COOKIE_NAME)
    return {"ok": True}


@app.get("/api/auth/me")
def me(user: User = Depends(current_user)):
    return {"username": user.username, "retention_days": config.RETENTION_DAYS,
            "api_key_created_at": auth.api_key_created_at(user)}


def _cookie_user(request: Request) -> User:
    """Logged in via the website (not an API key): only the website can manage keys."""
    user = auth.user_for_token(request.cookies.get(auth.COOKIE_NAME))
    if user is None:
        raise HTTPException(401, "Please log in")
    return user


@app.post("/api/auth/api-key")
def new_api_key(user: User = Depends(_cookie_user)):
    """Create the browser-extension key (replaces the old one). Shown only this once."""
    return {"key": auth.create_api_key(user)}


@app.delete("/api/auth/api-key")
def delete_api_key(user: User = Depends(_cookie_user)):
    auth.revoke_api_key(user)
    return {"ok": True}


# ---------------------------------------------------------------- helpers

def _job_or_404(job_id: str, user: User) -> jobs.Job:
    job = jobs.get_job(job_id)
    if job is None or job.owner != user.id:
        raise HTTPException(404, "Page not found")
    return job


def _batch_or_404(batch_id: str, user: User) -> jobs.Batch:
    batch = jobs.get_batch(batch_id)
    if batch is None or batch.owner != user.id:
        raise HTTPException(404, "Batch not found")
    return batch


def _job_json(job: jobs.Job) -> dict:
    data = asdict(job)
    data["original_url"] = f"/api/jobs/{job.id}/image/original"
    if job.status == "done":
        data["translated_url"] = f"/api/jobs/{job.id}/image/translated"
    return data


def _was_stopped(job: jobs.Job) -> bool:
    """Failed only because it was stopped (by the user, or by shutting the app down)."""
    return job.status == "error" and job.error.startswith(("Stopped", "Interrupted"))


def _batch_json(batch: jobs.Batch) -> dict:
    pages = batch.pages()
    done = sum(p.status == "done" for p in pages)
    failed = sum(p.status == "error" for p in pages)
    stopped = sum(_was_stopped(p) for p in pages)
    status = "done" if done + failed == len(pages) else "running"
    first = pages[0] if pages else None
    thumb = None
    if first:
        kind = "translated" if first.status == "done" else "original"
        thumb = f"/api/jobs/{first.id}/image/{kind}"
    return {
        "id": batch.id,
        "name": batch.name,
        "status": status,
        "total": len(pages),
        "done": done,
        "failed": failed,                 # includes stopped pages
        "stopped": stopped,               # failed only because translation was stopped
        "created_at": batch.created_at,
        "expires_at": batch.expires_at,
        "source_lang": batch.source_lang,
        "target_lang": batch.target_lang,
        "style": batch.style,
        "quality": batch.quality,
        # Estimated Claude cost so far, from the token counts each page's reply reported
        "cost_usd": sum((p.usage or {}).get("cost_usd", 0.0) for p in pages),
        "thumb_url": thumb,
        "download_url": f"/api/batches/{batch.id}/download",
        "pages": [{"id": p.id, "filename": p.filename, "status": p.status, "stage": p.stage,
                   "stopped": _was_stopped(p), "cost_usd": (p.usage or {}).get("cost_usd", 0.0)}
                  for p in pages],
    }


def _check_options(source_lang: str, target_lang: str, style: str, font_style: str = "manga",
                   quality: str = pricing.DEFAULT_QUALITY):
    if source_lang not in config.SOURCE_LANGS or target_lang not in config.TARGET_LANGS:
        raise HTTPException(400, "Unsupported language")
    if style not in STYLES:
        raise HTTPException(400, "Unsupported style")
    if font_style not in config.FONT_STYLES:
        raise HTTPException(400, "Unsupported font style")
    if quality not in pricing.QUALITIES:
        raise HTTPException(400, "Unsupported quality")


# ---------------------------------------------------------------- config

@app.get("/api/config")
def get_config():
    return {
        "source_langs": config.SOURCE_LANGS,
        "target_langs": config.TARGET_LANGS,
        "translator": config.TRANSLATOR,
        "styles": list(STYLES),
        "retention_days": config.RETENTION_DAYS,
        "font_styles": config.FONT_STYLES,
        # Estimated price of a typical page (~3,000 tokens in, ~600 out) for each quality
        "qualities": {q: {"label": v["label"],
                          "est_page_usd": pricing.cost_usd(v["model"], {"input_tokens": 3000, "output_tokens": 600})}
                      for q, v in pricing.QUALITIES.items()},
        "default_quality": pricing.DEFAULT_QUALITY,
    }


# ---------------------------------------------------------------- single pages

@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...),
                     source_lang: str = Form("auto"),
                     target_lang: str = Form("en"),
                     style: str = Form("standard"),
                     glossary: str = Form(""),
                     font_style: str = Form("manga"),
                     enhance: bool = Form(False),
                     quality: str = Form(pricing.DEFAULT_QUALITY),
                     user: User = Depends(current_user)):
    """Translate one image. (Recorded as a one-page batch, so it shows on the dashboard.)"""
    # Browsers often send HEIC as application/octet-stream, so also accept by extension
    if (file.content_type not in ALLOWED_TYPES
            and archives.ext_of(file.filename or "") not in archives.IMAGE_EXTS):
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")
    _check_options(source_lang, target_lang, style, font_style, quality)
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {config.MAX_UPLOAD_MB} MB")
    name = file.filename or "page.png"
    batch = jobs.create_batch([archives.Page(name, data)], name, user.id,
                              source_lang, target_lang, style, glossary[:2000], font_style, enhance,
                              quality)
    data = _job_json(jobs.get_job(batch.page_ids[0]))
    data["batch_id"] = batch.id
    return data


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, user: User = Depends(current_user)):
    return _job_json(_job_or_404(job_id, user))


@app.get("/api/jobs/{job_id}/image/{kind}")
def get_image(job_id: str, kind: str, user: User = Depends(current_user)):
    if kind not in ("original", "translated", "cleaned"):
        raise HTTPException(404)
    path = _job_or_404(job_id, user).dir / f"{kind}.png"
    if not path.exists():
        raise HTTPException(404, "Image not ready")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})


class Edit(BaseModel):
    id: int
    text: str


class EditRequest(BaseModel):
    translations: list[Edit] = []
    font_style: str | None = None


@app.put("/api/jobs/{job_id}/translations")
def edit_translations(job_id: str, body: EditRequest, user: User = Depends(current_user)):
    job = _job_or_404(job_id, user)
    if job.status != "done":
        raise HTTPException(409, "Page is not finished yet")
    if body.font_style and body.font_style not in config.FONT_STYLES:
        raise HTTPException(400, "Unsupported font style")
    job = jobs.update_translations(job, {e.id: e.text for e in body.translations}, body.font_style)
    return _job_json(job)


class RetryRequest(BaseModel):
    quality: str | None = None


@app.post("/api/jobs/{job_id}/retry")
def retry_page(job_id: str, body: RetryRequest | None = None, user: User = Depends(current_user)):
    """Translate one page again: after a failure, or to try it at another quality."""
    job = _job_or_404(job_id, user)
    quality = body.quality if body else None
    if quality and quality not in pricing.QUALITIES:
        raise HTTPException(400, "Unsupported quality")
    try:
        job = jobs.retry_job(job, quality)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return _job_json(job)


# ---------------------------------------------------------------- batches

@app.post("/api/batches")
async def create_batch(files: list[UploadFile] = File(...),
                       source_lang: str = Form("auto"),
                       target_lang: str = Form("en"),
                       style: str = Form("standard"),
                       glossary: str = Form(""),
                       font_style: str = Form("manga"),
                       enhance: bool = Form(False),
                       quality: str = Form(pricing.DEFAULT_QUALITY),
                       name: str = Form(""),
                       user: User = Depends(current_user)):
    """Upload several images and/or ZIP, CBZ, EPUB, RAR or PDF files; every page becomes a job."""
    _check_options(source_lang, target_lang, style, font_style, quality)
    uploads, total = [], 0
    for f in files:
        filename = f.filename or "upload"
        if archives.ext_of(filename) not in archives.SUPPORTED_EXTS:
            raise HTTPException(400, f"Unsupported file: {filename}")
        data = await f.read()
        total += len(data)
        if total > config.MAX_BATCH_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"Upload larger than {config.MAX_BATCH_UPLOAD_MB} MB")
        uploads.append((filename, data))
    try:
        pages = archives.expand_uploads(uploads)
    except archives.ArchiveError as e:
        raise HTTPException(400, str(e))
    # Optional display name (the extension sends the page title); otherwise use the file names
    name = name.strip()[:120] or (uploads[0][0] if len(uploads) == 1 else f"{len(uploads)} files")
    batch = jobs.create_batch(pages, name, user.id, source_lang, target_lang, style, glossary[:2000],
                              font_style, enhance, quality)
    return _batch_json(batch)


@app.get("/api/batches")
def list_batches(user: User = Depends(current_user)):
    """The dashboard: your recent uploads, newest first."""
    return {"retention_days": config.RETENTION_DAYS,
            "batches": [_batch_json(b) for b in jobs.list_batches(user.id)]}


@app.get("/api/batches/{batch_id}")
def get_batch(batch_id: str, user: User = Depends(current_user)):
    return _batch_json(_batch_or_404(batch_id, user))


@app.post("/api/batches/{batch_id}/stop")
def stop_batch(batch_id: str, user: User = Depends(current_user)):
    """Stop translating: pages not started yet are skipped (Retry failed resumes them)."""
    return {"stopped": jobs.stop_batch(_batch_or_404(batch_id, user))}


@app.delete("/api/batches/{batch_id}")
def delete_batch(batch_id: str, user: User = Depends(current_user)):
    jobs.delete_batch(_batch_or_404(batch_id, user))
    return {"ok": True}


@app.get("/api/batches/{batch_id}/download")
def download_batch(batch_id: str, user: User = Depends(current_user)):
    batch = _batch_or_404(batch_id, user)
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", archives.PurePosixPath(batch.name).stem)[:60] or "batch"
    return Response(jobs.batch_zip(batch), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{stem}_translated.zip"'})


class FreshStaticFiles(StaticFiles):
    """Static files the browser must re-check every time (no-cache = revalidate first).

    Without this, browsers may keep showing an old index.html/app.js after an update.
    The ETag check is cheap: unchanged files come back as a tiny 304 response.
    """
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


# Lettering fonts, so the font picker can preview them (all OFL licensed)
app.mount("/fonts", StaticFiles(directory=config.ROOT / "fonts"), name="fonts")
# Frontend (mounted last so /api routes take priority)
app.mount("/", FreshStaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
