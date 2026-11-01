"""Background job runner and storage.

Every upload is a Batch (one or more pages); every page is a Job. Jobs run on a single
worker thread (the OCR models are CPU/GPU heavy, so running several at once just makes
them all slower). State is kept in memory and mirrored to disk:

    data/<job_id>/job.json, original.png, cleaned.png, translated.png
    data/batches/<batch_id>.json

Each batch and job records its owner (user id) and creation time. Anything older than
RETENTION_DAYS is deleted automatically by `cleanup_expired()`.
"""
import io
import json
import logging
import re
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from PIL import Image

from . import config
from .pipeline import TextRegion, rerender, run_pipeline

log = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, "Job"] = {}
_batches: dict[str, "Batch"] = {}
_deleted: set[str] = set()   # job ids deleted while queued/running: skip or discard their work
_lock = threading.RLock()

ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


# ---------------------------------------------------------------------------- jobs

@dataclass
class Job:
    id: str
    filename: str
    source_lang: str
    target_lang: str
    style: str = "standard"
    glossary: str = ""
    owner: int = 0
    created_at: float = 0.0
    status: str = "queued"          # queued | running | done | error
    stage: str = "queued"           # detecting | reading | translating | cleaning | typesetting
    error: str = ""
    regions: list[dict] = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return config.DATA_DIR / self.id

    def save(self):
        if self.id in _deleted:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "job.json").write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1))


def _create_job(data: bytes, filename: str, owner: int, source_lang: str, target_lang: str,
                style: str, glossary: str) -> Job:
    job = Job(id=_new_id(), filename=filename, source_lang=source_lang, target_lang=target_lang,
              style=style, glossary=glossary, owner=owner, created_at=time.time())
    job.dir.mkdir(parents=True, exist_ok=True)
    # Normalize every upload (PNG/JPG/WebP/HEIC/...) to PNG
    Image.open(io.BytesIO(data)).convert("RGB").save(job.dir / "original.png")
    job.save()
    with _lock:
        _jobs[job.id] = job
    _executor.submit(_run, job)
    return job


def get_job(job_id: str) -> Job | None:
    if not ID_RE.match(job_id):
        return None
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            path = config.DATA_DIR / job_id / "job.json"
            if path.exists():
                job = Job(**json.loads(path.read_text()))
                _jobs[job_id] = job
    return job


def _run(job: Job):
    if job.id in _deleted:
        return

    def progress(stage: str):
        job.stage = stage
        job.save()

    try:
        job.status = "running"
        img = _load_rgb(job.dir / "original.png")
        regions, cleaned, result = run_pipeline(img, job.source_lang, job.target_lang, progress,
                                                style=job.style, glossary=job.glossary)
        if job.id in _deleted:
            return
        Image.fromarray(cleaned).save(job.dir / "cleaned.png")
        Image.fromarray(result).save(job.dir / "translated.png")
        job.regions = [r.to_dict() for r in regions]
        job.status, job.stage = "done", "done"
    except Exception as e:
        if job.id in _deleted:
            return
        log.error("Job %s failed:\n%s", job.id, traceback.format_exc())
        job.status, job.error = "error", f"{type(e).__name__}: {e}"
    job.save()


def update_translations(job: Job, edits: dict[int, str]) -> Job:
    """Apply hand-edited translations and redraw the page."""
    regions = [TextRegion.from_dict(d) for d in job.regions]
    for r in regions:
        if r.id in edits:
            r.translated_text = edits[r.id]
    cleaned = _load_rgb(job.dir / "cleaned.png")
    Image.fromarray(rerender(cleaned, regions, job.target_lang)).save(job.dir / "translated.png")
    job.regions = [r.to_dict() for r in regions]
    job.save()
    return job


def _delete_job(job_id: str):
    with _lock:
        _deleted.add(job_id)
        _jobs.pop(job_id, None)
    shutil.rmtree(config.DATA_DIR / job_id, ignore_errors=True)


# ---------------------------------------------------------------------------- batches

@dataclass
class Batch:
    """One upload: a single image, several images, or a document split into pages."""
    id: str
    name: str
    owner: int = 0
    created_at: float = 0.0
    source_lang: str = "auto"
    target_lang: str = "en"
    style: str = "standard"
    page_ids: list[str] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return config.DATA_DIR / "batches" / f"{self.id}.json"

    @property
    def expires_at(self) -> float:
        return self.created_at + config.RETENTION_DAYS * 86400

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self), ensure_ascii=False))

    def pages(self) -> list[Job]:
        return [j for j in (get_job(pid) for pid in self.page_ids) if j is not None]


def create_batch(pages, name: str, owner: int, source_lang: str, target_lang: str,
                 style: str = "standard", glossary: str = "") -> Batch:
    """pages: list of archives.Page. Jobs run one at a time, in page order."""
    batch = Batch(id=_new_id(), name=name, owner=owner, created_at=time.time(),
                  source_lang=source_lang, target_lang=target_lang, style=style)
    for page in pages:
        job = _create_job(page.data, page.name, owner, source_lang, target_lang, style, glossary)
        batch.page_ids.append(job.id)
    batch.save()
    with _lock:
        _batches[batch.id] = batch
    return batch


def get_batch(batch_id: str) -> Batch | None:
    if not ID_RE.match(batch_id):
        return None
    with _lock:
        batch = _batches.get(batch_id)
        if batch is None:
            path = config.DATA_DIR / "batches" / f"{batch_id}.json"
            if path.exists():
                batch = Batch(**json.loads(path.read_text()))
                _batches[batch_id] = batch
    return batch


def list_batches(owner: int) -> list[Batch]:
    """All of a user's batches, newest first."""
    folder = config.DATA_DIR / "batches"
    batches = []
    for path in folder.glob("*.json") if folder.exists() else []:
        batch = get_batch(path.stem)
        if batch and batch.owner == owner:
            batches.append(batch)
    return sorted(batches, key=lambda b: b.created_at, reverse=True)


def delete_batch(batch: Batch):
    for job_id in batch.page_ids:
        _delete_job(job_id)
    with _lock:
        _batches.pop(batch.id, None)
    batch.path.unlink(missing_ok=True)


def batch_zip(batch: Batch) -> bytes:
    """ZIP of every finished translated page, numbered in reading order."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:  # PNGs are already compressed
        for i, job in enumerate(batch.pages(), start=1):
            path = job.dir / "translated.png"
            if job.status == "done" and path.exists():
                stem = Path(job.filename.split(" / ")[-1]).stem
                stem = re.sub(r"[^\w\-]+", "_", stem)[:60] or "page"
                zf.write(path, f"{i:03d}_{stem}.png")
    return buf.getvalue()


# ---------------------------------------------------------------------------- retention

def cleanup_expired(now: float | None = None) -> int:
    """Delete batches (and their pages) older than RETENTION_DAYS, plus orphaned page folders.

    Returns the number of batches deleted.
    """
    now = now or time.time()
    cutoff = now - config.RETENTION_DAYS * 86400
    removed = 0
    folder = config.DATA_DIR / "batches"
    keep_jobs: set[str] = set()
    for path in list(folder.glob("*.json")) if folder.exists() else []:
        batch = get_batch(path.stem)
        if batch is None:
            continue
        # Batches from before accounts existed have no created_at: use the file's age instead,
        # so they aren't deleted the moment you update
        created = batch.created_at or path.stat().st_mtime
        if created < cutoff:
            delete_batch(batch)
            removed += 1
        else:
            keep_jobs.update(batch.page_ids)
    # Page folders that belong to no batch (e.g. from older versions) and are past retention
    for job_dir in config.DATA_DIR.iterdir() if config.DATA_DIR.exists() else []:
        if job_dir.is_dir() and ID_RE.match(job_dir.name) and job_dir.name not in keep_jobs:
            if job_dir.stat().st_mtime < cutoff:
                _delete_job(job_dir.name)
    if removed:
        log.info("Auto-deleted %d batch(es) older than %d days", removed, config.RETENTION_DAYS)
    return removed


def start_cleanup_thread(interval_s: int = 3600):
    """Run cleanup now and then every hour, in the background."""
    def loop():
        while True:
            try:
                cleanup_expired()
                from . import auth
                auth.purge_expired_sessions()
            except Exception:
                log.exception("Cleanup failed")
            time.sleep(interval_s)

    threading.Thread(target=loop, name="cleanup", daemon=True).start()
