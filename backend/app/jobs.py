"""Background job runner.

Each uploaded page becomes a job. Jobs run on a single worker thread (the OCR models
are CPU/GPU heavy, so running several at once just makes them all slower).
Job state is kept in memory and mirrored to data/<job_id>/job.json.
"""
import io
import json
import logging
import threading
import traceback
import uuid
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
_lock = threading.Lock()


@dataclass
class Job:
    id: str
    filename: str
    source_lang: str
    target_lang: str
    style: str = "standard"
    glossary: str = ""
    status: str = "queued"          # queued | running | done | error
    stage: str = "queued"           # detecting | reading | translating | cleaning | typesetting
    error: str = ""
    regions: list[dict] = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return config.DATA_DIR / self.id

    def save(self):
        (self.dir / "job.json").write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1))


def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def create_job(data: bytes, filename: str, source_lang: str, target_lang: str,
               style: str = "standard", glossary: str = "") -> Job:
    job = Job(id=uuid.uuid4().hex[:12], filename=filename,
              source_lang=source_lang, target_lang=target_lang,
              style=style, glossary=glossary)
    job.dir.mkdir(parents=True, exist_ok=True)
    # Normalize every upload (PNG/JPG/WebP/...) to PNG
    Image.open(io.BytesIO(data)).convert("RGB").save(job.dir / "original.png")
    job.save()
    with _lock:
        _jobs[job.id] = job
    _executor.submit(_run, job)
    return job


def get_job(job_id: str) -> Job | None:
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        path = config.DATA_DIR / job_id / "job.json"
        if path.exists() and job_id.isalnum():
            job = Job(**json.loads(path.read_text()))
            with _lock:
                _jobs[job_id] = job
    return job


def _run(job: Job):
    def progress(stage: str):
        job.stage = stage
        job.save()

    try:
        job.status = "running"
        img = _load_rgb(job.dir / "original.png")
        regions, cleaned, result = run_pipeline(img, job.source_lang, job.target_lang, progress,
                                                style=job.style, glossary=job.glossary)
        Image.fromarray(cleaned).save(job.dir / "cleaned.png")
        Image.fromarray(result).save(job.dir / "translated.png")
        job.regions = [r.to_dict() for r in regions]
        job.status, job.stage = "done", "done"
    except Exception as e:
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


# ---------------------------------------------------------------------------
# Batches: many pages (from several images, a ZIP/CBZ or a PDF) translated together.
# Each page is an ordinary Job; the batch just keeps their ids in reading order.
# ---------------------------------------------------------------------------

_batches: dict[str, "Batch"] = {}


@dataclass
class Batch:
    id: str
    name: str
    page_ids: list[str] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return config.DATA_DIR / "batches" / f"{self.id}.json"

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self), ensure_ascii=False))

    def pages(self) -> list[Job]:
        return [j for j in (get_job(pid) for pid in self.page_ids) if j is not None]


def create_batch(pages, name: str, source_lang: str, target_lang: str,
                 style: str = "standard", glossary: str = "") -> Batch:
    """pages: list of archives.Page. Jobs run one at a time, in page order."""
    batch = Batch(id=uuid.uuid4().hex[:12], name=name)
    for page in pages:
        job = create_job(page.data, page.name, source_lang, target_lang, style, glossary)
        batch.page_ids.append(job.id)
    batch.save()
    with _lock:
        _batches[batch.id] = batch
    return batch


def get_batch(batch_id: str) -> Batch | None:
    with _lock:
        batch = _batches.get(batch_id)
    if batch is None and batch_id.isalnum():
        path = config.DATA_DIR / "batches" / f"{batch_id}.json"
        if path.exists():
            batch = Batch(**json.loads(path.read_text()))
            with _lock:
                _batches[batch_id] = batch
    return batch


def batch_zip(batch: Batch) -> bytes:
    """ZIP of every finished translated page, numbered in reading order."""
    import re
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:  # PNGs are already compressed
        for i, job in enumerate(batch.pages(), start=1):
            path = job.dir / "translated.png"
            if job.status == "done" and path.exists():
                stem = Path(job.filename.split(" / ")[-1]).stem
                stem = re.sub(r"[^\w\-]+", "_", stem)[:60] or "page"
                zf.write(path, f"{i:03d}_{stem}.png")
    return buf.getvalue()
