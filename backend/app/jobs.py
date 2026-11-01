"""Background job runner and storage.

Every upload is a Batch (one or more pages); every page is a Job. Pages run on a pool of
WORKERS threads, so detection/OCR/erasing of several pages overlap. Translation is
chained: page N translates after page N-1 and gets its dialogue as context (see _Chain).
State is kept in memory and mirrored to disk:

    data/<job_id>/job.json, original.png, cleaned.png, translated.png
    data/batches/<batch_id>.json

Each batch and job records its owner (user id) and creation time. Anything older than
RETENTION_DAYS is deleted automatically by `cleanup_expired()`.
"""
import io
import json
import logging
import os
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

from . import config, pricing
from .pipeline import PageContext, TextRegion, rerender, run_pipeline
from .pipeline.enhance import enhance as enhance_scan

log = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=config.WORKERS, thread_name_prefix="page")
_jobs: dict[str, "Job"] = {}
_batches: dict[str, "Batch"] = {}
_deleted: set[str] = set()   # job ids deleted while queued/running: skip or discard their work
_stopped: set[str] = set()   # queued job ids the user stopped: skip them when their turn comes
STOPPED_MESSAGE = "Stopped before it was translated. Use Retry to translate it."
_lock = threading.RLock()

ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _write_json(path: Path, data: dict):
    """Save JSON safely: always UTF-8 (Windows defaults to a legacy encoding that can't
    store Japanese), and atomically (write a temp file, then swap it in), so a crash or
    error mid-write can never leave a half-written or empty file behind."""
    tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    """Load JSON, or None (with a warning) if the file is missing or damaged."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (ValueError, UnicodeDecodeError) as e:
        log.warning("Skipping damaged file %s (%s)", path, e)
        return None


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
    font_style: str = "manga"
    enhance: bool = False           # clean up a low-quality scan first (pipeline/enhance.py)
    quality: str = "economy"        # economy (Haiku) | best (Sonnet), see pricing.py
    usage: dict = field(default_factory=dict)  # Claude tokens + estimated cost for this page
    owner: int = 0
    created_at: float = 0.0
    batch_id: str = ""
    page_index: int = 0
    status: str = "queued"          # queued | running | done | error
    stage: str = "queued"           # enhancing | detecting | reading | waiting | translating | cleaning | typesetting
    error: str = ""
    regions: list[dict] = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return config.DATA_DIR / self.id

    def save(self):
        if self.id in _deleted:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.dir / "job.json", asdict(self))


class _Chain:
    """Links the pages of one batch so translation happens in page order.

    Page i's translation waits until page i-1 has published its dialogue, then gets the
    dialogue of the last CONTEXT_PAGES pages as context. Pages are submitted to the pool
    in order, so the page being waited on has always started already: no deadlock.
    """
    CONTEXT_PAGES = 2
    TIMEOUT_S = 15 * 60  # never wait forever on a stuck page

    def __init__(self, n: int):
        self.done = [threading.Event() for _ in range(n)]
        self.lines: list[list[dict]] = [[] for _ in range(n)]

    def context_for(self, i: int) -> PageContext:
        def wait_for_previous() -> list[dict]:
            if i > 0:
                self.done[i - 1].wait(timeout=self.TIMEOUT_S)
            start = max(0, i - self.CONTEXT_PAGES)
            return [line for page in self.lines[start:i] for line in page]

        def publish(lines: list[dict]):
            if not self.done[i].is_set():  # first call wins
                self.lines[i] = lines
                self.done[i].set()

        return PageContext(wait_for_previous, publish)


def _create_job(data: bytes, filename: str, owner: int, source_lang: str, target_lang: str,
                style: str, glossary: str, batch_id: str = "", page_index: int = 0,
                font_style: str = "manga", enhance: bool = False, quality: str = "economy") -> Job:
    job = Job(id=_new_id(), filename=filename, source_lang=source_lang, target_lang=target_lang,
              style=style, glossary=glossary, font_style=font_style, enhance=enhance,
              quality=quality, owner=owner,
              created_at=time.time(), batch_id=batch_id, page_index=page_index)
    job.dir.mkdir(parents=True, exist_ok=True)
    # Normalize every upload (PNG/JPG/WebP/HEIC/...) to PNG
    Image.open(io.BytesIO(data)).convert("RGB").save(job.dir / "original.png")
    job.save()
    with _lock:
        _jobs[job.id] = job
    return job


def get_job(job_id: str) -> Job | None:
    if not ID_RE.match(job_id):
        return None
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            data = _read_json(config.DATA_DIR / job_id / "job.json")
            if data is not None:
                job = Job(**{k: v for k, v in data.items() if k in Job.__dataclass_fields__})
                if job.status in ("queued", "running"):
                    # Every live page is kept in memory, so one only found on disk was
                    # cut off by a server restart and will never finish on its own
                    job.status, job.stage = "error", "error"
                    job.error = "Interrupted by a server restart. Please translate this page again."
                    job.save()
                _jobs[job_id] = job
    return job


def friendly_error(e: Exception) -> str:
    """Turn common failures into a plain-English reason and what to do about it."""
    text = str(e)
    try:
        import anthropic
        if isinstance(e, anthropic.AuthenticationError):
            return "Your Claude API key was rejected. Check ANTHROPIC_API_KEY in .env, then restart the app."
        if isinstance(e, anthropic.PermissionDeniedError):
            return "Your Claude API key isn't allowed to use this model. Check the key's workspace in the Claude Console."
        if isinstance(e, anthropic.RateLimitError):
            return "Too many requests to Claude right now (rate limit). Wait a minute, then retry."
        if isinstance(e, anthropic.APIConnectionError):
            return "Couldn't reach Claude. Check your internet connection, then retry."
        if isinstance(e, anthropic.APIStatusError):
            if "credit balance" in text.lower():
                return "Your Claude credit balance is too low. Add credits in the Claude Console, then retry."
            if e.status_code == 529 or "overloaded" in text.lower():
                return "Claude is busy right now (overloaded). Wait a minute, then retry."
            if e.status_code >= 500:
                return "Claude had a temporary server error. Retry this page."
            return f"Claude refused the request ({e.status_code}): {getattr(e, 'message', text)[:200]}"
    except ImportError:
        pass
    if isinstance(e, ValueError) and "JSON" in text:
        return "Claude's reply couldn't be read. Retry this page (Best quality often helps)."
    return f"{type(e).__name__}: {text[:250]}"


def _run(job: Job, context: PageContext | None = None, earlier_usage: dict | None = None):
    def progress(stage: str):
        job.stage = stage
        job.save()

    try:
        if job.id in _deleted or job.id in _stopped:
            return  # (the finally below still unblocks the next page)
        job.status = "running"
        img = _load_rgb(job.dir / "original.png")
        if job.enhance:
            progress("enhancing")
            img = enhance_scan(img)
        usage = pricing.empty_usage()
        if earlier_usage:  # a retry: keep counting what earlier attempts cost
            for key in usage:
                if key != "model":
                    usage[key] = earlier_usage.get(key, 0) or 0
        try:
            regions, cleaned, result = run_pipeline(img, job.source_lang, job.target_lang, progress,
                                                    style=job.style, glossary=job.glossary,
                                                    context=context, font_style=job.font_style,
                                                    quality=job.quality, usage=usage)
        finally:
            job.usage = usage  # record what Claude cost even if a later step failed
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
        job.status, job.error = "error", friendly_error(e)
    finally:
        if context:
            context.publish([])  # no-op if already published; unblocks the next page otherwise
    job.save()


def retry_job(job: Job, quality: str | None = None) -> Job:
    """Translate a page again (after a failure, or to try another quality).

    Uses the previous page's finished dialogue as context, like the first run did.
    Usage/cost from the earlier attempt is kept and added to, since it was billed.
    """
    if job.status in ("queued", "running"):
        raise ValueError("This page is already being translated")
    previous: list[dict] = []
    batch = get_batch(job.batch_id) if job.batch_id else None
    if batch and job.page_index > 0:
        prev = get_job(batch.page_ids[job.page_index - 1])
        if prev and prev.status == "done":
            previous = [{"source": r.get("source_text", ""), "text": r.get("translated_text", "")}
                        for r in prev.regions]
    if quality:
        job.quality = quality
    _stopped.discard(job.id)
    earlier_usage = dict(job.usage or {})
    job.status, job.stage, job.error = "queued", "queued", ""
    job.regions = []
    job.save()
    with _lock:
        _deleted.discard(job.id)
        _jobs[job.id] = job
    context = PageContext(wait_for_previous=lambda: previous, publish=lambda lines: None)
    _executor.submit(_run, job, context, earlier_usage)
    return job


def update_translations(job: Job, edits: dict[int, str], font_style: str | None = None) -> Job:
    """Apply hand-edited translations (and optionally a new font style) and redraw the page."""
    if font_style:
        job.font_style = font_style
    regions = [TextRegion.from_dict(d) for d in job.regions]
    for r in regions:
        if r.id in edits:
            r.translated_text = edits[r.id]
    cleaned = _load_rgb(job.dir / "cleaned.png")
    Image.fromarray(rerender(cleaned, regions, job.target_lang, job.font_style)).save(job.dir / "translated.png")
    job.regions = [r.to_dict() for r in regions]
    job.save()
    return job


def _recover_job(job_id: str, batch: "Batch", index: int) -> Job | None:
    """Rebuild a page whose job.json is missing or damaged, from the images on disk.

    Older versions could leave job.json empty on Windows (non-UTF-8 default encoding).
    The images were already saved, so a finished page is still viewable and downloadable;
    only its editable bubble list is lost.
    """
    folder = config.DATA_DIR / job_id
    if not ID_RE.match(job_id) or not (folder / "original.png").exists():
        return None
    done = (folder / "translated.png").exists()
    job = Job(id=job_id, filename=f"page {index + 1}", source_lang=batch.source_lang,
              target_lang=batch.target_lang, style=batch.style, font_style=batch.font_style,
              quality=batch.quality, owner=batch.owner, created_at=batch.created_at,
              batch_id=batch.id, page_index=index,
              status="done" if done else "error", stage="done" if done else "error",
              error="" if done else "This page's saved data was damaged. Please translate it again.")
    job.save()
    with _lock:
        _jobs[job_id] = job
    log.info("Recovered page %s from its images", job_id)
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
    font_style: str = "manga"
    enhance: bool = False
    quality: str = "economy"
    page_ids: list[str] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return config.DATA_DIR / "batches" / f"{self.id}.json"

    @property
    def expires_at(self) -> float:
        return self.created_at + config.RETENTION_DAYS * 86400

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(self.path, asdict(self))

    def pages(self) -> list[Job]:
        pages = []
        for i, pid in enumerate(self.page_ids):
            job = get_job(pid) or _recover_job(pid, self, i)
            if job is not None:
                pages.append(job)
        return pages


def create_batch(pages, name: str, owner: int, source_lang: str, target_lang: str,
                 style: str = "standard", glossary: str = "", font_style: str = "manga",
                 enhance: bool = False, quality: str = "economy") -> Batch:
    """pages: list of archives.Page, in reading order."""
    batch = Batch(id=_new_id(), name=name, owner=owner, created_at=time.time(),
                  source_lang=source_lang, target_lang=target_lang, style=style,
                  font_style=font_style, enhance=enhance, quality=quality)
    page_jobs = [_create_job(page.data, page.name, owner, source_lang, target_lang, style, glossary,
                             batch_id=batch.id, page_index=i, font_style=font_style, enhance=enhance,
                             quality=quality)
                 for i, page in enumerate(pages)]
    batch.page_ids = [j.id for j in page_jobs]
    batch.save()
    with _lock:
        _batches[batch.id] = batch
    # Submit in page order (required by _Chain), only once every page is saved
    chain = _Chain(len(page_jobs))
    for i, job in enumerate(page_jobs):
        _executor.submit(_run, job, chain.context_for(i))
    return batch


def get_batch(batch_id: str) -> Batch | None:
    if not ID_RE.match(batch_id):
        return None
    with _lock:
        batch = _batches.get(batch_id)
        if batch is None:
            data = _read_json(config.DATA_DIR / "batches" / f"{batch_id}.json")
            if data is not None:
                batch = Batch(**{k: v for k, v in data.items() if k in Batch.__dataclass_fields__})
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


def stop_batch(batch: Batch) -> int:
    """Stop a batch: pages not started yet are skipped (pages already running finish).

    Stopped pages count as failed with a clear message, so "Retry failed" resumes them.
    Returns how many pages were stopped.
    """
    stopped = 0
    for job in batch.pages():
        if job.status == "queued":
            with _lock:
                _stopped.add(job.id)
            job.status, job.stage, job.error = "error", "error", STOPPED_MESSAGE
            job.save()
            stopped += 1
    return stopped


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
