"""Batch upload: archive splitting and the /api/batches endpoints (fast, no OCR models)."""
import io
import time
import zipfile

import numpy as np
import pytest
from PIL import Image

from backend.app import archives, config, jobs


def png_bytes(w=40, h=60, color=(255, 255, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def make_zip(names: list[str], size=(40, 60)) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n in names:
            zf.writestr(n, png_bytes(*size) if not n.endswith(".txt") else b"hi")
    return buf.getvalue()


def test_natural_sort():
    names = ["p10.png", "p2.png", "p1.png"]
    assert sorted(names, key=archives.natural_key) == ["p1.png", "p2.png", "p10.png"]


def test_zip_pages_in_natural_order_and_junk_skipped():
    data = make_zip(["ch1/p10.png", "ch1/p2.png", "ch1/p1.png", "notes.txt",
                     "__MACOSX/ch1/._p1.png", "ch1/.hidden.png"])
    pages = archives.expand_uploads([("chapter.cbz", data)])
    assert [p.name for p in pages] == ["chapter.cbz / p1.png", "chapter.cbz / p2.png", "chapter.cbz / p10.png"]


def test_epub_drops_small_decorations():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("OEBPS/images/page001.png", png_bytes(800, 1200))
        zf.writestr("OEBPS/images/ornament.png", png_bytes(40, 40))
    pages = archives.expand_uploads([("vol1.epub", buf.getvalue())])
    assert [p.name for p in pages] == ["vol1.epub / page001.png"]


def test_pdf_pages_are_rendered():
    import pymupdf
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page(width=300, height=400)
    pages = archives.expand_uploads([("chapter.pdf", doc.tobytes())])
    assert len(pages) == 3
    assert Image.open(io.BytesIO(pages[0].data)).size[1] > 400  # rendered above 72 dpi


def test_loose_images_sorted_by_name():
    pages = archives.expand_uploads([("10.png", png_bytes()), ("2.png", png_bytes())])
    assert [p.name for p in pages] == ["2.png", "10.png"]


def test_bad_uploads_raise_clear_errors():
    with pytest.raises(archives.ArchiveError, match="not a valid"):
        archives.expand_uploads([("broken.cbz", b"not a zip")])
    with pytest.raises(archives.ArchiveError, match="No images"):
        archives.expand_uploads([("empty.zip", make_zip(["readme.txt"]))])
    with pytest.raises(archives.ArchiveError, match="unsupported"):
        archives.expand_uploads([("story.docx", b"x")])


# --------------------------------------------------------------------- API

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.app import main

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    def fake_pipeline(img, source_lang, target_lang, progress, **kwargs):
        progress("translating")
        return [], img, np.zeros_like(img)

    monkeypatch.setattr(jobs, "run_pipeline", fake_pipeline)
    client = TestClient(main.app)
    assert client.post("/api/auth/signup", json={"username": "reader", "password": "password123"}).status_code == 200
    return client


def wait_for(client, url, done=lambda d: d["status"] == "done", timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        data = client.get(url).json()
        if done(data):
            return data
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {url}")


def test_batch_upload_translates_every_page_and_zips_them(client):
    files = [("files", ("chapter.cbz", make_zip(["p2.png", "p1.png"]), "application/zip")),
             ("files", ("extra.png", png_bytes(), "image/png"))]
    res = client.post("/api/batches", files=files, data={"style": "faithful"})
    assert res.status_code == 200, res.text
    batch = res.json()
    assert batch["total"] == 3

    batch = wait_for(client, f"/api/batches/{batch['id']}")
    assert batch["done"] == 3
    assert [p["filename"] for p in batch["pages"]] == ["extra.png", "chapter.cbz / p1.png", "chapter.cbz / p2.png"]

    res = client.get(batch["download_url"])
    assert res.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(res.content)).namelist()
    assert names == ["001_extra.png", "002_p1.png", "003_p2.png"]


def test_batch_rejects_bad_files(client):
    res = client.post("/api/batches", files=[("files", ("story.docx", b"x", "application/octet-stream"))])
    assert res.status_code == 400
    res = client.post("/api/batches", files=[("files", ("x.cbz", b"nope", "application/zip"))])
    assert res.status_code == 400 and "not a valid" in res.json()["detail"]


def test_font_style_is_saved_and_can_be_changed(client):
    files = [("files", ("p1.png", png_bytes(), "image/png"))]
    batch = client.post("/api/batches", files=files, data={"font_style": "action"}).json()
    batch = wait_for(client, f"/api/batches/{batch['id']}")
    page = batch["pages"][0]["id"]
    assert client.get(f"/api/jobs/{page}").json()["font_style"] == "action"

    res = client.put(f"/api/jobs/{page}/translations", json={"font_style": "handwritten"})
    assert res.status_code == 200 and res.json()["font_style"] == "handwritten"
    assert client.put(f"/api/jobs/{page}/translations", json={"font_style": "wingdings"}).status_code == 400
    assert client.post("/api/batches", files=files, data={"font_style": "nope"}).status_code == 400


def test_scan_cleanup_option_is_saved(client):
    files = [("files", ("p1.png", png_bytes(), "image/png"))]
    batch = client.post("/api/batches", files=files, data={"enhance": "true"}).json()
    page = batch["pages"][0]["id"]
    assert client.get(f"/api/jobs/{page}").json()["enhance"] is True
