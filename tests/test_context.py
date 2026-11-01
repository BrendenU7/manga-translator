"""Chapters: pages run in parallel but translate in order, each seeing earlier dialogue."""
import io
import threading
import time

import numpy as np
import pytest
from PIL import Image

import backend.app.pipeline as pipeline
from backend.app import config, jobs
from backend.app.pipeline import translate
from backend.app.pipeline.models import TextRegion


def page_png(shade: int) -> bytes:
    """Each test page is a flat grey; its shade tells us which page it is."""
    buf = io.BytesIO()
    Image.new("RGB", (40, 60), (shade, shade, shade)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.app import main
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    c = TestClient(main.app)
    c.post("/api/auth/signup", json={"username": "reader", "password": "password123"})
    return c


def run_chapter(client, shades):
    files = [("files", (f"p{i + 1}.png", page_png(s), "image/png")) for i, s in enumerate(shades)]
    batch = client.post("/api/batches", files=files).json()
    end = time.time() + 20
    while client.get(f"/api/batches/{batch['id']}").json()["status"] != "done":
        assert time.time() < end, "batch did not finish"
        time.sleep(0.05)
    return client.get(f"/api/batches/{batch['id']}").json()


def test_pages_translate_in_order_with_previous_dialogue(client, monkeypatch):
    calls = []  # (page number, previous lines) in the order translations happened
    lock = threading.Lock()

    def fake_prepare(img, source_lang, progress):
        n = int(img[0, 0, 0]) // 10
        time.sleep(0.05 * (4 - n))  # later pages finish preparing FIRST
        return [TextRegion(id=0, box=(0, 0, 10, 10), source_text=f"src{n}")], None

    def fake_translate(regions, s, t, page=None, style="", glossary="", previous=None, **kw):
        n = int(page[0, 0, 0]) // 10
        with lock:
            calls.append((n, [p["text"] for p in previous or []]))
        regions[0].translated_text = f"line{n}"
        return regions

    monkeypatch.setattr(pipeline, "prepare_page", fake_prepare)
    monkeypatch.setattr(pipeline, "translate_regions", fake_translate)
    monkeypatch.setattr(pipeline, "finish_page", lambda img, *a: (img, img))

    batch = run_chapter(client, [10, 20, 30])
    assert batch["done"] == 3
    assert [n for n, _ in calls] == [1, 2, 3]                    # in page order...
    assert dict(calls) == {1: [], 2: ["line1"], 3: ["line1", "line2"]}  # ...with context


def test_a_failed_page_does_not_block_the_next(client, monkeypatch):
    def fake_prepare(img, source_lang, progress):
        if int(img[0, 0, 0]) == 10:
            raise RuntimeError("bad page")
        return [TextRegion(id=0, box=(0, 0, 10, 10), source_text="x")], None

    def fake_translate(regions, *a, **k):
        regions[0].translated_text = "ok"
        return regions

    monkeypatch.setattr(pipeline, "prepare_page", fake_prepare)
    monkeypatch.setattr(pipeline, "translate_regions", fake_translate)
    monkeypatch.setattr(pipeline, "finish_page", lambda img, *a: (img, img))

    batch = run_chapter(client, [10, 20])
    assert [p["status"] for p in batch["pages"]] == ["error", "done"]


def test_prompt_includes_previous_dialogue():
    regions = [TextRegion(id=0, box=(0, 0, 1, 1), source_text="待て")]
    previous = [{"source": "悟飯", "text": "Gohan!"}]
    system, messages = translate.build_messages(regions, "ja", "en", previous=previous)
    assert "Stay consistent" in system
    assert '"Gohan!"' in messages[0]["content"]
    _, messages = translate.build_messages(regions, "ja", "en")
    assert "previous page" not in messages[0]["content"]


def test_context_is_capped():
    previous = [{"source": str(i), "text": f"t{i}"} for i in range(100)]
    block = translate._context_block(previous)
    assert '"t99"' in block and '"t59"' not in block and '"t60"' in block
