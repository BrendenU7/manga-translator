"""Accounts, privacy between users, dashboard listing, delete and 10-day auto-delete."""
import io
import os
import time

import numpy as np
import pytest
from PIL import Image

from backend.app import auth, config, jobs


def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 60), "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def app(tmp_path, monkeypatch):
    from backend.app import main
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(jobs, "run_pipeline",
                        lambda img, *a, **k: ([], img, np.zeros_like(img)))
    return main.app


def new_client(app, username):
    from fastapi.testclient import TestClient
    c = TestClient(app)
    r = c.post("/api/auth/signup", json={"username": username, "password": "password123"})
    assert r.status_code == 200, r.text
    return c


def upload(client) -> dict:
    r = client.post("/api/batches", files=[("files", ("p1.png", png_bytes(), "image/png"))])
    assert r.status_code == 200, r.text
    batch = r.json()
    end = time.time() + 10
    while client.get(f"/api/batches/{batch['id']}").json()["status"] != "done":
        assert time.time() < end
        time.sleep(0.05)
    return client.get(f"/api/batches/{batch['id']}").json()


def test_password_hashing():
    stored = auth.hash_password("hunter2hunter2")
    assert "hunter2" not in stored
    assert auth.verify_password("hunter2hunter2", stored)
    assert not auth.verify_password("wrong-password", stored)


def test_signup_login_logout(app):
    from fastapi.testclient import TestClient
    c = new_client(app, "alice")
    assert c.get("/api/auth/me").json()["username"] == "alice"
    c.post("/api/auth/logout")
    assert c.get("/api/auth/me").status_code == 401

    fresh = TestClient(app)
    assert fresh.post("/api/auth/login", json={"username": "alice", "password": "nope-nope"}).status_code == 401
    assert fresh.post("/api/auth/login", json={"username": "ALICE", "password": "password123"}).status_code == 200
    assert fresh.get("/api/auth/me").status_code == 200


def test_signup_validation(app):
    from fastapi.testclient import TestClient
    c = TestClient(app)
    assert c.post("/api/auth/signup", json={"username": "ab", "password": "password123"}).status_code == 400
    assert c.post("/api/auth/signup", json={"username": "bob", "password": "short"}).status_code == 400
    new_client(app, "carol")
    assert c.post("/api/auth/signup", json={"username": "Carol", "password": "password123"}).status_code == 400


def test_must_be_logged_in(app):
    from fastapi.testclient import TestClient
    c = TestClient(app)
    assert c.get("/api/batches").status_code == 401
    r = c.post("/api/batches", files=[("files", ("p1.png", png_bytes(), "image/png"))])
    assert r.status_code == 401


def test_users_only_see_their_own_uploads(app):
    alice, bob = new_client(app, "alice"), new_client(app, "bob")
    batch = upload(alice)
    page = batch["pages"][0]["id"]

    assert [b["id"] for b in alice.get("/api/batches").json()["batches"]] == [batch["id"]]
    assert bob.get("/api/batches").json()["batches"] == []
    # Everything of Alice's is a 404 for Bob
    for url in [f"/api/batches/{batch['id']}", f"/api/batches/{batch['id']}/download",
                f"/api/jobs/{page}", f"/api/jobs/{page}/image/original"]:
        assert bob.get(url).status_code == 404, url
        assert alice.get(url).status_code == 200, url
    assert bob.delete(f"/api/batches/{batch['id']}").status_code == 404


def test_dashboard_shows_expiry_and_delete_works(app):
    alice = new_client(app, "alice")
    batch = upload(alice)
    listed = alice.get("/api/batches").json()
    assert listed["retention_days"] == config.RETENTION_DAYS
    b = listed["batches"][0]
    assert abs(b["expires_at"] - b["created_at"] - config.RETENTION_DAYS * 86400) < 1
    assert b["thumb_url"].endswith("/image/translated")

    assert alice.delete(f"/api/batches/{batch['id']}").status_code == 200
    assert alice.get("/api/batches").json()["batches"] == []
    assert not (config.DATA_DIR / batch["pages"][0]["id"]).exists()


def test_auto_delete_after_retention(app):
    alice = new_client(app, "alice")
    old, new = upload(alice), upload(alice)
    # Age the first batch past the retention window
    b = jobs.get_batch(old["id"])
    b.created_at -= (config.RETENTION_DAYS + 1) * 86400
    b.save()

    assert jobs.cleanup_expired() == 1
    ids = [x["id"] for x in alice.get("/api/batches").json()["batches"]]
    assert ids == [new["id"]]
    assert not (config.DATA_DIR / old["pages"][0]["id"]).exists()


def test_old_unowned_batches_use_file_age(app, tmp_path):
    # A batch saved before accounts existed (no created_at) isn't deleted on the spot
    legacy = jobs.Batch(id="aaaaaaaaaaaa", name="old.png")
    legacy.save()
    assert jobs.cleanup_expired() == 0
    old = time.time() - (config.RETENTION_DAYS + 1) * 86400
    os.utime(legacy.path, (old, old))
    jobs._batches.clear()
    assert jobs.cleanup_expired() == 1


def test_extension_api_key(app):
    from fastapi.testclient import TestClient
    alice = new_client(app, "alice")
    key = alice.post("/api/auth/api-key").json()["key"]
    assert key.startswith("mtk_")

    ext = TestClient(app)  # no cookies, like the extension
    headers = {"Authorization": f"Bearer {key}"}
    assert ext.get("/api/auth/me", headers=headers).json()["username"] == "alice"
    r = ext.post("/api/batches", headers=headers, files=[("files", ("p.png", png_bytes(), "image/png"))])
    assert r.status_code == 200
    assert len(alice.get("/api/batches").json()["batches"]) == 1  # shows on Alice's dashboard

    # An API key can't mint more keys, only the logged-in website can
    assert ext.post("/api/auth/api-key", headers=headers).status_code == 401

    # A new key replaces the old one; revoking kills it
    key2 = alice.post("/api/auth/api-key").json()["key"]
    assert ext.get("/api/auth/me", headers=headers).status_code == 401
    alice.delete("/api/auth/api-key")
    assert ext.get("/api/auth/me", headers={"Authorization": f"Bearer {key2}"}).status_code == 401


# --------------------------------------------------------------------- damaged / old data

def _forget_memory():
    jobs._jobs.clear()
    jobs._batches.clear()


def test_dashboard_survives_empty_job_file_from_old_windows_bug(app):
    alice = new_client(app, "alice")
    batch = upload(alice)
    page = batch["pages"][0]["id"]
    (config.DATA_DIR / page / "job.json").write_text("", encoding="utf-8")  # what the bug left behind
    _forget_memory()                                                      # as after a server restart

    res = alice.get("/api/batches")
    assert res.status_code == 200
    b = res.json()["batches"][0]
    assert b["total"] == 1 and b["pages"][0]["status"] == "done"          # rebuilt from its images
    assert alice.get(f"/api/jobs/{page}/image/translated").status_code == 200


def test_damaged_page_without_result_shows_an_error(app):
    alice = new_client(app, "alice")
    batch = upload(alice)
    page = batch["pages"][0]["id"]
    (config.DATA_DIR / page / "job.json").write_text("{not json", encoding="utf-8")
    (config.DATA_DIR / page / "translated.png").unlink()
    _forget_memory()
    p = alice.get(f"/api/batches/{batch['id']}").json()["pages"][0]
    assert p["status"] == "error"


def test_page_cut_off_by_restart_is_marked_interrupted(app):
    alice = new_client(app, "alice")
    batch = upload(alice)
    page = batch["pages"][0]["id"]
    job = jobs.get_job(page)
    job.status, job.stage = "running", "translating"
    job.save()
    _forget_memory()
    data = alice.get(f"/api/jobs/{page}").json()
    assert data["status"] == "error" and "restart" in data["error"]


def test_saves_are_utf8_and_never_rely_on_the_os_default(app, monkeypatch):
    from pathlib import Path
    real = Path.write_text

    def strict(self, data, encoding=None, *a, **k):
        assert encoding is not None, f"write_text without an encoding: {self}"
        return real(self, data, encoding, *a, **k)

    monkeypatch.setattr(Path, "write_text", strict)
    alice = new_client(app, "alice")
    r = alice.post("/api/batches", files=[("files", ("p.png", png_bytes(), "image/png"))],
                   data={"glossary": "悟飯 = Gohan", "name": "ドラゴンボール"})
    assert r.status_code == 200
    page = r.json()["pages"][0]["id"]
    raw = (config.DATA_DIR / page / "job.json").read_bytes().decode("utf-8")
    assert "悟飯" in raw


def test_save_retries_when_windows_locks_the_file(tmp_path, monkeypatch):
    """Windows refuses to replace a file another reader has open for a moment
    ("Access is denied"); the save should wait and retry instead of failing the page."""
    import os
    from backend.app import jobs
    real_replace, calls = os.replace, []

    def flaky_replace(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(5, "Access is denied")
        real_replace(src, dst)

    monkeypatch.setattr(jobs.os, "replace", flaky_replace)
    target = tmp_path / "job.json"
    jobs._write_json(target, {"title": "ページ"})
    assert jobs._read_json(target) == {"title": "ページ"}
    assert len(calls) == 3
    assert not list(tmp_path.glob("*.tmp"))
