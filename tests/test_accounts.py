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
