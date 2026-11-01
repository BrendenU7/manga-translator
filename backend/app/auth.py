"""User accounts, login sessions and API keys (SQLite, standard library only).

- Passwords are hashed with scrypt and a random per-user salt; we never store the password.
- A login creates a random session token. The browser keeps it in an httponly cookie;
  the database only keeps a SHA-256 hash of it, so a leaked database can't be used to log in.
- An API key (one per user, for the browser extension) works the same way but is sent as
  `Authorization: Bearer <key>`. It's shown once when created and can be revoked.
"""
import hashlib
import hmac
import re
import secrets
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass

from fastapi import HTTPException, Request

from . import config

COOKIE_NAME = "mt_session"
SESSION_DAYS = 30
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{3,32}$")
MIN_PASSWORD = 8


@dataclass
class User:
    id: int
    username: str


def _db() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DATA_DIR / "app.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL COLLATE NOCASE,
        password_hash TEXT NOT NULL,
        created_at REAL NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
        token_hash TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id),
        expires_at REAL NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS api_keys (
        user_id INTEGER PRIMARY KEY REFERENCES users(id),
        key_hash TEXT UNIQUE NOT NULL,
        created_at REAL NOT NULL)""")
    return conn


# ------------------------------------------------------------------ passwords

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1)
    return hmac.compare_digest(digest.hex(), digest_hex)


# ------------------------------------------------------------------ users

def create_user(username: str, password: str) -> User:
    username = username.strip()
    if not USERNAME_RE.match(username):
        raise ValueError("Username must be 3-32 characters: letters, numbers, _ . -")
    if len(password) < MIN_PASSWORD:
        raise ValueError(f"Password must be at least {MIN_PASSWORD} characters")
    with closing(_db()) as conn, conn:
        try:
            cur = conn.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                               (username, hash_password(password), time.time()))
        except sqlite3.IntegrityError:
            raise ValueError("That username is taken")
        return User(cur.lastrowid, username)


def authenticate(username: str, password: str) -> User | None:
    with closing(_db()) as conn:
        row = conn.execute("SELECT id, username, password_hash FROM users WHERE username = ?",
                           (username.strip(),)).fetchone()
    if row is None:
        hash_password(password)  # same work either way, so timing doesn't reveal valid usernames
        return None
    return User(row[0], row[1]) if verify_password(password, row[2]) else None


# ------------------------------------------------------------------ sessions

def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user: User) -> str:
    token = secrets.token_urlsafe(32)
    with closing(_db()) as conn, conn:
        conn.execute("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                     (_token_hash(token), user.id, time.time() + SESSION_DAYS * 86400))
    return token


def delete_session(token: str):
    with closing(_db()) as conn, conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))


def user_for_token(token: str | None) -> User | None:
    if not token:
        return None
    with closing(_db()) as conn:
        row = conn.execute(
            "SELECT u.id, u.username FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_at > ?", (_token_hash(token), time.time())).fetchone()
    return User(row[0], row[1]) if row else None


def purge_expired_sessions():
    with closing(_db()) as conn, conn:
        conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (time.time(),))


# ------------------------------------------------------------------ API keys (extension)

API_KEY_PREFIX = "mtk_"


def create_api_key(user: User) -> str:
    """New API key for the user, replacing any previous one. Returned once, never stored."""
    key = API_KEY_PREFIX + secrets.token_urlsafe(32)
    with closing(_db()) as conn, conn:
        conn.execute("INSERT OR REPLACE INTO api_keys (user_id, key_hash, created_at) VALUES (?, ?, ?)",
                     (user.id, _token_hash(key), time.time()))
    return key


def revoke_api_key(user: User):
    with closing(_db()) as conn, conn:
        conn.execute("DELETE FROM api_keys WHERE user_id = ?", (user.id,))


def api_key_created_at(user: User) -> float | None:
    with closing(_db()) as conn:
        row = conn.execute("SELECT created_at FROM api_keys WHERE user_id = ?", (user.id,)).fetchone()
    return row[0] if row else None


def user_for_api_key(key: str) -> User | None:
    with closing(_db()) as conn:
        row = conn.execute("SELECT u.id, u.username FROM api_keys k JOIN users u ON u.id = k.user_id "
                           "WHERE k.key_hash = ?", (_token_hash(key),)).fetchone()
    return User(row[0], row[1]) if row else None


# ------------------------------------------------------------------ FastAPI dependency

def current_user(request: Request) -> User:
    """Use as `user: User = Depends(current_user)` on routes that need a login.

    Accepts the website's session cookie, or an API key in the Authorization header.
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        user = user_for_api_key(auth_header[7:].strip())
        if user is None:
            raise HTTPException(401, "Invalid or revoked extension key")
        return user
    user = user_for_token(request.cookies.get(COOKIE_NAME))
    if user is None:
        raise HTTPException(401, "Please log in")
    return user
