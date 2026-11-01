"""App settings, read from environment variables (see .env.example)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

try:  # load .env if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
FRONTEND_DIR = ROOT / "frontend"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

# "claude" uses the Anthropic API; "mock" returns placeholder text (no key needed).
TRANSLATOR = os.getenv("TRANSLATOR", "claude" if ANTHROPIC_API_KEY else "mock")

# Font used to draw translated text. Point this at a comic font (e.g. Comic Neue, OFL licensed).
_FONT_CANDIDATES = [
    os.getenv("FONT_PATH", ""),
    str(ROOT / "fonts" / "comic.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]
FONT_PATH = next((p for p in _FONT_CANDIDATES if p and Path(p).exists()), "")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
USE_GPU = os.getenv("USE_GPU", "0") == "1"

SOURCE_LANGS = {"ja": "Japanese", "zh": "Chinese (Simplified)", "ko": "Korean"}
TARGET_LANGS = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "pt": "Portuguese", "it": "Italian", "vi": "Vietnamese", "id": "Indonesian",
}
