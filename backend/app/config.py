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

# Optional: force a specific font file. By default fonts.py picks the bundled comic font,
# or a system font that covers the target language's characters.
FONT_PATH = os.getenv("FONT_PATH", "")

# Draw translations in ALL CAPS, like English manga lettering
UPPERCASE_TEXT = os.getenv("UPPERCASE_TEXT", "1") == "1"

# Text detector: "ctd" = manga-trained comic-text-detector (downloads ~95 MB once),
# "easyocr" = generic detector (worse on manga, but no extra download)
DETECTOR = os.getenv("DETECTOR", "ctd")
CTD_MODEL_PATH = Path(os.getenv("CTD_MODEL_PATH", ROOT / "models" / "comictextdetector.pt.onnx"))
CTD_THREADS = int(os.getenv("CTD_THREADS", "0"))  # 0 = use all CPU cores

# Send Claude an image of each bubble so it can read the text itself, not just the OCR guess
CLAUDE_VISION = os.getenv("CLAUDE_VISION", "1") == "1"

# Uploads and results are deleted automatically after this many days
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "10"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))          # single image
MAX_BATCH_UPLOAD_MB = int(os.getenv("MAX_BATCH_UPLOAD_MB", "500"))  # total for a batch
USE_GPU = os.getenv("USE_GPU", "0") == "1"

# Languages. Names are shown in the dropdowns in their own script, like most translator UIs.
# "auto": Claude reads the bubble images and works out the language itself.
SOURCE_LANGS = {
    "auto": "Auto-detect",
    "ja": "日本語 (Japanese)",
    "zh-Hans": "简体中文 (Chinese, Simplified)",
    "zh-Hant": "繁體中文 (Chinese, Traditional)",
    "ko": "한국어 (Korean)",
    "en": "English",
}
TARGET_LANGS = {
    "en": "English",
    "zh-Hans": "简体中文",
    "zh-Hant": "繁體中文",
    "ja": "日本語",
    "ko": "한국어",
    "fr": "Français",
    "de": "Deutsch",
    "es": "Español",
    "pt": "Português",
    "it": "Italiano",
    "nl": "Nederlands",
    "pl": "Polski",
    "cs": "Čeština",
    "hu": "Magyar",
    "ro": "Română",
    "hr": "Hrvatski",
    "sr": "Srpski",
    "ru": "Русский",
    "uk": "Українська",
    "tr": "Türkçe",
    "vi": "Tiếng Việt",
    "id": "Bahasa Indonesia",
    "ms": "Bahasa Melayu",
    "tl": "Filipino",
    "th": "ไทย",
    "ar": "العربية",
    "fa": "فارسی",
}
# English names, used in the prompt to Claude
LANG_NAMES_EN = {
    "auto": "unknown (detect it from the images)", "ja": "Japanese",
    "zh-Hans": "Simplified Chinese", "zh-Hant": "Traditional Chinese", "ko": "Korean",
    "en": "English", "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese",
    "it": "Italian", "nl": "Dutch", "pl": "Polish", "cs": "Czech", "hu": "Hungarian",
    "ro": "Romanian", "hr": "Croatian", "sr": "Serbian (Latin script)", "ru": "Russian",
    "uk": "Ukrainian", "tr": "Turkish", "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay",
    "tl": "Filipino", "th": "Thai", "ar": "Arabic", "fa": "Persian",
}
# Source languages usually set vertically and read right-to-left (manga, manhua)
VERTICAL_SOURCES = {"ja", "zh-Hans", "zh-Hant"}
# Target languages without letter case (no ALL CAPS), ones that may break lines between
# any two characters (Chinese, Japanese), and right-to-left ones. Thai has no spaces
# between words but does between phrases, so it wraps at spaces like English.
NO_SPACE_TARGETS = {"zh-Hans", "zh-Hant", "ja", "th"}
CHAR_WRAP_TARGETS = {"zh-Hans", "zh-Hant", "ja"}
RTL_TARGETS = {"ar", "fa"}
