"""Pick a font that can actually draw the translated text.

Comic Neue (bundled) only covers Latin letters. For Russian, Chinese, Thai, Arabic, etc.
we walk a list of candidate fonts (comic-style first, then common system fonts on
Windows, macOS and Linux) and use the first one whose character map covers every
character in the text. Missing fonts are skipped, so the same list works on any OS.
"""
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

from .. import config

_ROOT = config.ROOT
_WIN = "C:/Windows/Fonts"

CANDIDATES = [
    # Comic-style first
    str(_ROOT / "fonts" / "comic.ttf"),
    str(_ROOT / "fonts" / "ComicNeue-Bold.ttf"),
    f"{_WIN}/comicbd.ttf",                                   # Comic Sans Bold: Latin, Cyrillic, Greek
    # Latin / Cyrillic / Greek / Vietnamese / Arabic
    f"{_WIN}/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    f"{_WIN}/tahomabd.ttf",                                  # Arabic, Persian, Thai
    # Chinese / Japanese / Korean
    f"{_WIN}/msyhbd.ttc",                                    # Microsoft YaHei Bold (Simplified)
    f"{_WIN}/msjhbd.ttc",                                    # Microsoft JhengHei Bold (Traditional)
    f"{_WIN}/YuGothB.ttc",                                   # Yu Gothic Bold (Japanese)
    f"{_WIN}/malgunbd.ttf",                                  # Malgun Gothic Bold (Korean)
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    # Thai
    f"{_WIN}/leelawdb.ttf",                                  # Leelawadee UI Bold
    "/System/Library/Fonts/Supplemental/Thonburi.ttc",
    "/usr/share/fonts/opentype/tlwg/Loma-Bold.otf",
    # Arabic / Persian
    f"{_WIN}/segoeuib.ttf",
    "/System/Library/Fonts/GeezaPro.ttc",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]


@lru_cache(maxsize=None)
def _charset(path: str) -> frozenset[int]:
    """All code points a font file can draw (first face of a .ttc collection)."""
    from fontTools.ttLib import TTCollection, TTFont

    try:
        if path.lower().endswith((".ttc", ".otc")):
            font = TTCollection(path, lazy=True).fonts[0]
        else:
            font = TTFont(path, lazy=True, fontNumber=0)
        cmap = font.getBestCmap() or {}
        return frozenset(cmap)
    except Exception:
        return frozenset()


def _candidates() -> list[str]:
    paths = ([config.FONT_PATH] if config.FONT_PATH else []) + CANDIDATES
    seen, out = set(), []
    for p in paths:
        if p and p not in seen and Path(p).exists():
            seen.add(p)
            out.append(p)
    return out


def font_path_for(text: str) -> str:
    """First candidate font that covers every visible character in `text`."""
    needed = {ord(c) for c in text if not c.isspace()}
    best, best_missing = "", None
    for path in _candidates():
        missing = len(needed - _charset(path))
        if missing == 0:
            return path
        if best_missing is None or missing < best_missing:
            best, best_missing = path, missing
    return best  # nothing covers everything: use the closest match


@lru_cache(maxsize=256)
def load(path: str, size: int, complex_layout: bool = False) -> ImageFont.FreeTypeFont:
    """complex_layout: use libraqm (if this Pillow build has it) to position stacked marks,
    e.g. Thai vowels and tone marks. Otherwise BASIC layout: Arabic/Persian are shaped by
    us (see typeset.shape), so they look the same with or without libraqm."""
    if not path:
        return ImageFont.load_default(size)
    from PIL import features
    engine = (ImageFont.Layout.RAQM if complex_layout and features.check("raqm")
              else ImageFont.Layout.BASIC)
    return ImageFont.truetype(path, size, layout_engine=engine)
