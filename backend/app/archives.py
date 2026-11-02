"""Turn uploads into a list of page images.

Accepts:
- images: PNG, JPG, WebP, GIF, BMP, AVIF, HEIC/HEIF
- ZIP / CBZ / EPUB (images inside, any folder depth)
- RAR / CBR (needs the UnRAR or 7-Zip program installed on the computer)
- PDF (each page rendered to an image)
Pages come back in natural order, so "page2.png" sorts before "page10.png".
"""
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

from PIL import Image
from . import config

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".avif", ".heic", ".heif"}
ARCHIVE_EXTS = {".zip", ".cbz", ".epub"}
RAR_EXTS = {".rar", ".cbr"}
PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = IMAGE_EXTS | ARCHIVE_EXTS | RAR_EXTS | PDF_EXTS

# HEIC/HEIF (iPhone photos) support comes from the pillow-heif plugin
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

MAX_PAGES = config.MAX_PAGES  # set in config.py (env MAX_PAGES)
MAX_UNCOMPRESSED_BYTES = 2 * 1024 ** 3  # refuse zip bombs
PDF_DPI = 150


class ArchiveError(ValueError):
    pass


@dataclass
class Page:
    name: str      # display name, e.g. "chapter22.cbz / 003.jpg"
    data: bytes    # encoded image bytes (any format Pillow can open)


def natural_key(name: str):
    """Sort key that compares the numbers in a string numerically."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def ext_of(filename: str) -> str:
    return PurePosixPath(filename.lower()).suffix


def _image_entries(filename: str, infos, name_of, size_of, is_dir) -> list:
    entries = [i for i in infos
               if not is_dir(i)
               and ext_of(name_of(i)) in IMAGE_EXTS
               and "__MACOSX" not in name_of(i)
               and not PurePosixPath(name_of(i)).name.startswith(".")]
    if sum(size_of(i) for i in entries) > MAX_UNCOMPRESSED_BYTES:
        raise ArchiveError(f"{filename}: archive is too large when unpacked")
    return sorted(entries, key=lambda i: natural_key(name_of(i)))


def _pages_from_zip(filename: str, data: bytes) -> list[Page]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ArchiveError(f"{filename}: not a valid ZIP/CBZ/EPUB file") from e
    entries = _image_entries(filename, zf.infolist(), lambda i: i.filename,
                             lambda i: i.file_size, lambda i: i.is_dir())
    pages = [Page(f"{filename} / {PurePosixPath(i.filename).name}", zf.read(i)) for i in entries]
    if ext_of(filename) == ".epub":
        # EPUBs also hold small decorations (icons, ornaments): keep only page-sized images
        pages = [p for p in pages if _min_side(p.data) >= 300]
    return pages


def _pages_from_rar(filename: str, data: bytes) -> list[Page]:
    import rarfile

    try:
        rf = rarfile.RarFile(io.BytesIO(data))
        entries = _image_entries(filename, rf.infolist(), lambda i: i.filename,
                                 lambda i: i.file_size, lambda i: i.is_dir())
        return [Page(f"{filename} / {PurePosixPath(i.filename).name}", rf.read(i)) for i in entries]
    except rarfile.RarCannotExec as e:
        raise ArchiveError(f"{filename}: opening RAR/CBR files needs UnRAR or 7-Zip installed "
                           "on this computer (or convert it to CBZ/ZIP)") from e
    except rarfile.Error as e:
        raise ArchiveError(f"{filename}: not a valid RAR/CBR file ({e})") from e


def _min_side(data: bytes) -> int:
    try:
        return min(Image.open(io.BytesIO(data)).size)
    except Exception:
        return 0


def _pages_from_pdf(filename: str, data: bytes) -> list[Page]:
    import pymupdf

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ArchiveError(f"{filename}: could not open PDF") from e
    pages = []
    for i, page in enumerate(doc):
        if i >= MAX_PAGES:
            break
        pix = page.get_pixmap(dpi=PDF_DPI)
        pages.append(Page(f"{filename} / page {i + 1}", pix.tobytes("png")))
    return pages


def _check_image(name: str, data: bytes) -> bool:
    try:
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


def expand_uploads(files: list[tuple[str, bytes]]) -> list[Page]:
    """[(filename, bytes), ...] -> pages, in upload order then natural order within archives.

    Loose images uploaded together are sorted by name, like files in a folder.
    """
    loose: list[Page] = []
    pages: list[Page] = []
    for filename, data in files:
        ext = ext_of(filename)
        if ext in IMAGE_EXTS:
            loose.append(Page(filename, data))
        elif ext in ARCHIVE_EXTS:
            pages += _pages_from_zip(filename, data)
        elif ext in RAR_EXTS:
            pages += _pages_from_rar(filename, data)
        elif ext in PDF_EXTS:
            pages += _pages_from_pdf(filename, data)
        else:
            raise ArchiveError(f"{filename}: unsupported file type")
    loose.sort(key=lambda p: natural_key(p.name))
    pages = loose + pages
    pages = [p for p in pages if _check_image(p.name, p.data)]
    if not pages:
        raise ArchiveError("No images found in the upload")
    if len(pages) > MAX_PAGES:
        raise ArchiveError(f"Too many pages ({len(pages)}); the limit is {MAX_PAGES}")
    return pages
