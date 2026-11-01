"""Rebuild the artwork behind erased text.

Text inside white bubbles is simply painted over (clean.py). Text sitting on artwork
needs the drawing behind it reconstructed. Two ways:

- "lama" (default): LaMa, a neural inpainting model, in a version fine-tuned on anime and
  manga art (TorchScript, ~200 MB, downloaded to models/ on first use). It continues
  screentone, lines and shading convincingly.
- "opencv": OpenCV's Telea inpainting. Instant, but smears over textured areas.

LaMa runs on small crops around each masked area (scaled to at most 512 px), not the
whole page, so a page with a few captions takes seconds even on a CPU.
"""
import logging
import threading
import urllib.request

import cv2
import numpy as np

from .. import config

log = logging.getLogger(__name__)

LAMA_URL = ("https://github.com/Sanster/models/releases/download/"
            "AnimeMangaInpainting/anime-manga-big-lama.pt")
MAX_SIDE = 512   # crops are scaled down to this before LaMa (speed), then back up

_model = None
_device = "cpu"
_model_failed = False
_lock = threading.Lock()  # one LaMa run at a time: it's memory hungry


def _load_lama():
    """Load the model once (downloading it on first use). None if unavailable."""
    global _model, _model_failed, _device
    with _lock:
        if _model is None and not _model_failed:
            try:
                import torch
                path = config.LAMA_MODEL_PATH
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    log.info("Downloading LaMa inpainting model (~200 MB) to %s", path)
                    tmp = path.with_suffix(".part")
                    urllib.request.urlretrieve(LAMA_URL, tmp)
                    tmp.replace(path)
                _device = "cuda" if config.USE_GPU and torch.cuda.is_available() else "cpu"
                _model = torch.jit.load(str(path), map_location=_device).eval()
            except Exception as e:
                log.warning("LaMa unavailable (%s); using OpenCV inpainting", e)
                _model_failed = True
        return _model


def _areas(mask: np.ndarray, margin_ratio: float = 0.6) -> list[tuple[int, int, int, int]]:
    """Boxes around each blob of the mask, with margin for context, overlapping ones merged."""
    h, w = mask.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    boxes = []
    for x, y, bw, bh, _area in stats[1:]:
        m = max(24, int(max(bw, bh) * margin_ratio))
        boxes.append([max(0, x - m), max(0, y - m), min(w, x + bw + m), min(h, y + bh + m)])
    merged = True
    while merged:  # merge overlapping boxes until stable
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(j)
                    merged = True
                    break
            if merged:
                break
    return [tuple(b) for b in boxes]


def _lama_crop(model, crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import torch

    h, w = crop.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    sh, sw = max(8, int(h * scale)), max(8, int(w * scale))
    img = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
    # LaMa needs sides that are multiples of 8
    ph, pw = (8 - sh % 8) % 8, (8 - sw % 8) % 8
    img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    m = np.pad(m, ((0, ph), (0, pw)), mode="constant")

    t_img = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(_device)
    t_mask = torch.from_numpy((m > 0).astype(np.float32))[None, None].to(_device)
    with torch.no_grad():
        out = model(t_img, t_mask)[0].permute(1, 2, 0).cpu().numpy()
    out = np.clip(out * 255, 0, 255).astype(np.uint8)[:sh, :sw]
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)


def inpaint(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill the masked pixels of an RGB image. Only masked pixels change."""
    if not mask.any():
        return img
    model = _load_lama() if config.INPAINTER == "lama" else None
    if model is None:
        return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)

    out = img.copy()
    for x1, y1, x2, y2 in _areas(mask):
        crop_mask = mask[y1:y2, x1:x2]
        with _lock:
            filled = _lama_crop(model, img[y1:y2, x1:x2], crop_mask)
        region = out[y1:y2, x1:x2]
        region[crop_mask > 0] = filled[crop_mask > 0]
    return out
