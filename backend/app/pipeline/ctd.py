"""Manga-trained text detector (comic-text-detector, ONNX).

Model: https://github.com/dmMaze/comic-text-detector (weights from the
manga-image-translator release). It was trained on manga/comic pages, so unlike
EasyOCR's CRAFT it finds whole text blocks in bubbles, hand-lettering next to
screentone, and vertical Japanese.

One forward pass gives us:
  - blk: YOLO-style text-block boxes   -> our TextRegions
  - seg: per-pixel "this is text" mask  -> used to erase lettering precisely
"""
import logging
import os
import threading
import urllib.request

import cv2
import numpy as np

from .. import config

log = logging.getLogger(__name__)

MODEL_URL = ("https://github.com/zyddnys/manga-image-translator/releases/download/"
             "beta-0.3/comictextdetector.pt.onnx")
INPUT_SIZE = 1024
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.35

_session = None
_lock = threading.Lock()


def _download(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    log.info("Downloading text detector model (~95 MB) to %s", path)
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.replace(path)


def get_session():
    """Load the ONNX model once (downloading it on first use)."""
    global _session
    import onnxruntime as ort

    with _lock:
        if _session is None:
            path = config.CTD_MODEL_PATH
            if not path.exists():
                _download(path)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = config.CTD_THREADS or (os.cpu_count() or 2)
            providers = ["CPUExecutionProvider"]
            if config.USE_GPU:
                providers.insert(0, "CUDAExecutionProvider")
            _session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        return _session


def detect(img: np.ndarray) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    """Return (text block boxes, text mask) for an RGB image.

    The mask is a uint8 array the same size as the image, 255 where there's lettering.
    """
    h, w = img.shape[:2]
    # Letterbox: scale the page to fit 1024x1024, pad the rest with black
    scale = min(INPUT_SIZE / h, INPUT_SIZE / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    blk, seg, _det = get_session().run(None, {"images": blob})

    # Boxes: rows are [cx, cy, w, h, objectness, class scores...] in 1024-space
    b = blk[0]
    conf = b[:, 4] * b[:, 5:].max(axis=1)
    keep = conf > CONF_THRESHOLD
    b, conf = b[keep], conf[keep]
    xywh = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 2], b[:, 3]], axis=1)
    idx = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)).flatten()

    boxes = []
    for i in idx:
        x, y, bw, bh = xywh[i] / scale
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
        if x2 - x1 > 3 and y2 - y1 > 3:
            boxes.append((x1, y1, x2, y2))

    # Text mask, resized back to the original page
    prob = cv2.resize(seg[0, 0, :nh, :nw], (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (prob > 0.3).astype(np.uint8) * 255
    return boxes, mask
