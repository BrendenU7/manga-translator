"""Optional cleanup for low-quality scans, run before anything else.

Raw scans are often small, noisy (JPEG artifacts), and unevenly lit (grey paper, a dark
corner from the scanner). That hurts detection, OCR and the final look. This step:

1. Upscales small pages (under ~1100 px wide) up to 2x, so text has enough pixels.
2. Black-and-white pages: removes noise, divides out the uneven background lighting
   so paper becomes white, and sets clean black/white levels.
   Color pages: gentle color-preserving denoise.
3. Sharpens slightly to restore edges softened by denoising.

The cleaned page is what gets translated, so the result is cleaner too.
"""
import cv2
import numpy as np

TARGET_WIDTH = 1100   # upscale pages narrower than this (at most 2x)


def is_grayscale(img: np.ndarray) -> bool:
    small = cv2.resize(img, (128, int(128 * img.shape[0] / img.shape[1]) or 1))
    spread = small.max(axis=2).astype(int) - small.min(axis=2).astype(int)
    return float(np.mean(spread)) < 8


def upscale(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if w >= TARGET_WIDTH:
        return img
    f = min(2.0, TARGET_WIDTH / w)
    return cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_LANCZOS4)


def _even_out_lighting(gray: np.ndarray) -> np.ndarray:
    """Divide out uneven lighting (grey paper, a dark scanner corner) but keep the art.

    Lighting changes slowly across a page, while artwork has edges. So we estimate the
    paper brightness from the brightest pixels in big windows (~1/3 of the page: almost
    always some paper in there), then blur that into a smooth lighting map. Grey shading
    smaller than the window keeps its tone because the window still sees white paper.
    """
    k = max(31, min(gray.shape) // 3)
    paper = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    lighting = cv2.GaussianBlur(paper, (0, 0), k / 2)
    return cv2.divide(gray, lighting, scale=255)


def _levels(gray: np.ndarray, black: int = 30, white: int = 215) -> np.ndarray:
    """Stretch so near-white becomes pure white and near-black pure black."""
    out = (gray.astype(np.float32) - black) * (255.0 / (white - black))
    return np.clip(out, 0, 255).astype(np.uint8)


def _sharpen(img: np.ndarray, amount: float = 0.6) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), 1.2)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def enhance(img: np.ndarray) -> np.ndarray:
    """Clean up a scanned page (RGB in, RGB out; may be larger than the input)."""
    img = upscale(img)
    if is_grayscale(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, None, h=9, templateWindowSize=7, searchWindowSize=21)
        gray = _levels(_even_out_lighting(gray))
        gray = _sharpen(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    img = cv2.fastNlMeansDenoisingColored(img, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21)
    return _sharpen(img, 0.4)
