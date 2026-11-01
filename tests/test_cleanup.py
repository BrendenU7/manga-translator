"""Scan cleanup (enhance.py) and artwork inpainting (inpaint.py)."""
import numpy as np
import pytest

from backend.app import config
from backend.app.pipeline import enhance as en
from backend.app.pipeline import inpaint as ip


# ------------------------------------------------------------------ inpainting

def test_inpaint_only_changes_masked_pixels(monkeypatch):
    import torch
    calls = []

    def fake_lama(img, mask):  # stands in for the real model: paints everything white
        calls.append(tuple(img.shape))
        return torch.ones_like(img)

    monkeypatch.setattr(config, "INPAINTER", "lama")
    monkeypatch.setattr(ip, "_load_lama", lambda: fake_lama)
    img = np.full((600, 900, 3), 80, np.uint8)
    mask = np.zeros((600, 900), np.uint8)
    mask[100:140, 100:300] = 255      # two separate text blobs
    mask[450:480, 600:700] = 255
    out = ip.inpaint(img, mask)

    assert (out[mask > 0] == 255).all()          # filled
    assert (out[mask == 0] == 80).all()          # everything else untouched
    assert len(calls) == 2                       # one small crop per blob, not the page
    assert all(h % 8 == 0 and w % 8 == 0 and max(h, w) <= ip.MAX_SIDE for _, _, h, w in calls)


def test_nearby_blobs_share_one_crop():
    mask = np.zeros((400, 400), np.uint8)
    mask[100:120, 100:150] = 255
    mask[100:120, 170:220] = 255  # 20px apart: their context margins overlap
    assert len(ip._areas(mask)) == 1


def test_opencv_fallback(monkeypatch):
    monkeypatch.setattr(config, "INPAINTER", "opencv")
    img = np.full((100, 100, 3), 200, np.uint8)
    img[40:60, 40:60] = 0
    mask = np.zeros((100, 100), np.uint8)
    mask[40:60, 40:60] = 255
    out = ip.inpaint(img, mask)
    assert out[50, 50].mean() > 150


# ------------------------------------------------------------------ scan cleanup

def _bad_scan(w=500, h=700, seed=0):
    """Grey, unevenly lit, noisy page with some black 'ink' lines."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    page = 210 - 70 * (xx / w) * (yy / h)
    page[::50, :] = 20                                   # ink
    page += rng.normal(0, 8, page.shape)
    gray = np.clip(page, 0, 255).astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


def test_enhance_whitens_paper_evens_light_and_keeps_ink():
    bad = _bad_scan()
    good = en.enhance(bad)
    assert good.shape[1] > bad.shape[1]                  # small page upscaled
    paper = good[good.shape[0] // 2 + 10, :, 0]          # a row of paper, left to right
    assert paper.mean() > bad[bad.shape[0] // 2 + 5, :, 0].mean() + 20   # whiter
    assert paper.std() < 15                              # lighting evened out
    ink_row = good[0, :, 0]
    assert ink_row.mean() < 60                           # ink stays dark


def test_enhance_keeps_color_pages_in_color():
    img = np.zeros((400, 1200, 3), np.uint8)
    img[:, :, 0] = 200  # red page
    out = en.enhance(img)
    assert out.shape == img.shape                        # wide enough: no upscale
    assert out[..., 0].mean() > 150 and out[..., 2].mean() < 50


def test_large_pages_are_not_upscaled():
    assert en.upscale(np.zeros((2000, 1400, 3), np.uint8)).shape == (2000, 1400, 3)
    assert en.upscale(np.zeros((1000, 400, 3), np.uint8)).shape == (2000, 800, 3)  # max 2x
