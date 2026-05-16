"""Module 2 (Lessons 3 & 4) stage implementations.

Importing exposes `M2_STAGES`, ready to merge into the HW1 STAGES registry:

    from main import STAGES
    from stages_m2 import M2_STAGES
    STAGES.update(M2_STAGES)
"""
import cv2
import numpy as np


# =============================================================================
# Module 2, Lesson 3 — filtering
# =============================================================================
def s_median_blur(img, p, ctx):
    return cv2.medianBlur(img, p.get('ksize', 5)), ctx

def s_bilateral(img, p, ctx):
    return cv2.bilateralFilter(img,
                               p.get('d', 9),
                               p.get('sigma_color', 75),
                               p.get('sigma_space', 75)), ctx

def s_sharpen(img, p, ctx):
    """Manual cross-kernel sharpening."""
    a = p.get('amount', 1.0)
    kernel = np.array([[ 0, -1,  0],
                       [-1, 4+a, -1],
                       [ 0, -1,  0]], dtype=np.float32) / max(a, 1e-6)
    return cv2.filter2D(img, -1, kernel), ctx

def s_average_filter(img, p, ctx):
    """Mean smoothing via ones-kernel filter2D."""
    k = p.get('ksize', 5)
    kernel = np.ones((k, k), np.float32) / (k * k)
    return cv2.filter2D(img, -1, kernel), ctx

def s_box_blur(img, p, ctx):
    """cv2.blur — built-in averaging (same effect as average_filter, different API)."""
    k = p.get('ksize', 5)
    return cv2.blur(img, (k, k)), ctx

def s_pil_sharpen(img, p, ctx):
    """PIL ImageEnhance.Sharpness — factor<1 softens, 1=neutral, >1 sharpens."""
    from PIL import Image, ImageEnhance
    factor = p.get('factor', 2.0)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img)
    out = ImageEnhance.Sharpness(pil).enhance(factor)
    arr = np.array(out)
    return (cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr), ctx

def s_pil_filter(img, p, ctx):
    """PIL ImageFilter predefined modes: BLUR / CONTOUR / DETAIL / EDGE_ENHANCE / SHARPEN / SMOOTH."""
    from PIL import Image, ImageFilter
    mode = p.get('mode', 'DETAIL')
    flt = getattr(ImageFilter, mode)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img)
    out = pil.filter(flt)
    arr = np.array(out)
    return (cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr), ctx


# =============================================================================
# Module 2, Lesson 4 — brightness / contrast
# =============================================================================
def s_log_transform(img, p, ctx):
    """s = c·log(1+r). Brightens dark regions, compresses highlights."""
    c = p.get('c', 255.0 / np.log(1 + 255))
    out = c * np.log1p(img.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8), ctx

def s_pow_transform(img, p, ctx):
    """Gamma: s = c·r^γ. γ<1 brightens, γ>1 darkens."""
    gamma = p.get('gamma', 1.0)
    c = p.get('c', 1.0)
    norm = img.astype(np.float32) / 255.0
    out = c * np.power(norm, gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8), ctx

def s_exp_transform(img, p, ctx):
    """s = (b^r − 1)/(b − 1) on normalized r. Mirror of log — compresses darks, expands brights."""
    b = p.get('b', 2.0)
    norm = img.astype(np.float32) / 255.0
    out = (np.power(b, norm) - 1.0) / (b - 1.0) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8), ctx

def s_hist_stretch(img, p, ctx):
    """Linear stretch: remap [low_pct, high_pct] percentiles to [0, 255]."""
    lo = np.percentile(img, p.get('low_pct', 2))
    hi = np.percentile(img, p.get('high_pct', 98))
    if hi <= lo:
        return img, ctx
    out = np.clip((img.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
    return out.astype(np.uint8), ctx

def s_hist_slide(img, p, ctx):
    """Additive brightness shift, clamped to 0..255."""
    out = img.astype(np.int16) + p.get('shift', 30)
    return np.clip(out, 0, 255).astype(np.uint8), ctx


def _segment_eq_lut(hist, lo, hi):
    """Equalize one [lo, hi] segment of an intensity histogram. Returns LUT slice for that range."""
    total = hist.sum()
    if total == 0:
        return np.arange(lo, hi + 1, dtype=np.uint8)
    cdf = hist.cumsum() / total
    return np.round(lo + cdf * (hi - lo)).astype(np.uint8)

def _bbhe_gray(g):
    """BBHE: split histogram at mean intensity, equalize halves independently."""
    L, Xm = 256, int(g.mean())
    hist_low, _ = np.histogram(g, bins=Xm + 1,     range=(0, Xm + 1))
    hist_up,  _ = np.histogram(g, bins=L - Xm - 1, range=(Xm + 1, L))
    lut = np.empty(L, dtype=np.uint8)
    lut[:Xm + 1] = _segment_eq_lut(hist_low, 0, Xm)
    lut[Xm + 1:] = _segment_eq_lut(hist_up,  Xm + 1, L - 1)
    return cv2.LUT(g, lut)

def _dsihe_gray(g):
    """DSIHE: split histogram at CDF median (intensity where CDF ≈ 0.5)."""
    L = 256
    hist, _ = np.histogram(g, bins=L, range=(0, L))
    cdf = hist.cumsum() / max(hist.sum(), 1)
    Xm = int(np.clip(np.searchsorted(cdf, 0.5), 0, L - 2))
    lut = np.empty(L, dtype=np.uint8)
    lut[:Xm + 1] = _segment_eq_lut(hist[:Xm + 1], 0, Xm)
    lut[Xm + 1:] = _segment_eq_lut(hist[Xm + 1:], Xm + 1, L - 1)
    return cv2.LUT(g, lut)

def _eq_on_y(img, gray_fn):
    """Apply a grayscale equalization function on the Y channel of YCbCr (color), or directly (gray)."""
    if img.ndim == 3:
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycc[..., 0] = gray_fn(ycc[..., 0])
        return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    return gray_fn(img)

def s_bbhe(img, p, ctx):
    """Brightness-preserving Bi-Histogram Equalization."""
    return _eq_on_y(img, _bbhe_gray), ctx

def s_dsihe(img, p, ctx):
    """Dualistic Sub-Image Histogram Equalization."""
    return _eq_on_y(img, _dsihe_gray), ctx


# =============================================================================
# Module 2, Lesson 3 — PIL color extras (filtr_im_PIL.py)
# =============================================================================
def s_sepia(img, p, ctx):
    """Classic sepia matrix transform."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    arr = img.astype(np.float32)
    b, g, r = arr[..., 0], arr[..., 1], arr[..., 2]
    nr = 0.393 * r + 0.769 * g + 0.189 * b
    ng = 0.349 * r + 0.686 * g + 0.168 * b
    nb = 0.272 * r + 0.534 * g + 0.131 * b
    out = np.stack([nb, ng, nr], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8), ctx


def s_warm_cool_mask(img, p, ctx):
    """Separate warm vs cool pixels via the LAB b-channel.

    `b` in LAB is the blue↔yellow axis: pixels with b<128 are cool/bluish,
    b>=128 are warm/yellowish. Useful for distinguishing e.g. green roofs
    (cool) from grass/trees (warm) which collapse together in RGB.

    Params:
      keep      : 'cool' (default) or 'warm'  — which side to keep
      threshold : int 0..255 (default 128)    — split point on the b channel
      mode      : 'masked' (default) returns the color image with rejected
                  pixels set to black; 'mask' returns a binary 0/255 image.
    """
    if img.ndim != 3:
        return img, ctx
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b = lab[..., 2]
    thr = p.get("threshold", 128)
    keep_cool = p.get("keep", "cool") == "cool"
    mask = (b < thr) if keep_cool else (b >= thr)
    if p.get("mode", "masked") == "mask":
        return (mask.astype(np.uint8) * 255), ctx
    out = img.copy()
    out[~mask] = 0
    return out, ctx


# Public registry — merge into the HW1 STAGES dict
M2_STAGES = {
    # filters (Lesson 3)
    'median_blur':    s_median_blur,
    'bilateral':      s_bilateral,
    'sharpen':        s_sharpen,
    'average_filter': s_average_filter,
    'box_blur':       s_box_blur,
    'pil_sharpen':    s_pil_sharpen,
    'pil_filter':     s_pil_filter,
    # brightness / contrast (Lesson 4)
    'log_transform':  s_log_transform,
    'pow_transform':  s_pow_transform,
    'exp_transform':  s_exp_transform,
    'hist_stretch':   s_hist_stretch,
    'hist_slide':     s_hist_slide,
    'bbhe':           s_bbhe,
    'dsihe':          s_dsihe,
    # extras
    'sepia':            s_sepia,
    'warm_cool_mask':   s_warm_cool_mask,
}
