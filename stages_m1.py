"""Module 1 (HW1) stage implementations.

Importing exposes `M1_STAGES`, ready to merge into the pipeline STAGES registry:

    from pipeline import STAGES
    from stages_m1 import M1_STAGES
    STAGES.update(M1_STAGES)
"""
import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Stage functions. Signature: (img, params, ctx) -> (img_out, ctx_out)        #
#   ctx carries shared state: original_bgr, count, matches, rect_overlay.     #
# --------------------------------------------------------------------------- #
def s_to_gray(img, p, ctx):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), ctx


def s_equalize(img, p, ctx):
    return cv2.equalizeHist(img), ctx


def s_negative(img, p, ctx):
    return cv2.bitwise_not(img), ctx


def s_gaussian_blur(img, p, ctx):
    k = p.get("ksize", 3)
    return cv2.GaussianBlur(img, (k, k), p.get("sigma", 0)), ctx


def s_kmeans(img, p, ctx):
    K = p.get("K", 4)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 \
          else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    Z = rgb.reshape((-1, 3)).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, lbl, c = cv2.kmeans(Z, K, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    res = np.uint8(c)[lbl.flatten()].reshape(rgb.shape)
    return cv2.cvtColor(res, cv2.COLOR_RGB2BGR), ctx


def s_threshold(img, p, ctx):
    flag = cv2.THRESH_BINARY_INV if p.get("invert", True) else cv2.THRESH_BINARY
    thr = p.get("thr", 127)
    if p.get("use_otsu", False):
        flag |= cv2.THRESH_OTSU
        thr = 0
    _, b = cv2.threshold(img, thr, 255, flag)
    return b, ctx


def s_canny(img, p, ctx):
    return cv2.Canny(img, p.get("low", 10), p.get("high", 250)), ctx


def s_morphology(img, p, ctx):
    shape_map = {"rect":    cv2.MORPH_RECT,
                 "ellipse": cv2.MORPH_ELLIPSE,
                 "cross":   cv2.MORPH_CROSS}
    op_map = {"close":    cv2.MORPH_CLOSE,
              "open":     cv2.MORPH_OPEN,
              "gradient": cv2.MORPH_GRADIENT,
              "tophat":   cv2.MORPH_TOPHAT,
              "blackhat": cv2.MORPH_BLACKHAT}
    shape = shape_map[p.get("shape", "rect")]
    op = op_map[p.get("op", "close")]
    k = cv2.getStructuringElement(shape, (p.get("ksize", 7), p.get("ksize", 7)))
    return cv2.morphologyEx(img, op, k, iterations=p.get("iters", 1)), ctx


def s_find_rectangles(img, p, ctx):
    """Lesson 2 / image_recognition.py: arcLength + approxPolyDP, count N-gons.

    Params:
      approx_eps     — fraction of perimeter used as approxPolyDP tolerance
      min_vertices   — minimum vertex count to accept (default 4)
      max_vertices   — maximum vertex count to accept (default 4)
      min_area       — minimum contour area (default 0)
      max_area       — maximum contour area (default 1e12)
      only_convex    — if True, reject contours whose vertices form a
                       concave polygon (e.g. arrow / Pac-Man / pinched shapes).
                       Default False.
    """
    cnts, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    overlay = ctx["original_bgr"].copy()
    eps = p.get("approx_eps", 0.02)
    min_v, max_v = p.get("min_vertices", 4), p.get("max_vertices", 4)
    min_a, max_a = p.get("min_area", 0), p.get("max_area", 1e12)
    only_convex = p.get("only_convex", False)
    matches = []
    for c in cnts:
        a = cv2.contourArea(c)
        if not (min_a <= a <= max_a):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if not (min_v <= len(approx) <= max_v):
            continue
        if only_convex and not cv2.isContourConvex(approx):
            continue
        cv2.drawContours(overlay, [approx], -1, (0, 255, 0), 2)
        matches.append(approx)
    ctx = {**ctx, "count": len(matches), "matches": matches,
           "rect_overlay": overlay}
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), ctx


# Public registry — merge into the pipeline STAGES dict
M1_STAGES = {
    "to_gray":         s_to_gray,
    "equalize":        s_equalize,
    "negative":        s_negative,
    "gaussian_blur":   s_gaussian_blur,
    "kmeans":          s_kmeans,
    "threshold":       s_threshold,
    "canny":           s_canny,
    "morphology":      s_morphology,
    "find_rectangles": s_find_rectangles,
}
