"""HW1 Level 3 - Raster & Vector Image Processing.

Count buildings on the KPI main campus from two sources, compare counts, and
manually tune Landsat preprocessing so its count approaches the Bing reference.
  - media/bing.png    -> Bing Maps (high-res, reference)
  - media/landsat.png -> Landsat   (low-res, tuning target)

Methods are restricted to those used in SSWU-CV/module_1:
  cv2.cvtColor, cv2.GaussianBlur, cv2.Canny,
  cv2.morphologyEx, cv2.getStructuringElement,
  cv2.findContours, cv2.arcLength, cv2.approxPolyDP, cv2.drawContours
                                       Lesson 2 / image_recognition.py
  cv2.equalizeHist                     Lesson 1 / Im_quality enhanc.py
  cv2.kmeans                           Lesson 1 / Im_klastering.py
  cv2.threshold (manual / OTSU)        Lesson 1 / im_segment.py
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
MEDIA = ROOT / "media"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

BING_PATH = MEDIA / "bing.png"
LANDSAT_PATH = MEDIA / "landsat.png"


# --------------------------------------------------------------------------- #
# Per-image pipeline definitions.                                             #
# Each entry is (stage_name, stage_params). The runner walks the list in      #
# order, so to add/remove a stage for one image just edit its list.           #
# Available stage names: see STAGES dict below.                               #
# --------------------------------------------------------------------------- #
PARAMS_BING = {
    "stages": [
        ("to_gray",         {}),
        ("equalize",        {}),
        # ("negative",        {}),

        ("gaussian_blur",   {"ksize": 13}),
        ("canny",           {"low": 10, "high": 150}),
        ("morph_close",     {"shape": "rect", "ksize": 7, "iters": 2}),
        ("negative",        {}),

        ("find_rectangles", {"approx_eps": 0.04, "min_area": 10, "max_area": 100000,
                             "min_vertices": 4, "max_vertices": 20}),
    ],
}

PARAMS_LANDSAT = {
    "stages": [
        ("to_gray",         {}),
        ("equalize",        {}),
        # ("gaussian_blur",   {"ksize": 5}),
        ("canny",           {"low": 30, "high": 120}),
        ("morph_close",     {"shape": "rect", "ksize": 3, "iters": 1}),
        ("find_rectangles", {"approx_eps": 0.02, "min_area": 20, "max_area": 200,
                             "min_vertices": 20, "max_vertices": 200}),
    ],
}
 

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


def s_morph_close(img, p, ctx):
    shape_map = {"rect": cv2.MORPH_RECT,
                 "ellipse": cv2.MORPH_ELLIPSE,
                 "cross": cv2.MORPH_CROSS}
    shape = shape_map[p.get("shape", "rect")]
    k = cv2.getStructuringElement(shape, (p.get("ksize", 7), p.get("ksize", 7)))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, k,
                            iterations=p.get("iters", 1)), ctx


def s_find_rectangles(img, p, ctx):
    """Lesson 2 / image_recognition.py: arcLength + approxPolyDP, count N-gons."""
    cnts, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    overlay = ctx["original_bgr"].copy()
    eps = p.get("approx_eps", 0.02)
    min_v, max_v = p.get("min_vertices", 4), p.get("max_vertices", 4)
    min_a, max_a = p.get("min_area", 0), p.get("max_area", 1e12)
    matches = []
    for c in cnts:
        a = cv2.contourArea(c)
        if not (min_a <= a <= max_a):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if min_v <= len(approx) <= max_v:
            cv2.drawContours(overlay, [approx], -1, (0, 255, 0), 2)
            matches.append(approx)
    ctx = {**ctx, "count": len(matches), "matches": matches,
           "rect_overlay": overlay}
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), ctx


STAGES = {
    "to_gray":         s_to_gray,
    "equalize":        s_equalize,
    "negative":        s_negative,
    "gaussian_blur":   s_gaussian_blur,
    "kmeans":          s_kmeans,
    "threshold":       s_threshold,
    "canny":           s_canny,
    "morph_close":     s_morph_close,
    "find_rectangles": s_find_rectangles,
}


# --------------------------------------------------------------------------- #
# Pipeline runner.                                                            #
# --------------------------------------------------------------------------- #
def run_pipeline(img_bgr, params):
    ctx = {"original_bgr": img_bgr.copy(), "count": 0,
           "matches": [], "rect_overlay": img_bgr.copy()}
    snapshots = [("original", cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))]
    img = img_bgr
    for name, p in params["stages"]:
        if name not in STAGES:
            raise KeyError(f"unknown stage: {name}. valid: {list(STAGES)}")
        img, ctx = STAGES[name](img, p, ctx)
        snapshots.append((name, img))
    return snapshots, ctx


# --------------------------------------------------------------------------- #
# Visualisations.                                                             #
# --------------------------------------------------------------------------- #
def show_originals(bing, landsat):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].imshow(cv2.cvtColor(bing, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Bing Maps - high-res reference")
    ax[1].imshow(cv2.cvtColor(landsat, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Landsat - low-res tuning target")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Originals: KPI main campus (input images)")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "01_originals.png", dpi=120)
    plt.show()


def show_pipeline_grid(snaps_top, snaps_bot, count_top, count_bot,
                       label_top, label_bot):
    cols = max(len(snaps_top), len(snaps_bot))
    fig, ax = plt.subplots(2, cols, figsize=(3.0 * cols, 6.5))
    if cols == 1:
        ax = np.array([[ax[0]], [ax[1]]])
    for row, (snaps, label, count) in enumerate((
        (snaps_top, label_top, count_top),
        (snaps_bot, label_bot, count_bot),
    )):
        for j in range(cols):
            a = ax[row, j]
            if j < len(snaps):
                name, img = snaps[j]
                a.imshow(img, cmap=None if img.ndim == 3 else "gray")
                a.set_title(name, fontsize=10)
            else:
                a.axis("off")
            a.set_xticks([]); a.set_yticks([])
        ax[row, 0].set_ylabel(f"{label}\ncount = {count}", fontsize=11)
    fig.suptitle("Pipeline manipulations - top: Bing reference, bottom: Landsat")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "02_pipeline.png", dpi=120)
    plt.show()


def show_contours_side_by_side(ctx_b, ctx_l):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].imshow(cv2.cvtColor(ctx_b["rect_overlay"], cv2.COLOR_BGR2RGB))
    ax[0].set_title(f"Bing - detected rectangles: {ctx_b['count']}")
    ax[1].imshow(cv2.cvtColor(ctx_l["rect_overlay"], cv2.COLOR_BGR2RGB))
    ax[1].set_title(f"Landsat - detected rectangles: {ctx_l['count']}")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Vectorized contours (approxPolyDP, 4 vertices) - building candidates")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "03_contours.png", dpi=120)
    plt.show()


# --------------------------------------------------------------------------- #
# Entry point.                                                                #
# --------------------------------------------------------------------------- #
def main():
    bing = cv2.imread(str(BING_PATH), cv2.IMREAD_COLOR)
    landsat = cv2.imread(str(LANDSAT_PATH), cv2.IMREAD_COLOR)
    if bing is None:    raise FileNotFoundError(BING_PATH)
    if landsat is None: raise FileNotFoundError(LANDSAT_PATH)

    # show_originals(bing, landsat)

    snaps_b, ctx_b = run_pipeline(bing, PARAMS_BING)
    snaps_l, ctx_l = run_pipeline(landsat, PARAMS_LANDSAT)

    show_pipeline_grid(snaps_b, snaps_l, ctx_b["count"], ctx_l["count"],
                       "Bing (reference)", "Landsat (tuned)")
    # show_contours_side_by_side(ctx_b, ctx_l)

    print(f"\nBing reference: {ctx_b['count']} buildings")
    print(f"Landsat:        {ctx_l['count']} buildings")
    print(f"Delta:          {ctx_l['count'] - ctx_b['count']:+d}")


if __name__ == "__main__":
    main()
