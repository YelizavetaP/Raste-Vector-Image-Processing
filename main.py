"""HW1 Level 3 - Raster & Vector Image Processing.

Done by Popova Yelyzaveta.

Count buildings on the KPI main campus from two sources, compare counts, and
manually tune Landsat preprocessing so its count approaches the Bing reference.
  - media/bing.png    -> Bing Maps (high-res, reference)
  - media/landsat.png -> Landsat   (low-res, tuning target)

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

# LANDSAT_PATH = MEDIA / "ls2.png"
# LANDSAT_PATH = MEDIA / "ls_2pm.png"



# --------------------------------------------------------------------------- #
# Per-image pipeline definitions.                                             #
# Each entry is (stage_name, stage_params). The runner walks the list in      #
# order, so to add/remove a stage for one image just edit its list.           #
# Available stage names: see STAGES dict below.                               #
# --------------------------------------------------------------------------- #
PARAMS_BING = {
    "stages": [
        ("kmeans",          {"K": 3}),
        ("to_gray",         {}),
        ("equalize",        {}),
        ("gaussian_blur",   {"ksize": 15}),
        ("canny",           {"low": 50, "high": 160}),
        ("morphology",      {"op": "close", "gradient": "rect", "ksize": 8, "iters": 2}),
        ("negative",        {}),
        ("find_rectangles", {"approx_eps": 0.08, "min_area": 400, "max_area": 3000,
                             "min_vertices": 4, "max_vertices": 4}),
    ],
}

PARAMS_LANDSAT = {
    "stages": [
        ("kmeans",          {"K": 9}),
        ("to_gray",         {}),
        ("equalize",        {}),
        # ("gaussian_blur",   {"ksize": 5}),
        ("canny",           {"low": 150, "high": 200}),
        ("morphology",      {"op": "close", "shape": "rect", "ksize": 8, "iters": 2}),
        ("negative",        {}),
        ("find_rectangles", {"approx_eps": 0.08, "min_area": 400, "max_area": 3000,
                             "min_vertices": 4, "max_vertices": 4}),
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
    "morphology":      s_morphology,
    "find_rectangles": s_find_rectangles,
}


# --------------------------------------------------------------------------- #
# Pipeline runner.                                                            #
# --------------------------------------------------------------------------- #
def run_pipeline(img_bgr, params):
    ctx = {"original_bgr": img_bgr.copy(), "count": 0,
           "matches": [], "rect_overlay": img_bgr.copy()}
    snapshots = [("original", cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), {})]
    img = img_bgr
    for name, p in params["stages"]:
        if name not in STAGES:
            raise KeyError(f"unknown stage: {name}. valid: {list(STAGES)}")
        img, ctx = STAGES[name](img, p, ctx)
        snapshots.append((name, img, p))
    return snapshots, ctx


# --------------------------------------------------------------------------- #
# Visualisations.                                                             #
# --------------------------------------------------------------------------- #
def save_stage_snapshots(snaps, prefix):
    """Write every pipeline snapshot to outputs/stages/ as a PNG."""
    out_dir = OUTPUTS / "stages"
    out_dir.mkdir(exist_ok=True)
    rgb_stages = {"original", "find_rectangles"}  # stored as RGB; flip to BGR
    for i, (name, img, _) in enumerate(snaps):
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) \
              if (img.ndim == 3 and name in rgb_stages) else img
        cv2.imwrite(str(out_dir / f"{prefix}_{i:02d}_{name}.png"), out)


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
                name, img, _ = snaps[j]
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

    show_originals(bing, landsat)

    snaps_b, ctx_b = run_pipeline(bing, PARAMS_BING)
    snaps_l, ctx_l = run_pipeline(landsat, PARAMS_LANDSAT)

    save_stage_snapshots(snaps_b, "bing")
    save_stage_snapshots(snaps_l, "landsat")

    show_pipeline_grid(snaps_b, snaps_l, ctx_b["count"], ctx_l["count"],
                       "Bing (reference)", "Landsat (tuned)")
    show_contours_side_by_side(ctx_b, ctx_l)

    print(f"\nBing reference: {ctx_b['count']} buildings")
    print(f"Landsat:        {ctx_l['count']} buildings")
    print(f"Delta:          {ctx_l['count'] - ctx_b['count']:+d}")


if __name__ == "__main__":
    main()



# Notes - results & conclusions
#
# Generated plots and full write-up: see README.md and the outputs/ folder
# (01_originals.png, 02_pipeline.png, 03_contours.png).
#
# General observations:
# - Canny low/high: tested with same values to see what gets rejected/accepted.

# - With Canny alone the contours came out as "double outlines" - two parallel
#   lines on each side of every real edge (one for the building edge, one for
#   its shadow). findContours then traced both as separate contours, so what
#   looked like a building was really a thin frame around it. morphology(close)
#   merges those parallel lines into a single thick band, which fixes it.

# - morphology(close) on Canny output produces white texture-soup with building-
#   shaped holes -> need negative right after to flip into white blobs that
#   findContours can detect.

# - kmeans BEFORE to_gray separates colors that would otherwise collapse to
#   the same grayscale value:
#     * On Bing it reduced trees / sport fields being mis-identified as roofs
#       (green vs gray were getting flattened to the same brightness).
#     * On Landsat it sharpened the boundaries of blurry blobs - by snapping
#       each pixel to one of K dominant colors, fuzzy gradients became flat
#       regions with crisp edges, so objects look more defined and texture
#       noise inside them disappears.

# - Strict 4-vertex filter from the lesson works once approx_eps is loosened to
#   ~0.08 (heavy polygon simplification collapses curvy roofs to 4 corners).
#
# --- Bing (high-res reference) ---
# Final stages:
#   kmeans (K=3)        
#   to_gray
#   equalize            stretches contrast so Canny gradients are stronger
#   gaussian_blur (15)  
#   canny (50, 160)     
#   morphology close    rect ksize=8 iters=2 
#   negative            flip "frame around hole" -> "filled blob"
#   find_rectangles     eps=0.08, area 400-3000, exactly 4 vertices
# Conclusion: not all buildings detected and some false positives remain,
# but a clear subset of real buildings is recognised. 

# --- Landsat (low-res tuning target) ---
# Final stages:
#   kmeans (K=9)        
#   to_gray
#   equalize            critical here - input contrast is very poor
#   (no gaussian_blur)  image is already blurry, extra blur kills weak edges
#   canny (150, 200)    high thresholds because equalize amplified everything
#   morphology close    same as Bing
#   negative            same polarity flip
#   find_rectangles     same filter as Bing
# Conclusion: count is approximate; landsat resolution loses small buildings
# entirely. Same find_rectangles params used as Bing
