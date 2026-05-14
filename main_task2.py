"""HW2 Level 3 - Filtering & Image Quality Enhancement.

Done by Popova Yelyzaveta.

Continuation of HW1: same KPI building-counting goal, now extended with
filtering and color-correction stages from module_2 (Lessons 3 & 4).
HW1 stages stay available; this file only adds new ones.

  - media/bing.png    -> Bing Maps (high-res, reference)
  - media/landsat.png -> Landsat   (low-res, tuning target)

"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
MEDIA = ROOT / "media"
OUTPUTS = ROOT / "outputs_task2"
OUTPUTS.mkdir(exist_ok=True)

BING_PATH = MEDIA / "bing.png"
LANDSAT_PATH = MEDIA / "landsat.png"


# --------------------------------------------------------------------------- #
# Per-image pipeline definitions.                                             #
# Each entry is (stage_name, stage_params). Available stages: see STAGES.     #
# Starting point = HW1 final params; tune with new module_2 stages from       #
# the notebook (experiments.ipynb).                                           #
# --------------------------------------------------------------------------- #
PARAMS_BING = {
    "stages": [
        ("kmeans",          {"K": 3}),
        ("to_gray",         {}),
        ("equalize",        {}),
        ("gaussian_blur",   {"ksize": 15}),
        ("canny",           {"low": 50, "high": 160}),
        ("morphology",      {"op": "close", "shape": "rect", "ksize": 8, "iters": 2}),
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
        ("canny",           {"low": 150, "high": 200}),
        ("morphology",      {"op": "close", "shape": "rect", "ksize": 8, "iters": 2}),
        ("negative",        {}),
        ("find_rectangles", {"approx_eps": 0.08, "min_area": 400, "max_area": 3000,
                             "min_vertices": 4, "max_vertices": 4}),
    ],
}


# --------------------------------------------------------------------------- #
# Stage functions. Signature: (img, params, ctx) -> (img_out, ctx_out)        #
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


# --- new in HW2 (Lesson 3: filtr_im.py) ----------------------------------- #
def s_median_blur(img, p, ctx):
    """cv2.medianBlur — removes salt-and-pepper noise, preserves edges better
    than mean / Gaussian blur."""
    return cv2.medianBlur(img, p.get("ksize", 5)), ctx


def s_bilateral(img, p, ctx):
    """cv2.bilateralFilter — edge-preserving smoothing. Smooths flat regions
    while keeping strong edges (roof / road borders) sharp."""
    return cv2.bilateralFilter(img,
                               p.get("d", 9),
                               p.get("sigma_color", 75),
                               p.get("sigma_space", 75)), ctx


def s_sharpen(img, p, ctx):
    """Sharpening kernel via cv2.filter2D. amount=1.0 -> standard sharpen,
    higher = stronger edge emphasis."""
    a = p.get("amount", 1.0)
    kernel = np.array([[0, -1, 0],
                       [-1, 4 + a, -1],
                       [0, -1, 0]], dtype=np.float32) / max(a, 1e-6)
    return cv2.filter2D(img, -1, kernel), ctx


def s_average_filter(img, p, ctx):
    """cv2.filter2D with mean kernel (Lesson 3 example #1)."""
    k = p.get("ksize", 5)
    kernel = np.ones((k, k), np.float32) / (k * k)
    return cv2.filter2D(img, -1, kernel), ctx


# --- new in HW2 (Lesson 4: brightness / contrast correction) -------------- #
def s_log_transform(img, p, ctx):
    """s = c * log(1 + r). Brightens dark regions, compresses highlights."""
    c = p.get("c", 255.0 / np.log(1 + 255))
    out = c * np.log1p(img.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8), ctx


def s_pow_transform(img, p, ctx):
    """Gamma / power-law: s = c * r^gamma. gamma<1 brightens, gamma>1 darkens."""
    gamma = p.get("gamma", 1.0)
    c = p.get("c", 1.0)
    norm = img.astype(np.float32) / 255.0
    out = c * np.power(norm, gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8), ctx


def s_hist_stretch(img, p, ctx):
    """Linear contrast stretch — remap [low_pct, high_pct] percentiles to [0,255]."""
    low_pct = p.get("low_pct", 2)
    high_pct = p.get("high_pct", 98)
    lo = np.percentile(img, low_pct)
    hi = np.percentile(img, high_pct)
    if hi <= lo:
        return img, ctx
    out = np.clip((img.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
    return out.astype(np.uint8), ctx


def s_hist_slide(img, p, ctx):
    """Histogram sliding — additive brightness shift (Lesson 4 hist_slide)."""
    shift = p.get("shift", 30)
    out = img.astype(np.int16) + shift
    return np.clip(out, 0, 255).astype(np.uint8), ctx


# --- HW1 stages, unchanged ------------------------------------------------- #
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
    # HW1
    "to_gray":         s_to_gray,
    "equalize":        s_equalize,
    "negative":        s_negative,
    "gaussian_blur":   s_gaussian_blur,
    "kmeans":          s_kmeans,
    "threshold":       s_threshold,
    "canny":           s_canny,
    "morphology":      s_morphology,
    "find_rectangles": s_find_rectangles,
    # HW2 - Lesson 3 filters
    "median_blur":     s_median_blur,
    "bilateral":       s_bilateral,
    "sharpen":         s_sharpen,
    "average_filter":  s_average_filter,
    # HW2 - Lesson 4 brightness/contrast
    "log_transform":   s_log_transform,
    "pow_transform":   s_pow_transform,
    "hist_stretch":    s_hist_stretch,
    "hist_slide":      s_hist_slide,
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
    out_dir = OUTPUTS / "stages"
    out_dir.mkdir(exist_ok=True)
    rgb_stages = {"original", "find_rectangles"}
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


def show_histogram(img, title="histogram"):
    """Lesson 4 — quick brightness histogram for diagnosing contrast."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].imshow(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 cmap="gray" if img.ndim == 2 else None)
    ax[0].set_title(title); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].hist(img.ravel(), 256, [0, 256])
    ax[1].set_title("brightness histogram")
    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# Entry point.                                                                #
# --------------------------------------------------------------------------- #
def load_images():
    bing = cv2.imread(str(BING_PATH), cv2.IMREAD_COLOR)
    landsat = cv2.imread(str(LANDSAT_PATH), cv2.IMREAD_COLOR)
    if bing is None:    raise FileNotFoundError(BING_PATH)
    if landsat is None: raise FileNotFoundError(LANDSAT_PATH)
    return bing, landsat


def main():
    bing, landsat = load_images()
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
