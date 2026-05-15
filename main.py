"""Entry point for both HW1 and HW2.

Done by Popova Yelyzaveta.

Count buildings on the KPI main campus from two sources, compare counts, and
manually tune Landsat preprocessing so its count approaches the Bing reference.
  - media/bing.png    -> Bing Maps (high-res, reference)
  - media/landsat.png -> Landsat   (low-res, tuning target)

To switch tasks: edit the three lines inside `main()` to point at the HW1 or
HW2 params/save_dir. Set the corresponding SAVE_DIR_* to `None` for no save,
or to a directory path to write all artifacts there.
"""

from pathlib import Path

import cv2

from pipeline import STAGES, run_pipeline, save_stage_snapshots
from stages_m1 import M1_STAGES
from stages_m2 import M2_STAGES
from viz import show_originals, show_pipeline_grid, show_contours_side_by_side

# Make every stage from both modules available to the pipeline
STAGES.update(M1_STAGES)
STAGES.update(M2_STAGES)


ROOT = Path(__file__).parent
MEDIA = ROOT / "media"
OUTPUTS = ROOT / "outputs"

BING_PATH = MEDIA / "bing.png"
LANDSAT_PATH = MEDIA / "landsat.png"

# LANDSAT_PATH = MEDIA / "ls2.png"
# LANDSAT_PATH = MEDIA / "ls_2pm.png"


# --------------------------------------------------------------------------- #
# Save directories — None = don't save anything for that task.                #
# Set to a Path to write all of that task's artifacts under that directory.   #
# --------------------------------------------------------------------------- #
SAVE_DIR_HW1 = 'outputs/hw1'   # e.g. OUTPUTS / "hw1"
SAVE_DIR_HW2 = 'outputs/hw2'    # e.g. OUTPUTS / "hw2"


# --------------------------------------------------------------------------- #
# HW1 — raster & vector basics (color correction + Canny + contours).        #
# --------------------------------------------------------------------------- #
PARAMS_BING_HW1 = {
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

PARAMS_LANDSAT_HW1 = {
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
# HW2 — adds filtering + brightness/contrast correction stages from M2.      #
# Starting point is the latest CANDIDATE_* from task2.ipynb; tune further    #
# in the notebook then paste the winner back here.                            #
# --------------------------------------------------------------------------- #
PARAMS_BING_HW2 = {
    "stages": [
        ("kmeans",          {"K": 3}),
        ("bilateral",       {"d": 9, "sigma_color": 75, "sigma_space": 75}),
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

PARAMS_LANDSAT_HW2 = {
    "stages": [
        ("kmeans",          {"K": 9}),
        ("to_gray",         {}),
        ("pow_transform",   {"gamma": 0.7}),
        ("hist_stretch",    {"low_pct": 2, "high_pct": 98}),
        ("median_blur",     {"ksize": 3}),
        ("canny",           {"low": 150, "high": 200}),
        ("morphology",      {"op": "close", "shape": "rect", "ksize": 8, "iters": 2}),
        ("negative",        {}),
        ("find_rectangles", {"approx_eps": 0.08, "min_area": 400, "max_area": 3000,
                             "min_vertices": 4, "max_vertices": 4}),
    ],
}


# --------------------------------------------------------------------------- #
# Entry point.                                                                #
# --------------------------------------------------------------------------- #
def main():
    bing = cv2.imread(str(BING_PATH), cv2.IMREAD_COLOR)
    landsat = cv2.imread(str(LANDSAT_PATH), cv2.IMREAD_COLOR)
    if bing is None:    raise FileNotFoundError(BING_PATH)
    if landsat is None: raise FileNotFoundError(LANDSAT_PATH)

    # # ===== switch task here =====
    # params_bing    = PARAMS_BING_HW1     # PARAMS_BING_HW2 to run HW2
    # params_landsat = PARAMS_LANDSAT_HW1  # PARAMS_LANDSAT_HW2 to run HW2
    # save_dir       = SAVE_DIR_HW1        # SAVE_DIR_HW2 to run HW2

    params_bing    = PARAMS_BING_HW2     
    params_landsat = PARAMS_LANDSAT_HW2  
    save_dir       = SAVE_DIR_HW2        # SAVE_DIR_HW2 to run HW2
    # ============================

    # accept str or Path; relative paths resolve against the script's own dir
    save_dir = Path(save_dir) if save_dir else None
    if save_dir and not save_dir.is_absolute():
        save_dir = ROOT / save_dir

    show_originals(bing, landsat,
                   save_path=(save_dir / "01_originals.png") if save_dir else None)

    snaps_b, ctx_b = run_pipeline(bing,    params_bing)
    snaps_l, ctx_l = run_pipeline(landsat, params_landsat)

    if save_dir:
        save_stage_snapshots(snaps_b, "bing",    out_dir=save_dir / "stages")
        save_stage_snapshots(snaps_l, "landsat", out_dir=save_dir / "stages")

    show_pipeline_grid(snaps_b, snaps_l, ctx_b["count"], ctx_l["count"],
                       "Bing (reference)", "Landsat (tuned)",
                       save_path=(save_dir / "02_pipeline.png") if save_dir else None)
    show_contours_side_by_side(ctx_b, ctx_l,
                               save_path=(save_dir / "03_contours.png") if save_dir else None)

    print(f"\nBing reference: {ctx_b['count']} buildings")
    print(f"Landsat:        {ctx_l['count']} buildings")
    print(f"Delta:          {ctx_l['count'] - ctx_b['count']:+d}")


if __name__ == "__main__":
    main()


# Notes - results & conclusions
#
# Generated plots and full write-up: see README.md and the outputs/ folder
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
# --- Bing (HW1 high-res reference) ---
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

# --- Landsat (HW1 low-res tuning target) ---
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
# entirely. Same find_rectangles params used as Bing.
