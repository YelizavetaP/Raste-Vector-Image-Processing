"""Entry point for both HW1 and HW2.

Done by Popova Yelyzaveta.

Count buildings on the KPI main campus from two sources, compare counts, and
manually tune Landsat preprocessing so its count approaches the Bing reference.
  - media/bing.png    -> Bing Maps (high-res, reference)
  - media/landsat.png -> Landsat   (low-res, tuning target)

`main()` runs the HW1 baseline pipelines first, then the HW2 candidate
pipelines (with `seed=42` for reproducibility), and finishes with a side-by-side
contour overlay so the two runs can be compared visually.

Set `SAVE_DIR_HW1` / `SAVE_DIR_HW2` near the top to a path to write all of that
task's artifacts there, or leave as `None` to skip saving.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from pipeline import STAGES, run_pipeline, save_stage_snapshots
from stages_m1 import M1_STAGES
from stages_m2 import M2_STAGES
from viz import (
    show_originals, show_pipeline_grid, show_contours_side_by_side,
    show_contours_overlay,
)


ROOT = Path(__file__).parent
MEDIA = ROOT / "media"
OUTPUTS = ROOT / "outputs"

BING_PATH = MEDIA / "bing.png"
LANDSAT_PATH = MEDIA / "landsat.png"

# LANDSAT_PATH = MEDIA / "ls2.png"
# LANDSAT_PATH = MEDIA / "ls_2pm.png"


# --------------------------------------------------------------------------- #
# Save directories — None = don't save anything for that task.                #
# Set to a Path / string to write all of that task's artifacts under that dir.#
# --------------------------------------------------------------------------- #
SAVE_DIR_HW1 = 'outputs/hw1'   # None to skip
SAVE_DIR_HW2 = 'outputs/hw2'   # None to skip


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
# HW2 — final candidate pipelines (copied from task2.ipynb).                  #
# --------------------------------------------------------------------------- #
PARAMS_BING_HW2 = {
    "stages": [
        ("warm_cool_mask",  {"threshold": 140}),
        ("bilateral",       {"d": 9, "sigma_color": 75, "sigma_space": 75}),
        ("to_gray",         {}),
        ("equalize",        {}),
        ("gaussian_blur",   {"ksize": 15}),
        ("canny",           {"low": 50, "high": 180}),
        ("morphology",      {"op": "close", "shape": "rect", "ksize": 8, "iters": 2}),
        ("negative",        {}),
        ("find_rectangles", {"approx_eps": 0.08, "min_area": 400, "max_area": 3000,
                             "min_vertices": 4, "max_vertices": 6, "only_convex": True}),
    ],
}

PARAMS_LANDSAT_HW2 = {
    "stages": [
        ("pow_transform",   {"gamma": 1.5}),
        ("hist_stretch",    {"low_pct": 5, "high_pct": 95}),
        ("pil_filter",      {"mode": "EDGE_ENHANCE"}),
        ("kmeans",          {"K": 4}),
        ("to_gray",         {}),
        ("equalize",        {}),
        ("canny",           {"low": 80, "high": 150}),
        ("morphology",      {"op": "close", "shape": "rect", "ksize": 9, "iters": 3}),
        ("negative",        {}),
        ("find_rectangles", {"approx_eps": 0.08, "min_area": 700, "max_area": 5000,
                             "min_vertices": 4, "max_vertices": 20, "only_convex": False}),
    ],
}

HW2_SEED = 42   # passed to run_pipeline so kmeans is reproducible


# Register every stage once, before any pipeline runs.
STAGES.update(M1_STAGES)
STAGES.update(M2_STAGES)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _resolve_save_dir(save_dir):
    """str/Path/None -> absolute Path or None. Relative paths resolve against ROOT."""
    if not save_dir:
        return None
    save_dir = Path(save_dir)
    if not save_dir.is_absolute():
        save_dir = ROOT / save_dir
    return save_dir


def _path(dir_, filename):
    return (dir_ / filename) if dir_ else None


def load_images():
    bing = cv2.imread(str(BING_PATH), cv2.IMREAD_COLOR)
    landsat = cv2.imread(str(LANDSAT_PATH), cv2.IMREAD_COLOR)
    if bing is None:    raise FileNotFoundError(BING_PATH)
    if landsat is None: raise FileNotFoundError(LANDSAT_PATH)
    return bing, landsat


def run_task(bing, landsat, params_bing, params_landsat, save_dir, label,
             seed=None):
    """Run one task (HW1 or HW2) on both images, show + optionally save."""
    save_dir = _resolve_save_dir(save_dir)

    show_originals(bing, landsat, save_path=_path(save_dir, "01_originals.png"))

    snaps_b, ctx_b = run_pipeline(bing,    params_bing,    seed=seed)
    snaps_l, ctx_l = run_pipeline(landsat, params_landsat, seed=seed)

    if save_dir:
        save_stage_snapshots(snaps_b, "bing",    out_dir=save_dir / "stages")
        save_stage_snapshots(snaps_l, "landsat", out_dir=save_dir / "stages")

    show_pipeline_grid(snaps_b, snaps_l, ctx_b["count"], ctx_l["count"],
                       f"Bing ({label})", f"Landsat ({label})",
                       save_path=_path(save_dir, "02_pipeline.png"))
    show_contours_side_by_side(ctx_b, ctx_l,
                               save_path=_path(save_dir, "03_contours.png"))
    return ctx_b, ctx_l


def show_overlay_pair(ctx_b_a, ctx_b_b, ctx_l_a, ctx_l_b,
                      label_a="HW1", label_b="HW2", save_path=None):
    """Side-by-side: Bing on the left, Landsat on the right. Each panel shows
    contours from both pipelines overlaid in different colors."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    show_contours_overlay(ctx_b_a, ctx_b_b, label_a, label_b,
                          ax=axes[0], title="Bing")
    show_contours_overlay(ctx_l_a, ctx_l_b, label_a, label_b,
                          ax=axes[1], title="Landsat")
    fig.suptitle(f"Contour overlay — {label_a} vs {label_b}")
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    plt.show()


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def main():
    bing, landsat = load_images()

    # ----- HW1 baseline -----
    ctx_b_hw1, ctx_l_hw1 = run_task(
        bing, landsat,
        PARAMS_BING_HW1, PARAMS_LANDSAT_HW1,
        save_dir=SAVE_DIR_HW1, label="HW1",
    )

    # ----- HW2 candidate (seeded so kmeans is reproducible) -----
    ctx_b_hw2, ctx_l_hw2 = run_task(
        bing, landsat,
        PARAMS_BING_HW2, PARAMS_LANDSAT_HW2,
        save_dir=SAVE_DIR_HW2, label="HW2",
        seed=HW2_SEED,
    )

    # ----- Final side-by-side overlay: HW1 vs HW2 on each image -----
    hw2_dir = _resolve_save_dir(SAVE_DIR_HW2)
    show_overlay_pair(ctx_b_hw1, ctx_b_hw2,
                      ctx_l_hw1, ctx_l_hw2,
                      label_a="HW1", label_b="HW2",
                      save_path=_path(hw2_dir, "04_overlay.png"))

    print(f"\nHW1   bing={ctx_b_hw1['count']}   landsat={ctx_l_hw1['count']}")
    print(f"HW2   bing={ctx_b_hw2['count']}   landsat={ctx_l_hw2['count']}")
    print(f"delta bing={ctx_b_hw2['count'] - ctx_b_hw1['count']:+d}   "
          f"landsat={ctx_l_hw2['count'] - ctx_l_hw1['count']:+d}")


if __name__ == "__main__":
    main()
