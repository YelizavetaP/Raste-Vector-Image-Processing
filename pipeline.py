"""Pipeline orchestration — no stage definitions live here.

Exposes:
    STAGES               -- shared registry; populate via STAGES.update(...)
    run_pipeline         -- run a parameterised list of stages on an image
    save_stage_snapshots -- write per-stage PNGs to a directory

Callers (main.py / notebooks / main_task2.py) populate STAGES from
`stages_m1.M1_STAGES` and optionally `stages_m2.M2_STAGES`.
"""

from pathlib import Path

import cv2


# Shared registry. Callers do `STAGES.update(M1_STAGES)`, `STAGES.update(M2_STAGES)`, …
STAGES = {}


def run_pipeline(img_bgr, params, seed=None):
    """Run a parameterised stage list. Pass `seed=<int>` to reset OpenCV's RNG
    for reproducible kmeans (or other RNG-using stages); leave it `None` to
    keep the original nondeterministic behaviour."""
    if seed is not None:
        cv2.setRNGSeed(seed)
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


def save_stage_snapshots(snaps, prefix, out_dir):
    """Write every pipeline snapshot to <out_dir> as a PNG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_stages = {"original", "find_rectangles"}  # stored as RGB; flip to BGR
    for i, (name, img, _) in enumerate(snaps):
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) \
              if (img.ndim == 3 and name in rgb_stages) else img
        cv2.imwrite(str(out_dir / f"{prefix}_{i:02d}_{name}.png"), out)
