# HW2 — Experiment Plan

Reference catalog of methods: see [`METHODS.md`](./METHODS.md).
Experiments live in [`task2.ipynb`](./task2.ipynb); final winning script in [`main_task2.py`](./main_task2.py).

## Goal (Level 3)

Continue with the HW1 KPI campus image. Find a combination of **filtering** + **color/histogram correction** + the existing HW1 vectorization pipeline such that the building count from the **Landsat** image matches the count from **Bing Maps** (treated as ground truth).

Also keep HW1's requirement 4: features must remain usable for cross-image identification.

## Phases

### §2 — Baseline (done)
`run_pipeline(image, PARAMS_*)` with the HW1 stage list, on `bing` and `landsat`. Record `BASELINE_BING` and `BASELINE_LANDSAT`.

### §3 — Register every Module 2 stage
Add **all** Module 2 methods to the same `STAGES` registry used by `run_pipeline`, so any of them can be plugged into a pipeline by name.

**Filters (Lesson 3):**
- `average_filter` — `cv2.filter2D` with normalized ones kernel
- `box_blur` — `cv2.blur` (built-in averaging)
- `gaussian_blur` — already in HW1 STAGES
- `median_blur` — `cv2.medianBlur`
- `bilateral` — `cv2.bilateralFilter` (edge-preserving)
- `sharpen` — manual cross-kernel sharpening
- `pil_sharpen` — `PIL.ImageEnhance.Sharpness`
- `pil_filter` — `PIL.ImageFilter` predefined modes (`BLUR`, `CONTOUR`, `DETAIL`, `EDGE_ENHANCE`, `SHARPEN`, `SMOOTH`)

**Brightness / contrast (Lesson 4):**
- `equalize` — already in HW1 STAGES (classical `cv2.equalizeHist`)
- `hist_stretch` — min-max / percentile stretch
- `hist_slide` — additive brightness shift
- `bbhe` — Brightness-preserving Bi-Histogram Equalization (split at mean)
- `dsihe` — Dualistic Sub-Image Histogram Equalization (split at CDF median)
- `log_transform` — `s = c · log(1 + r)`
- `pow_transform` — gamma `s = c · r^γ`
- `exp_transform` — `s = c · (b^r − 1)`

**Color extras (Lesson 3 — PIL):**
- `sepia` — classic matrix sepia tone

### §4 — Per-stage exploration on both images (qualitative)
For every stage above, run `try_stage_both(name, params)` to see how it behaves on **bing** and **landsat** side by side. No pipeline, no count yet — just intuition for which transforms are worth combining later.

Helpers:
- `plot_two_images(img_a, img_b, ...)` — reusable side-by-side plotter (handles 1-ch and 3-ch BGR).
- `try_stage_both(stage_name, params)` — apply stage to both images, then call `plot_two_images`.

Sweeps grouped into three batch cells:
- **§4a Filters** — 8 calls
- **§4b Brightness / contrast** — 8 calls
- **§4c Color & PIL extras** — sepia + every `pil_filter` mode

### §5 — Single-stage swap inside the HW1 pipeline (quantitative)
For each promising stage from §4, either prepend it to the HW1 pipeline or substitute the analogous HW1 stage. Run `run_pipeline`, record the building count for bing and landsat, compute delta vs baseline.

### §6 — Combine winners
Pick the top 2–3 filters and top 2–3 corrections from §5. Try every pair in **both orders** (filter→correction and correction→filter — order matters with non-linear ops).

### §7 — Tune the winner
Sweep parameters of the winning combination (kernel sizes, γ, σ, percentiles, …) to minimize `|count_landsat − count_bing|`.

### §8 — Finalize
Copy the winning per-image pipelines (`CANDIDATE_BING`, `CANDIDATE_LANDSAT`) into `main_task2.py`. Keep `task2.ipynb` as the experiment log.

## Tracking table format

| ID | Stage / pipeline | Filter | Correction | Order | Count bing | Count landsat | Δ vs baseline | Notes |
|--|--|--|--|--|--|--|--|--|
| E0 | baseline (HW1)   | — | — | — | _Nb_ | _Nl_ | 0 | from §2 |
| E1 | + median_blur    | median | — | filter only | | | | §5 |
| E2 | + hist_stretch   | — | stretch | correction only | | | | §5 |
| E12| median + stretch | median | stretch | F→C | | | | §6 |
| ... | | | | | | | | |

## Success criteria
1. **Counting accuracy** — `count_landsat` within X% of `count_bing` for the same KPI region (define X, e.g. 10%).
2. **Identification stability** — same preprocessing pipeline still produces usable SIFT / FLANN matches against a second KPI image (HW1 req 4, re-asked in HW2 Group 1).
