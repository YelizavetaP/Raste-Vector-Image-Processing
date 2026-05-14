# CV Methods — Modules 1 & 2

Study notes mapping every CV approach shown in the lectures to its source file, separated by module. Path references are relative to the **lecture repo root** (`SSWU-CV/`).

---

## TL;DR — How the modules are organized

| | Module 1 | Module 2 |
|--|--|--|
| Lesson 1 name | **Raster images** | **Filtering** (Lesson 3) |
| Lesson 2 name | **Vector images** | **Brightness / Contrast** (Lesson 4) |
| Character | Broad survey + the focused stuff needed for HW1 | Focused deep-dive on pixel/intensity ops |
| Operates on | Color channels, regions, contours, frames | Single pixel intensities |
| Goal | "What is in the image" | "How the image looks pixel-by-pixel" |
| HW | HW1 — color correction → vectorization → identify | HW2 — filtering, histogram/contrast correction |

**Why M1 looks bloated:** it mixes the actual HW1 toolkit *with* teaser demos of techniques that get a full dedicated lesson in later modules (object tracking, descriptors, recognition, k-means clustering, video segmentation, 3D projection). So if you see a flashy method in M1 lectures, check the **[preview]** tag below before deciding to use it.

**HW1 explicit requirements (from `hw/hw1/task1.pdf`):**
- Color correction: grayscale, sepia, negative, *other*
- Vectorization: Contours, Canny, Gabor
- Object identification by comparing the same object across images (algorithm of your choice)
- Pipeline: *quality enhancement → vectorization → identification*

---

## Module 1 — Raster & Vector basics (HW1)

### Lesson 1 — Raster images

#### Core (the HW1 toolkit)

| Method | What it does | Source |
|--|--|--|
| Image loading (PIL / OpenCV) | Read file into pixel matrix | `module_1/Lesson 1/Image_Start/PIL_example.py` |
| Save image to JPEG | Persist processed result | `module_1/Lesson 1/Image_Start/PIL_example.py` |
| **Grayscale conversion** (RGB average / `cv2.cvtColor`) | Collapse RGB to single luminance channel | `module_1/Lesson 1/Image_Start/PIL_example.py`, `module_1/Lesson 1/Image_Processing/Im_vektor_circuit_segment.py` |
| **Sepia tone** | Warm brown shift on top of grayscale | `module_1/Lesson 1/Image_Start/PIL_example.py` |
| **Negative inversion** | `255 - channel` | `module_1/Lesson 1/Image_Start/PIL_example.py` |
| Brightness adjustment | Add constant offset, clamp 0..255 | `module_1/Lesson 1/Image_Start/PIL_example.py` |
| Additive random noise | Per-channel uniform offset (for robustness tests) | `module_1/Lesson 1/Image_Start/PIL_example.py` |
| Monochrome thresholding (binarization) | Split into pure black/white via threshold | `module_1/Lesson 1/Image_Start/PIL_example.py` |
| PIL `ImageFilter.CONTOUR` / `BLUR` / `DETAIL` | Predefined-kernel filters | `module_1/Lesson 1/Image_Start/PIL_example.py` |
| BGR ↔ RGB channel reorder | Bridge OpenCV ↔ matplotlib | `module_1/Lesson 1/Image_Processing/Im_klastering.py` |
| HSV color-space conversion | For color-based masking | `module_1/Lesson 1/Descriptors_tracking/colorspaces.py` |
| HSV in-range masking | Binary mask of pixels in a hue/sat/val range | `module_1/Lesson 1/Descriptors_tracking/colorspaces.py` |
| Median blur (`cv2.medianBlur`) | Salt-and-pepper noise removal | `module_1/Lesson 1/Descriptors_tracking/colorspaces.py` |
| Brightness histogram plot | Distribution of pixel intensities | `module_1/Lesson 1/Image_Processing/Im_quality enhanc.py` |
| matplotlib contour plot (all levels) | Draw iso-intensity curves over image | `module_1/Lesson 1/Image_Processing/Im_vektor_circuit_segment.py` |
| Single-level iso-intensity contour | Extract one bright/dark contour line | `module_1/Lesson 1/Image_Processing/Im_vektor_circuit_segment.py` |
| **Canny edge detection** | Gradient-based edges with hysteresis | `module_1/Lesson 1/Image_Processing/Im_vektor_circuit_segment.py` |

#### Preview / teaser (deep-dive elsewhere — don't reach for these in HW1)

| Method | What it does | Source | Note |
|--|--|--|--|
| **K-means color clustering** | Quantize pixel colors into K clusters | `module_1/Lesson 1/Image_Processing/Im_klastering.py`, `module_1/Lesson 1/Image_Processing/im_segment.py` | **You used this in HW1 — it's a preview; HW4 is dedicated to it.** |
| Otsu's adaptive thresholding | Auto-pick binary threshold from histogram | `module_1/Lesson 1/Image_Processing/im_segment.py` | preview |
| Morphological dilation/closing | Expand foreground / fill holes | `module_1/Lesson 1/Image_Processing/im_segment.py` | preview |
| Distance transform (Euclidean) | Pixel-to-nearest-background distance | `module_1/Lesson 1/Image_Processing/im_segment.py` | preview |
| Watershed segmentation | Marker-based flood segmentation | `module_1/Lesson 1/Image_Processing/im_segment.py` | preview |
| Connected components labelling | Unique ID per disjoint region | `module_1/Lesson 1/Image_Processing/im_segment.py` | preview |
| Roberts cross edge operator | 2×2 cross-kernel edge detection | `module_1/Lesson 1/Image_Processing/im_segment.py` | preview |
| Video capture + per-frame ops | Stream frames for analysis | `module_1/Lesson 1/Image_Processing/Im_video_segmentation.py` | preview (video) |
| Canny on streaming video | Real-time edge segmentation per frame | `module_1/Lesson 1/Image_Processing/Im_video_segmentation.py` | preview |
| 2D translation / rotation matrices | Homogeneous-matrix 2D transforms | `module_1/Lesson 1/2D_3D_proection/2D_proection.py` | preview (geometry) |
| 3D translation / rotation / orthographic / dimetric projection | Homogeneous-matrix 3D-to-2D transforms | `module_1/Lesson 1/2D_3D_proection/3D_proection.py` | preview |
| Wireframe polygon rendering | Draw projected cube faces | `module_1/Lesson 1/2D_3D_proection/3D_proection.py` | preview |
| Harris corner detection | Find corner-like keypoints | `module_1/Lesson 1/2D_3D_proection/image_descriptor.py`, `module_1/Lesson 1/Descriptors_tracking/image_descriptor.py` | preview (descriptors) |
| SIFT descriptor compute / detectAndCompute | 128-dim descriptors at keypoints | same two files | preview |
| FLANN matcher (KDTree, knnMatch) | Fast approximate descriptor matching | same two files | preview |
| Lowe's ratio test (0.7) | Filter ambiguous matches | same two files | preview |
| `drawKeypoints` / `drawMatchesKnn` | Visualize features and matches | same two files | preview |
| Background subtraction (MOG2) | Adaptive background model | `module_1/Lesson 1/Descriptors_tracking/background_subtraction.py` | preview → M3 |
| Three-frame absolute differencing | Motion detection from frame deltas | `module_1/Lesson 1/Descriptors_tracking/frame_diff.py` | preview → M3 |
| ROI selection (`cv2.selectROI`) | Pick rectangle in first frame | `module_1/Lesson 1/Descriptors_tracking/object_tracking.py` | preview |
| ROI hue histogram + back-projection | Probability map from color model | `module_1/Lesson 1/Descriptors_tracking/object_tracking.py` | preview |
| MeanShift tracking | Mode-seeking centroid follower | `module_1/Lesson 1/Descriptors_tracking/object_tracking.py` | preview → M3 |
| CamShift tracking | MeanShift with adaptive window+rotation | `module_1/Lesson 1/Descriptors_tracking/object_tracking.py` | preview → M3 |
| Mouse-driven interactive ROI | Drag-rectangle selection on stream | `module_1/Lesson 1/Descriptors_tracking/object_tracking.py` | preview |
| Histogram equalization (`cv2.equalizeHist`) | Flatten histogram for contrast | `module_1/Lesson 1/Image_Processing/Im_quality enhanc.py` | **preview → M2 L4 deep-dive** |
| CDF computation + masked-array normalization | Manual equalization pipeline | `module_1/Lesson 1/Image_Processing/Im_quality enhanc.py` | preview → M2 L4 |

### Lesson 2 — Vector images

#### Core (the HW1 toolkit)

| Method | What it does | Source |
|--|--|--|
| **Gaussian blur** (`cv2.GaussianBlur`) | Pre-step inside Canny — denoise before edge detection | `module_1/Lesson_2/Image_Recognition/image_recognition.py` |
| **Canny edge detection** | Gradient-based edges with hysteresis | `module_1/Lesson_2/Image_Recognition/image_recognition.py` |
| Morphological closing (rect kernel) | Bridge gaps between edge fragments | `module_1/Lesson_2/Image_Recognition/image_recognition.py` |
| **`cv2.findContours`** (RETR_EXTERNAL + CHAIN_APPROX_SIMPLE) | Extract contour polylines from binary edge map | `module_1/Lesson_2/Image_Recognition/image_recognition.py` |
| Contour perimeter (`cv2.arcLength`) | Measure contour length (used as ε reference) | `module_1/Lesson_2/Image_Recognition/image_recognition.py` |
| **Polygon approximation** (`cv2.approxPolyDP`, ε = 0.02·arc) | Reduce contour to few vertices | `module_1/Lesson_2/Image_Recognition/image_recognition.py` |
| Shape recognition by vertex count | Count rectangles via `len(approx)==4` | `module_1/Lesson_2/Image_Recognition/image_recognition.py` |
| `cv2.drawContours` overlay | Visualize detected shapes | `module_1/Lesson_2/Image_Recognition/image_recognition.py` |
| **Gabor kernel** (`cv2.getGaborKernel`) | Oriented sinusoidal Gaussian filter | `module_1/Lesson_2/Image_Vectorization/gabor_filter.py` |
| 2D convolution with Gabor (`cv2.filter2D`) | Extract oriented texture/edges | `module_1/Lesson_2/Image_Vectorization/gabor_filter.py` |
| PIL grayscale via `.convert('L')` | Luminance array for contour extraction | `module_1/Lesson_2/Image_Vectorization/vektor_circuit.py` |
| matplotlib contour (single level, e.g. 170) | Iso-luminance vector outline | `module_1/Lesson_2/Image_Vectorization/vektor_circuit.py` |
| Manual sum-of-RGB thresholding | Cleaner binary input before vectorization | `module_1/Lesson_2/Image_Vectorization/vektor_circuit.py` |

#### Preview / teaser

| Method | What it does | Source | Note |
|--|--|--|--|
| 1D linear / cubic interpolation (`scipy.interp1d`) | Piecewise-linear / cubic fit between samples | `module_1/Lesson_2/Image_Vectorization/approximation.py` | preview (curve math) |
| Cubic B-spline fitting (`splrep` / `splev`) | Spline through scattered data | `module_1/Lesson_2/Image_Vectorization/approximation.py` | preview |
| Weighted spline with custom knots | Non-uniform spline weighting | `module_1/Lesson_2/Image_Vectorization/approximation.py` | preview |
| Cubic Hermite spline (`BPoly.from_derivatives`) | Spline with tangent-vector constraints | `module_1/Lesson_2/Image_Vectorization/approximation.py` | preview |
| Video capture + per-frame Canny | Real-time edge vectorization | `module_1/Lesson_2/Image_Vectorization/segment_video.py` | preview (video) |
| Bresenham line rasterization | Pixel-by-pixel line discretization | `module_1/Lesson_2/Image_Vectorization_LSM_removing_edges/vectorization_LSM.py` | preview |
| **LSM (least-squares / МНК) line fitting** | `C = (FᵀF)⁻¹Fᵀy` regression line through pixels | `module_1/Lesson_2/Image_Vectorization_LSM_removing_edges/vectorization_LSM.py` | preview (advanced vectorization) |
| 3D cube construction + homogeneous transforms | Translation, X-rotation, dimetric, ortho-XY projection | `module_1/Lesson_2/Image_Vectorization_LSM_removing_edges/removing_edges.py`, `.../vectorization_LSM.py` | preview (geometry) |
| Hidden-surface removal ("floating horizon") | Face-visibility test for wireframe cube | `module_1/Lesson_2/Image_Vectorization_LSM_removing_edges/removing_edges.py` | preview |
| SIFT + FLANN + ratio test on Gabor output | Feature matching after Gabor filtering | `module_1/Lesson_2/Image_Vectorization/gabor_filter.py` | preview (descriptors) — *useful for HW1 requirement 4 (object comparison) if you go that route* |

> **Note on HW1 requirement 4 (object comparison):** the PDF says "choose your own comparison algorithm". SIFT + FLANN + Lowe's ratio test (preview material) is a legitimate choice for this part, even though descriptors get a dedicated module later. That's an exception — preview ≠ forbidden, just usually overkill.

---

## Module 2 — Pixel-level deep-dive (HW2)

Module 2 has no preview/teaser overflow — every method belongs to the lesson it sits in.

### Lesson 3 — Spatial filtering

| Method | What it does | Source |
|--|--|--|
| Averaging filter (manual 5×5 ones kernel via `cv2.filter2D`) | Mean smoothing — basic blur | `module_2/Lesson_3/filtr_im.py` |
| Box blur (`cv2.blur` 5×5) | Built-in averaging blur | `module_2/Lesson_3/filtr_im.py` |
| **Gaussian blur** (`cv2.GaussianBlur` 5×5) | Weighted averaging with Gaussian kernel | `module_2/Lesson_3/filtr_im.py` |
| **Median blur** (`cv2.medianBlur` size 5) | Replace pixel with neighbourhood median — robust to salt-and-pepper noise | `module_2/Lesson_3/filtr_im.py` |
| **Bilateral filter** (`cv2.bilateralFilter` d=9, σ=75) | Edge-preserving smoothing | `module_2/Lesson_3/filtr_im.py` |
| PIL `ImageEnhance.Sharpness` | Scalar sharpness adjustment (0.05 under, 1 neutral, 2 over) | `module_2/Lesson_3/filtr_im.py` |
| PIL pixel-manipulation utilities | Grayscale (manual avg), sepia, negative, noise, brightness, threshold | `module_2/Lesson_3/filtr_im_PIL.py` *(same demos as M1 L1 `PIL_example.py`)* |
| PIL `ImageFilter.DETAIL` (also BLUR, CONTOUR) | Predefined-kernel enhancement | `module_2/Lesson_3/filtr_im_PIL.py` |

### Lesson 4 — Brightness & contrast correction

#### Brightness histogram correction

| Method | What it does | Source |
|--|--|--|
| Brightness histogram plotting (256 bins) | Pixel-intensity distribution | `module_2/Lesson_4/brightness_histogram_correction/im quality enhanc.py` |
| CDF computation | Cumulative histogram | same |
| CDF normalization via masked min/max | Manual mapping to 0..255 | same |
| **Histogram equalization** (`cv2.equalizeHist`) | Auto contrast enhancement on grayscale | same |
| Side-by-side comparison (`np.hstack`) | Visual before/after | same |
| ROI mask construction (zeros + slice = 255) | Rectangular region of interest | `module_2/Lesson_4/brightness_histogram_correction/im_quality_enhanc_segmrnt.py` |
| Bitwise-AND masking | Isolate ROI pixels | same |
| Global vs. ROI histogram comparison (`cv2.calcHist`) | Compare full-image vs. local distributions | same |

#### Contrast enhancement

| Method | What it does | Source |
|--|--|--|
| **Classical global histogram equalization** | Histogram → PDF → CDF, map via (L−1)·CDF; YCbCr for RGB | `module_2/Lesson_4/contrast_enhancement/hist_eq.py` |
| **Histogram stretching** (min-max normalization) | Linearly remap [min,max] → [0, L−1] | `module_2/Lesson_4/contrast_enhancement/hist_stretch.py` |
| **Histogram sliding** | Add/subtract constant; brighten or darken with clamp | `module_2/Lesson_4/contrast_enhancement/hist_slide.py` |
| **BBHE** (Brightness-preserving Bi-Histogram Equalization) | Split histogram at the **mean**; equalize halves independently to preserve average brightness | `module_2/Lesson_4/contrast_enhancement/bbheq.py` |
| **DSIHE** (Dualistic Sub-Image Histogram Equalization) | Split histogram at the **CDF median** (not mean); equalize halves | `module_2/Lesson_4/contrast_enhancement/dsiheq.py` |
| **Logarithmic transform** `c·log(1+x)` | Expand dark detail, compress highlights | `module_2/Lesson_4/contrast_enhancement/log_transform.py` |
| **Power-law / gamma transform** `c·x^γ` | γ<1 brightens, γ>1 darkens | `module_2/Lesson_4/contrast_enhancement/pow_law_transform.py` |
| **Exponential transform** `c·((1+a)^x − 1)` | Compress darks, expand highlights | `module_2/Lesson_4/contrast_enhancement/exp_transform.py` |
| YCbCr-based luminance-only processing | Apply intensity transform on Y channel only, preserve color | every contrast-enhancement file above |

---

## Side-by-side: when M1 method overlaps with M2

| Concept | M1 (preview, used as a tool) | M2 (deep-dive, the topic itself) |
|--|--|--|
| Gaussian blur | One-liner pre-step inside Canny pipeline (`image_recognition.py`) | First-class filter compared against box / median / bilateral (`filtr_im.py`) |
| Median blur | Cleanup after HSV mask (`colorspaces.py`) | First-class filter (`filtr_im.py`) |
| Histogram equalization | Quick demo (`Im_quality enhanc.py`) | Whole sub-lesson with BBHE, DSIHE, stretch, slide alternatives |
| Grayscale / sepia / negative / brightness | Color-correction toolkit for HW1 (`PIL_example.py`) | Same demos repeated in `filtr_im_PIL.py` as the baseline before filtering |
| Thresholding | Binarize before contour extraction | (not extended here — Otsu/adaptive return in segmentation module) |

---
