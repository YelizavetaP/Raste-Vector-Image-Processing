"""HW1 - Raster & Vector Image Processing (Level 3).

Count buildings on the KPI main campus from two sources:
  - Landsat (low-res)  -> media/landsat.png
  - Bing Maps (high-res, treated as ground truth) -> media/bing.png

Compare counts and search for a preprocessing + vectorization combination
that brings the Landsat count close to the Bing reference count.
"""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
MEDIA = ROOT / "media"
OUTPUTS = ROOT / "outputs"

LANDSAT_PATH = MEDIA / "landsat.png"
BING_PATH = MEDIA / "bing.png"


def load(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def preprocess(gray: np.ndarray) -> np.ndarray:
    # placeholder: blur to suppress noise before edge detection
    return cv2.GaussianBlur(gray, (5, 5), 0)


def vectorize_canny(gray: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    return cv2.Canny(gray, low, high)


def find_contours(edges: np.ndarray):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def count_buildings(contours, min_area: float = 50.0, max_area: float = 1e6) -> int:
    return sum(1 for c in contours if min_area <= cv2.contourArea(c) <= max_area)


def draw_overlay(img: np.ndarray, contours) -> np.ndarray:
    out = img.copy()
    cv2.drawContours(out, contours, -1, (0, 255, 0), 1)
    return out


def process(name: str, path: Path) -> int:
    img = load(path)
    gray = to_gray(img)
    pre = preprocess(gray)
    edges = vectorize_canny(pre)
    contours = find_contours(edges)
    n = count_buildings(contours)

    cv2.imwrite(str(OUTPUTS / f"{name}_edges.png"), edges)
    cv2.imwrite(str(OUTPUTS / f"{name}_contours.png"), draw_overlay(img, contours))
    print(f"{name}: {n} candidate buildings")
    return n


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    bing_n = process("bing", BING_PATH)
    landsat_n = process("landsat", LANDSAT_PATH)
    print(f"\nReference (Bing): {bing_n}")
    print(f"Landsat:          {landsat_n}")
    print(f"Delta:            {landsat_n - bing_n:+d}")


if __name__ == "__main__":
    main()
