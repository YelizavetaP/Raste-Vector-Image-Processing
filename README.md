# Raste-Vector-Image-Processing

HW1 - Level 3. Count buildings on the KPI main campus using two image sources
and find a raster-processing + vectorization combo that aligns the low-res
Landsat count with the high-res Bing Maps reference.

## Sources

- Landsat: https://livingatlas2.arcgis.com/landsatexplorer/
- Bing Maps: https://www.bing.com/maps

Place the screenshots into `media/` as `landsat.png` and `bing.png`.

## Setup

```bat
py -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
python main.py
```

## Layout

- `main.py` - pipeline: load -> color correction -> vectorize -> count
- `media/` - input images
- `outputs/` - generated edge maps, contour overlays, comparison plots
