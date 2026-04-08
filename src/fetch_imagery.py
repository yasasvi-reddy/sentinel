"""
fetch_imagery.py

Pulls Sentinel-2 true color composite for Kharkiv, Ukraine
over a 3-month window (Jun–Aug 2023), applies pixel-level cloud masking
via the S2 QA60 band, and saves a PNG to data/.
"""

import os
import io
import requests
from PIL import Image
import ee

# ── Auth & init ────────────────────────────────────────────────────────────────
GEE_PROJECT = "project-8232277e-d8ce-4a1f-bc6"
ee.Initialize(project=GEE_PROJECT)

# ── Parameters ────────────────────────────────────────────────────────────────
# AOI: central Kharkiv, Ukraine
AOI = ee.Geometry.Rectangle([36.15, 49.95, 36.40, 50.10])

START_DATE = "2023-06-01"
END_DATE   = "2023-08-31"
CLOUD_THR  = 20          # scene-level pre-filter (%)

OUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_FILE   = os.path.join(OUT_DIR, "kharkiv_true_color_2023.png")


# ── Pixel-level cloud mask ─────────────────────────────────────────────────────
def mask_s2_clouds(image):
    """Mask clouds and cirrus using the Sentinel-2 QA60 bitmask band."""
    qa = image.select("QA60")
    cloud_bit  = 1 << 10   # bit 10: opaque clouds
    cirrus_bit = 1 << 11   # bit 11: cirrus
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(
           qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask)


# ── Build composite ───────────────────────────────────────────────────────────
s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(AOI)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_THR))
    .map(mask_s2_clouds)
    .select(["B4", "B3", "B2"])   # Red, Green, Blue
    .median()
    .clip(AOI)
)

# ── Download as thumbnail PNG ─────────────────────────────────────────────────
vis_params = {
    "min": 0,
    "max": 3000,
    "bands": ["B4", "B3", "B2"],
    "dimensions": 1024,           # longest edge in pixels
    "region": AOI,
    "format": "png",
}

url = s2.getThumbURL(vis_params)
print(f"Downloading thumbnail from Earth Engine...")
print(f"URL: {url}\n")

response = requests.get(url, timeout=120)
response.raise_for_status()

img = Image.open(io.BytesIO(response.content)).convert("RGB")

# ── Save PNG ──────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
img.save(OUT_FILE)
print(f"Saved: {OUT_FILE}")

print(f"Done. Saved to {OUT_FILE}")
