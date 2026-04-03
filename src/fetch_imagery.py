"""
fetch_imagery.py

Pulls Sentinel-2 true color composite for a test AOI in Ukraine
over a 3-month window (Jun–Aug 2023) and saves a PNG to data/.
"""

import os
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import ee

# ── Auth & init ────────────────────────────────────────────────────────────────
# Credentials already saved via `earthengine authenticate` CLI
ee.Initialize()

# ── Parameters ────────────────────────────────────────────────────────────────
# Test AOI: central Kharkiv, Ukraine
AOI = ee.Geometry.Rectangle([36.15, 49.95, 36.40, 50.10])

START_DATE = "2023-06-01"
END_DATE   = "2023-08-31"
CLOUD_THR  = 20          # max cloud cover %

OUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_FILE   = os.path.join(OUT_DIR, "kharkiv_true_color_2023.png")

# ── Build composite ───────────────────────────────────────────────────────────
s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(AOI)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_THR))
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

# ── Display ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(np.array(img))
ax.set_title(
    f"Sentinel-2 True Color — Kharkiv, Ukraine\n{START_DATE} to {END_DATE}",
    fontsize=13,
)
ax.axis("off")
plt.tight_layout()
plt.show()
