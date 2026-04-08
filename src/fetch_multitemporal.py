"""
fetch_multitemporal.py

Fetches Sentinel-2 true color composites for Kharkiv and Mariupol
across three temporal windows:
  - Pre-war:        Oct–Dec 2021
  - Post-war early: Mar–May 2022  (matches UNOSAT label dates)
  - Post-war late:  Jun–Aug 2023

AOIs are derived from the UNOSAT shapefile bounds (EPSG:3857 → WGS84).
Cloud masking is applied per-image via the QA60 bitmask band.
Outputs: 6 PNGs saved to data/imagery/
"""

import os
import io
import requests
import geopandas as gpd
import ee

# ── Auth & init ────────────────────────────────────────────────────────────────
GEE_PROJECT = "project-8232277e-d8ce-4a1f-bc6"
ee.Initialize(project=GEE_PROJECT)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = os.path.join(os.path.dirname(__file__), "..")
SHP_DIR = os.path.join(ROOT, "data", "damage_assessments")
OUT_DIR = os.path.join(ROOT, "data", "imagery")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Cities: derive AOI from shapefile bounds ───────────────────────────────────
def bounds_to_ee_rect(shp_path):
    """Read shapefile, reproject to WGS84, return ee.Geometry.Rectangle."""
    gdf  = gpd.read_file(shp_path).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds
    return ee.Geometry.Rectangle([minx, miny, maxx, maxy])

CITIES = {
    "kharkiv": bounds_to_ee_rect(
        os.path.join(SHP_DIR, "kharkiv_north", "Kharkiv_23March2022_RDA.shp")
    ),
    "mariupol": bounds_to_ee_rect(
        os.path.join(SHP_DIR, "mariupol", "Damage_Point_All_No_Military_V2.shp")
    ),
}

# ── Time windows ───────────────────────────────────────────────────────────────
PERIODS = {
    "prewar_oct_dec2021":      ("2021-10-01", "2021-12-31"),
    "postwar_early_mar_may2022": ("2022-03-01", "2022-05-31"),
    "postwar_late_jun_aug2023":  ("2023-06-01", "2023-08-31"),
}

CLOUD_THR = 20   # scene-level pre-filter (%)


# ── Pixel-level cloud mask ─────────────────────────────────────────────────────
def mask_s2_clouds(image):
    """Mask clouds and cirrus using the Sentinel-2 QA60 bitmask band."""
    qa         = image.select("QA60")
    cloud_bit  = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(
           qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask)


# ── Fetch and save one composite ───────────────────────────────────────────────
def fetch_composite(city, period_name, aoi, start, end):
    out_path = os.path.join(OUT_DIR, f"{city}_{period_name}.png")
    if os.path.exists(out_path):
        print(f"  [skip] {os.path.basename(out_path)} already exists")
        return

    print(f"  Fetching {city} / {period_name} ({start} → {end}) …")
    composite = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_THR))
        .map(mask_s2_clouds)
        .select(["B4", "B3", "B2"])
        .median()
        .clip(aoi)
    )

    vis = {
        "min": 0,
        "max": 3000,
        "bands": ["B4", "B3", "B2"],
        "dimensions": 1024,
        "region": aoi,
        "format": "png",
    }

    url      = composite.getThumbURL(vis)
    response = requests.get(url, timeout=180)
    response.raise_for_status()

    from PIL import Image
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    img.save(out_path)
    print(f"  Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for city, aoi in CITIES.items():
        print(f"\n{'─'*50}")
        print(f"City: {city.upper()}")
        for period_name, (start, end) in PERIODS.items():
            fetch_composite(city, period_name, aoi, start, end)

    print("\nAll done.")
