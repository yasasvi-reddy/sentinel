"""
extract_patches.py

Slices Sentinel-2 GeoTIFFs and UNOSAT damage labels into 256×256 patches.
Designed to work with full-resolution GeoTIFFs from export_geotiffs.py;
falls back to low-resolution PNGs if GeoTIFFs are not present.

Damage mask encoding:
  Kharkiv  – polygon grid,  Main_Dam_4 > 0 → damaged
  Mariupol – point features, buffered 150 m  → damaged

Output structure:
  data/patches/
    {city}/
      pre/    – pre-war image patches   (Oct–Dec 2021)
      post/   – post-war early patches  (Mar–May 2022, matches UNOSAT dates)
      masks/  – binary masks            (uint8: 0=undamaged, 255=damaged)
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import geopandas as gpd
import rasterio
import rasterio.features
import rasterio.transform
from rasterio.enums import Resampling
from shapely.geometry import mapping

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
IMG_DIR   = ROOT / "data" / "imagery"
TIFF_DIR  = ROOT / "data" / "imagery" / "geotiffs"
SHP_DIR   = ROOT / "data" / "damage_assessments"
OUT_DIR   = ROOT / "data" / "patches"

PATCH_SIZE = 256
STRIDE     = 128
MIN_VALID  = 0.05    # skip patches where mean brightness < this (mostly empty)

# ── City config ────────────────────────────────────────────────────────────────
CITIES = {
    "kharkiv": {
        "shp":        SHP_DIR / "kharkiv_north" / "Kharkiv_23March2022_RDA.shp",
        "damage_col": "Main_Dam_4",
        "damage_fn":  lambda col: col > 0,
        "buffer_m":   None,
        "pre_tif":    TIFF_DIR / "kharkiv_prewar_oct_dec2021.tif",
        "post_tif":   TIFF_DIR / "kharkiv_postwar_early_mar_may2022.tif",
        "pre_png":    IMG_DIR  / "kharkiv_prewar_oct_dec2021.png",
        "post_png":   IMG_DIR  / "kharkiv_postwar_early_mar_may2022.png",
    },
    "mariupol": {
        "shp":        SHP_DIR / "mariupol" / "Damage_Point_All_No_Military_V2.shp",
        "damage_col": "Main_Damag",
        "damage_fn":  lambda col: col > 0,
        "buffer_m":   150,
        "pre_tif":    TIFF_DIR / "mariupol_prewar_oct_dec2021.tif",
        "post_tif":   TIFF_DIR / "mariupol_postwar_early_mar_may2022.tif",
        "pre_png":    IMG_DIR  / "mariupol_prewar_oct_dec2021.png",
        "post_png":   IMG_DIR  / "mariupol_postwar_early_mar_may2022.png",
    },
}


# ── Image loading ──────────────────────────────────────────────────────────────
def load_geotiff(path: Path):
    """
    Load a GeoTIFF as a float32 HxWx3 array in [0,1].
    S2 surface reflectance is scaled 0–10000; we normalise by 3000 and clip.
    Returns (array, transform, crs).
    """
    with rasterio.open(path) as src:
        # Bands: (3, H, W) → (H, W, 3)
        data = src.read([1, 2, 3]).astype(np.float32).transpose(1, 2, 0)
        transform = src.transform
        crs       = src.crs
    data = np.nan_to_num(data, nan=0.0)   # NoData pixels → 0
    data = np.clip(data / 3000.0, 0.0, 1.0)
    return data, transform, crs


def load_png(path: Path):
    """Load a PNG as float32 HxWx3 in [0,1] with a dummy transform."""
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return arr, None, None


def load_image(tif_path: Path, png_path: Path):
    """Use GeoTIFF if available, otherwise fall back to PNG."""
    if tif_path.exists():
        print(f"  Loading GeoTIFF: {tif_path.name}")
        return load_geotiff(tif_path), "geotiff"
    print(f"  [fallback] GeoTIFF not found, using PNG: {png_path.name}")
    return load_png(png_path), "png"


# ── Damage mask ────────────────────────────────────────────────────────────────
def build_damage_mask_from_tiff(cfg: dict, transform, crs, img_h: int, img_w: int):
    """Rasterize UNOSAT features using the GeoTIFF's own CRS and transform."""
    gdf = gpd.read_file(cfg["shp"]).to_crs(crs)

    if cfg["buffer_m"] is not None:
        metric_crs       = gdf.estimate_utm_crs()
        gdf_m            = gdf.to_crs(metric_crs)
        gdf_m["geometry"] = gdf_m.buffer(cfg["buffer_m"])
        gdf              = gdf_m.to_crs(crs)

    damaged = gdf[cfg["damage_fn"](gdf[cfg["damage_col"]])]
    shapes  = ((mapping(g), 1) for g in damaged.geometry if g is not None)

    return rasterio.features.rasterize(
        shapes,
        out_shape=(img_h, img_w),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )


def build_damage_mask_from_bounds(cfg: dict, img_h: int, img_w: int):
    """Rasterize using bounds derived from the shapefile (PNG fallback path)."""
    gdf = gpd.read_file(cfg["shp"]).to_crs("EPSG:4326")

    if cfg["buffer_m"] is not None:
        metric_crs        = gdf.estimate_utm_crs()
        gdf_m             = gdf.to_crs(metric_crs)
        gdf_m["geometry"] = gdf_m.buffer(cfg["buffer_m"])
        gdf               = gdf_m.to_crs("EPSG:4326")

    damaged   = gdf[cfg["damage_fn"](gdf[cfg["damage_col"]])]
    minx, miny, maxx, maxy = gdf.total_bounds
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, img_w, img_h)
    shapes    = ((mapping(g), 1) for g in damaged.geometry if g is not None)

    return rasterio.features.rasterize(
        shapes,
        out_shape=(img_h, img_w),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )


def pad_arrays(*arrays, target_h: int, target_w: int):
    """Zero-pad arrays to at least target_h × target_w."""
    out = []
    for arr in arrays:
        h, w = arr.shape[:2]
        ph   = max(0, target_h - h)
        pw   = max(0, target_w - w)
        if ph or pw:
            if arr.ndim == 3:
                arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)))
            else:
                arr = np.pad(arr, ((0, ph), (0, pw)))
        out.append(arr)
    return out


# ── Patch extraction ───────────────────────────────────────────────────────────
def extract_patches(pre_arr, post_arr, mask_arr, city: str):
    """
    Slide PATCH_SIZE window with STRIDE over both time steps simultaneously.
    Mask saved once; pre and post images share patch IDs.
    Returns number of patches saved.
    """
    pre_dir  = OUT_DIR / city / "pre"
    post_dir = OUT_DIR / city / "post"
    mask_dir = OUT_DIR / city / "masks"
    for d in (pre_dir, post_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    H, W, _ = pre_arr.shape
    n_saved  = 0
    idx      = 0

    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):
            pre_patch = pre_arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            if pre_patch.mean() < MIN_VALID:
                idx += 1
                continue

            def to_png(arr_float):
                return Image.fromarray((arr_float * 255).astype(np.uint8))

            to_png(pre_patch).save(pre_dir / f"patch_{idx:05d}.png")
            to_png(post_arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE]).save(
                post_dir / f"patch_{idx:05d}.png"
            )
            mask_patch = mask_arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE] * 255
            Image.fromarray(mask_patch.astype(np.uint8)).save(
                mask_dir / f"patch_{idx:05d}.png"
            )

            n_saved += 1
            idx += 1

    return n_saved


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    total_patches = 0

    for city, cfg in CITIES.items():
        print(f"\n{'─'*60}")
        print(f"City: {city.upper()}")

        (pre_arr, pre_tf, pre_crs), pre_mode   = load_image(cfg["pre_tif"],  cfg["pre_png"])
        (post_arr, post_tf, _),     _           = load_image(cfg["post_tif"], cfg["post_png"])

        H, W, _ = pre_arr.shape
        print(f"  Image dimensions: {W} × {H} px  [{pre_mode}]")

        print("  Rasterizing damage mask …")
        if pre_mode == "geotiff":
            mask = build_damage_mask_from_tiff(cfg, pre_tf, pre_crs, H, W)
        else:
            mask = build_damage_mask_from_bounds(cfg, H, W)

        dmg_pct = 100 * mask.sum() / (H * W)
        print(f"  Damaged pixels: {mask.sum():,} / {H*W:,}  ({dmg_pct:.1f} %)")

        # Pad if either dim < PATCH_SIZE
        pre_arr, post_arr, mask = pad_arrays(
            pre_arr, post_arr, mask,
            target_h=PATCH_SIZE, target_w=PATCH_SIZE,
        )

        print("  Extracting patches …")
        n = extract_patches(pre_arr, post_arr, mask, city)
        print(f"  Patches saved: {n:,}")
        total_patches += n

    print(f"\n{'='*60}")
    print(f"Total patches: {total_patches:,}")
