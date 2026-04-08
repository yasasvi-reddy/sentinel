"""
extract_patches.py

Slices Sentinel-2 imagery and UNOSAT damage labels into 256×256 patches.

Damage mask encoding:
  Kharkiv  – polygon grid,  Main_Dam_4 > 0  → damaged
  Mariupol – point features, all points buffered 150 m → damaged

Output structure:
  data/patches/
    {city}/
      pre/    – pre-war image patches   (Oct–Dec 2021)
      post/   – post-war early patches  (Mar–May 2022, matches UNOSAT dates)
      masks/  – binary damage masks     (uint8: 0=undamaged, 255=damaged)
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import geopandas as gpd
import rasterio.features
import rasterio.transform
from shapely.geometry import mapping

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
IMG_DIR = ROOT / "data" / "imagery"
SHP_DIR = ROOT / "data" / "damage_assessments"
OUT_DIR = ROOT / "data" / "patches"

PATCH_SIZE = 256
STRIDE     = 128
MIN_VALID  = 0.05   # drop patches where >95 % of image pixels are black

# ── City config ────────────────────────────────────────────────────────────────
CITIES = {
    "kharkiv": {
        "shp":        SHP_DIR / "kharkiv_north" / "Kharkiv_23March2022_RDA.shp",
        "damage_col": "Main_Dam_4",
        "damage_fn":  lambda col: col > 0,        # any damage class
        "buffer_m":   None,                        # polygons — no buffer needed
        "pre_img":    IMG_DIR / "kharkiv_prewar_oct_dec2021.png",
        "post_img":   IMG_DIR / "kharkiv_postwar_early_mar_may2022.png",
    },
    "mariupol": {
        "shp":        SHP_DIR / "mariupol" / "Damage_Point_All_No_Military_V2.shp",
        "damage_col": "Main_Damag",
        "damage_fn":  lambda col: col > 0,         # all points are damaged
        "buffer_m":   150,                         # buffer points → visible footprint
        "pre_img":    IMG_DIR / "mariupol_prewar_oct_dec2021.png",
        "post_img":   IMG_DIR / "mariupol_postwar_early_mar_may2022.png",
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_image_array(path):
    """Load PNG as float32 HxWx3 array normalised to [0,1]."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def pad_to_patch_size(arr_img, arr_mask):
    """Zero-pad image and mask so both dims are >= PATCH_SIZE."""
    h, w = arr_img.shape[:2]
    pad_h = max(0, PATCH_SIZE - h)
    pad_w = max(0, PATCH_SIZE - w)
    if pad_h or pad_w:
        arr_img  = np.pad(arr_img,  ((0, pad_h), (0, pad_w), (0, 0)))
        arr_mask = np.pad(arr_mask, ((0, pad_h), (0, pad_w)))
    return arr_img, arr_mask


def build_damage_mask(cfg, img_h, img_w):
    """
    Rasterize UNOSAT damage geometries to a binary HxW uint8 mask
    aligned to the imagery extent derived from the shapefile bounds.
    """
    gdf = gpd.read_file(cfg["shp"]).to_crs("EPSG:4326")

    if cfg["buffer_m"] is not None:
        # Buffer in metres — reproject to a metric CRS, buffer, reproject back
        gdf_m = gdf.to_crs("EPSG:32637")          # UTM zone 37N covers Ukraine
        gdf_m["geometry"] = gdf_m.buffer(cfg["buffer_m"])
        gdf = gdf_m.to_crs("EPSG:4326")

    # Select damaged features
    damaged = gdf[cfg["damage_fn"](gdf[cfg["damage_col"]])]

    # Affine transform: image spans the shapefile bounding box
    minx, miny, maxx, maxy = gdf.total_bounds
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, img_w, img_h)

    shapes = ((mapping(geom), 1) for geom in damaged.geometry if geom is not None)
    mask = rasterio.features.rasterize(
        shapes,
        out_shape=(img_h, img_w),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask


def extract_patches(img_arr, mask_arr, city, pre_or_post):
    """
    Slide a PATCH_SIZE window over the image and mask with STRIDE.
    Save patches that pass the MIN_VALID threshold.
    Returns list of patch indices saved.
    """
    H, W, _ = img_arr.shape
    saved = []
    idx = 0

    img_dir  = OUT_DIR / city / pre_or_post
    mask_dir = OUT_DIR / city / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):
            patch = img_arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            # Skip mostly-black patches (clouds masked out / no data)
            if patch.mean() < MIN_VALID:
                idx += 1
                continue

            patch_img  = Image.fromarray((patch * 255).astype(np.uint8))
            patch_img.save(img_dir / f"patch_{idx:04d}.png")

            # Save mask only once (same for pre and post)
            if pre_or_post == "pre":
                mask_patch = mask_arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE] * 255
                Image.fromarray(mask_patch.astype(np.uint8)).save(
                    mask_dir / f"patch_{idx:04d}.png"
                )

            saved.append(idx)
            idx += 1

    return saved


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for city, cfg in CITIES.items():
        print(f"\n{'─'*55}")
        print(f"City: {city.upper()}")

        pre_arr  = load_image_array(cfg["pre_img"])
        post_arr = load_image_array(cfg["post_img"])
        H, W, _  = pre_arr.shape
        print(f"  Image size: {W}×{H} px")

        print("  Rasterizing damage mask …")
        mask = build_damage_mask(cfg, H, W)   # build at original dims
        damaged_px = mask.sum()
        print(f"  Damaged pixels: {damaged_px} / {H*W}  ({100*damaged_px/(H*W):.1f}%)")

        # Pad all three arrays together so dims match
        pre_arr,  mask = pad_to_patch_size(pre_arr,  mask)
        post_arr, _    = pad_to_patch_size(post_arr, mask)

        print("  Extracting pre-war patches …")
        pre_ids  = extract_patches(pre_arr,  mask, city, "pre")
        print(f"  Extracting post-war patches …")
        post_ids = extract_patches(post_arr, mask, city, "post")

        print(f"  Patches saved — pre: {len(pre_ids)}, post: {len(post_ids)}")

    print("\nDone.")
