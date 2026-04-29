"""
api.py

FastAPI backend for the War Damage Detection System.

Endpoints
---------
GET  /health   — liveness check
POST /analyze  — full pipeline: GEE fetch → U-Net inference → ViT classification

Pipeline (per request)
----------------------
1. Build GEE Sentinel-2 composites: pre-war baseline (1 yr before start) + post-war window
2. Download as in-memory GeoTIFFs (falls back to thumbnails)
3. Slide 256×256 patches; run U-Net → pre_prob / post_prob maps
4. Run ViT → per-pixel class (0=undamaged, 1=newly damaged, 2=pre-existing)
5. Vectorise damage blobs → lat/lng polygons with label + confidence
6. Aggregate metrics and monthly temporal progression
7. Encode pre-war, post-war RGB composites and segmentation mask as base64 PNGs

Run with:
    uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000 --loop asyncio
"""

import base64
import hashlib
import io
import json
import sys
import zipfile
import calendar
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio
import rasterio.features
import rasterio.transform
from affine import Affine
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

try:
    import ee
    _EE_AVAILABLE = True
except ImportError:
    _EE_AVAILABLE = False

try:
    import torch
    import segmentation_models_pytorch as smp
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

warnings.filterwarnings("ignore")

# ── Paths & constants ─────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
MODEL_DIR  = ROOT / "models"
UNET_CKPT  = MODEL_DIR / "unet_resnet34_best.pth"
VIT_CKPT   = MODEL_DIR / "temporal_vit_best.pth"

GEE_PROJECT  = "project-8232277e-d8ce-4a1f-bc6"
CLOUD_THR    = 20
PATCH_SIZE   = 256
INFER_STRIDE = 64      # sliding-window step; smaller = smoother blending
DAMAGE_THR   = 0.35    # U-Net post_prob threshold — below this → undamaged
MIN_VALID    = 0.05

# Pre-downloaded imagery fallback
IMAGERY_DIR  = ROOT / "data" / "imagery" / "geotiffs"
CACHE_DIR    = ROOT / "data" / "cache"
CITIES = {
    "kharkiv":  (49.9935, 36.2304),   # lat, lng
    "mariupol": (47.0966, 37.5416),
}

# Thumbnail size for base64 images sent to frontend
THUMB_SIZE  = 512

# Mask color coding (RGBA)
MASK_COLORS = {
    1: (226, 75,  74,  180),   # newly damaged  → red
    2: (230, 160, 70,  180),   # pre-existing   → orange
}

# ── Lazy globals ──────────────────────────────────────────────────────────────
GEE_OK = False
DEVICE = None
UNET   = None
VIT    = None


def _get_device():
    if not _TORCH_AVAILABLE:
        return None
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


def _load_unet(device):
    import segmentation_models_pytorch as smp
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None,
        in_channels=6, classes=1, activation=None,
    ).to(device)
    if UNET_CKPT.exists():
        ckpt = torch.load(UNET_CKPT, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[api] U-Net loaded from {UNET_CKPT.name}", flush=True)
    else:
        print(f"[warn] U-Net checkpoint not found: {UNET_CKPT}", file=sys.stderr, flush=True)
    model.eval()
    return model


def _load_vit(device):
    sys.path.insert(0, str(ROOT / "src"))
    from temporal_vit import TemporalViT
    model = TemporalViT(
        img_size=256, patch_size=16, in_channels=2,
        num_classes=3, embed_dim=192, depth=4,
        num_heads=6, mlp_ratio=4.0, dropout=0.1,
    ).to(device)
    if VIT_CKPT.exists():
        ckpt = torch.load(VIT_CKPT, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[api] TemporalViT loaded from {VIT_CKPT.name}", flush=True)
    else:
        print(f"[warn] ViT checkpoint not found: {VIT_CKPT}", file=sys.stderr, flush=True)
    model.eval()
    return model


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global GEE_OK, DEVICE, UNET, VIT
    print("[api] Starting up …", flush=True)

    if _TORCH_AVAILABLE:
        DEVICE = _get_device()
        print(f"[api] Device: {DEVICE}", flush=True)
        UNET = _load_unet(DEVICE)
        VIT  = _load_vit(DEVICE)
    else:
        print("[warn] torch not available — ML pipeline disabled (local imagery only)", file=sys.stderr, flush=True)

    if _EE_AVAILABLE:
        try:
            ee.Initialize(project=GEE_PROJECT)
            GEE_OK = True
            print("[api] GEE initialised", flush=True)
        except Exception as e:
            print(f"[warn] GEE init failed: {e}", file=sys.stderr, flush=True)
    else:
        print("[warn] earthengine-api not installed — GEE disabled", file=sys.stderr, flush=True)

    print("[api] Ready.", flush=True)
    yield
    print("[api] Shutting down.", flush=True)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="War Damage Detection API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Schemas ───────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    location: str
    start_date: str
    end_date: str
    infrastructure_type: Optional[str] = "all"

class DamageZone(BaseModel):
    polygon: List[List[float]]
    damage_class: int
    label: str
    confidence: float

class TemporalPoint(BaseModel):
    date: str
    damage_count: int

class AnalyzeResponse(BaseModel):
    damage_zones: List[DamageZone]
    metrics: dict
    temporal_progression: List[TemporalPoint]
    pre_image_b64: Optional[str] = None   # base64 PNG of pre-war satellite composite
    post_image_b64: Optional[str] = None  # base64 PNG of post-war satellite composite
    mask_b64: Optional[str] = None        # base64 RGBA PNG of segmentation mask


# ── Image encoding helpers ────────────────────────────────────────────────────
def _arr_to_b64_png(arr: np.ndarray) -> str:
    """Convert float32 RGB array [0,1] shape (H,W,3) to base64 PNG string."""
    uint8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    img = Image.fromarray(uint8, "RGB")
    # Resize to thumbnail so the JSON payload stays manageable
    if max(img.size) > THUMB_SIZE:
        img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mask_to_b64_png(label_map: np.ndarray) -> str:
    """Convert uint8 label map (0/1/2) to color-coded RGBA PNG."""
    H, W = label_map.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for cls, color in MASK_COLORS.items():
        rgba[label_map == cls] = color
    img = Image.fromarray(rgba, "RGBA")
    if max(img.size) > THUMB_SIZE:
        img = img.resize(
            (THUMB_SIZE, int(THUMB_SIZE * H / W)) if W >= H else (int(THUMB_SIZE * W / H), THUMB_SIZE),
            Image.NEAREST
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── Local GeoTIFF fallback helpers ────────────────────────────────────────────
def _city_from_coords(lat: float, lng: float) -> Optional[str]:
    """Return the nearest known city name if within ~1.5 degrees, else None."""
    best, best_dist = None, float("inf")
    for name, (clat, clng) in CITIES.items():
        dist = ((lat - clat) ** 2 + (lng - clng) ** 2) ** 0.5
        if dist < best_dist:
            best, best_dist = name, dist
    return best if best_dist < 1.5 else None


def _pick_local_tiffs(city: str, end_date: str) -> Optional[tuple]:
    """Return (pre_path, post_path) for the given city and analysis end date."""
    pre_path = IMAGERY_DIR / f"{city}_prewar_oct_dec2021.tif"
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    if end_dt <= datetime(2022, 8, 31):
        post_path = IMAGERY_DIR / f"{city}_postwar_early_mar_may2022.tif"
    else:
        post_path = IMAGERY_DIR / f"{city}_postwar_late_jun_aug2023.tif"
    if pre_path.exists() and post_path.exists():
        return pre_path, post_path
    return None


def _load_local_tiff(path: Path) -> tuple:
    """Load a GeoTIFF and return (rgb_float32 [H,W,3], Affine transform, crs str)."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        tf   = src.transform
        crs  = src.crs.to_string() if src.crs else "EPSG:4326"
    # Select first 3 bands; GEE exports are B4/B3/B2 scaled [0, ~3000+]
    rgb = data[:3].transpose(1, 2, 0) if data.shape[0] >= 3 else np.stack([data[0]] * 3, -1)
    rgb = np.clip(np.nan_to_num(rgb) / 3000.0, 0.0, 1.0)
    return rgb, tf, crs


def _load_local_imagery(lat: float, lng: float, end_date: str) -> Optional[tuple]:
    """Return (pre_arr, pre_tf, post_arr, post_tf) from local GeoTIFFs, or None."""
    city = _city_from_coords(lat, lng)
    if city is None:
        return None
    tiffs = _pick_local_tiffs(city, end_date)
    if tiffs is None:
        return None
    pre_path, post_path = tiffs
    print(f"[api] GEE offline — using local {city} GeoTIFFs", flush=True)
    pre_arr,  pre_tf,  _ = _load_local_tiff(pre_path)
    post_arr, post_tf, _ = _load_local_tiff(post_path)
    return pre_arr, pre_tf, post_arr, post_tf


# ── GEE helpers ───────────────────────────────────────────────────────────────
def _mask_s2_clouds(image):
    qa   = image.select("QA60")
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask)


def _s2_composite(aoi, start: str, end: str):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_THR))
        .map(_mask_s2_clouds)
        .select(["B4", "B3", "B2"])
        .median()
        .clip(aoi)
    )


def _download_as_array(image, aoi, scale: int = 10) -> tuple:
    import requests  # local import — only used when GEE is available
    try:
        url = image.getDownloadURL({
            "bands": ["B4", "B3", "B2"], "region": aoi,
            "scale": scale, "format": "GEO_TIFF", "crs": "EPSG:4326",
        })
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()
        buf = io.BytesIO(resp.content)
        if resp.content[:2] == b"PK":
            with zipfile.ZipFile(buf) as zf:
                tif_names = [n for n in zf.namelist() if n.endswith(".tif")]
                buf = io.BytesIO(zf.read(tif_names[0]))
        with rasterio.open(buf) as src:
            data = src.read().astype(np.float32)
            tf, crs = src.transform, (src.crs.to_string() if src.crs else "EPSG:4326")
        rgb = data[:3].transpose(1, 2, 0) if data.shape[0] >= 3 else np.stack([data[0]] * 3, -1)
        rgb = np.clip(np.nan_to_num(rgb) / 3000.0, 0.0, 1.0)
        return rgb, tf, crs
    except Exception as e:
        print(f"[warn] GeoTIFF download failed ({e}), using thumbnail", file=sys.stderr, flush=True)
        return _download_thumbnail(image, aoi)


def _download_thumbnail(image, aoi) -> tuple:
    import requests  # local import — only used when GEE is available
    url = image.getThumbURL({
        "min": 0, "max": 3000, "bands": ["B4", "B3", "B2"],
        "dimensions": 512, "region": aoi, "format": "png",
    })
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    arr = np.array(Image.open(io.BytesIO(resp.content)).convert("RGB"), dtype=np.float32) / 255.0
    coords = aoi.bounds().getInfo()["coordinates"][0]
    lngs = [c[0] for c in coords]; lats = [c[1] for c in coords]
    tf = rasterio.transform.from_bounds(min(lngs), min(lats), max(lngs), max(lats),
                                        arr.shape[1], arr.shape[0])
    return arr, tf, "EPSG:4326"


# ── Inference helpers ─────────────────────────────────────────────────────────
def _rgb_to_tensor(arr: np.ndarray):
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)


def _unet_prob(pre_arr: np.ndarray, post_arr: np.ndarray) -> tuple:
    with torch.no_grad():
        pre_t  = _rgb_to_tensor(pre_arr)
        post_t = _rgb_to_tensor(post_arr)
        pre_prob  = torch.sigmoid(UNET(torch.cat([pre_t,  pre_t],  1))).squeeze().cpu().numpy()
        post_prob = torch.sigmoid(UNET(torch.cat([pre_t,  post_t], 1))).squeeze().cpu().numpy()
    return pre_prob.astype(np.float32), post_prob.astype(np.float32)


def _vit_classify(pre_prob: np.ndarray, post_prob: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.stack([torch.from_numpy(pre_prob), torch.from_numpy(post_prob)],
                        0).unsqueeze(0).float().to(DEVICE)
        return VIT(x).argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)


def _pad_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    ph = max(0, h - arr.shape[0]); pw = max(0, w - arr.shape[1])
    return np.pad(arr, ((0, ph), (0, pw), (0, 0))) if arr.ndim == 3 else np.pad(arr, ((0, ph), (0, pw)))


def _run_pipeline(pre_arr: np.ndarray, post_arr: np.ndarray) -> np.ndarray:
    """
    Two-stage sliding-window inference with Hanning-weighted blending.

    Stage 1 — U-Net:  tile the full image with overlap → accumulate pre_prob
                       and post_prob at full resolution using a Hanning window.
    Stage 2 — ViT:    tile the stitched probability maps → accumulate per-pixel
                       class scores → gate by post_prob > DAMAGE_THR.
    """
    H, W = pre_arr.shape[:2]
    if UNET is None or VIT is None:
        print("[warn] Models not loaded; returning empty label map", file=sys.stderr, flush=True)
        return np.zeros((H, W), dtype=np.uint8)

    # Pad so at least one full patch fits and sliding window reaches the edges
    pH = max(H, PATCH_SIZE)
    pW = max(W, PATCH_SIZE)
    pre_arr  = _pad_to(pre_arr,  pH, pW)
    post_arr = _pad_to(post_arr, pH, pW)

    # 2-D Hanning window for smooth blending at patch boundaries
    hanning = np.outer(
        np.hanning(PATCH_SIZE + 2)[1:-1],
        np.hanning(PATCH_SIZE + 2)[1:-1],
    ).astype(np.float32)

    # Ensure the last patch column/row is always covered
    ys = list(range(0, pH - PATCH_SIZE, INFER_STRIDE))
    xs = list(range(0, pW - PATCH_SIZE, INFER_STRIDE))
    if not ys or ys[-1] + PATCH_SIZE < pH:
        ys.append(pH - PATCH_SIZE)
    if not xs or xs[-1] + PATCH_SIZE < pW:
        xs.append(pW - PATCH_SIZE)

    # ── Stage 1: U-Net → full-resolution probability maps ─────────────────────
    pre_prob_acc  = np.zeros((pH, pW), np.float32)
    post_prob_acc = np.zeros((pH, pW), np.float32)
    weight_acc    = np.zeros((pH, pW), np.float32)

    for y in ys:
        for x in xs:
            pre_p  = pre_arr [y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            post_p = post_arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            if pre_p.mean() < MIN_VALID:
                continue
            pp, qp = _unet_prob(pre_p, post_p)
            pre_prob_acc [y:y+PATCH_SIZE, x:x+PATCH_SIZE] += pp * hanning
            post_prob_acc[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += qp * hanning
            weight_acc   [y:y+PATCH_SIZE, x:x+PATCH_SIZE] += hanning

    safe_w        = np.where(weight_acc == 0, 1.0, weight_acc)
    pre_prob_full  = pre_prob_acc  / safe_w
    post_prob_full = post_prob_acc / safe_w

    # ── Stage 2: ViT → per-pixel class scores ─────────────────────────────────
    class_acc = np.zeros((pH, pW), np.float32)
    class_wt  = np.zeros((pH, pW), np.float32)

    for y in ys:
        for x in xs:
            pp = pre_prob_full [y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            qp = post_prob_full[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            cls = float(_vit_classify(pp, qp))   # scalar: 0, 1, or 2
            class_acc[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += cls * hanning
            class_wt [y:y+PATCH_SIZE, x:x+PATCH_SIZE] += hanning

    safe_wt   = np.where(class_wt == 0, 1.0, class_wt)
    class_full = np.round(class_acc / safe_wt).astype(np.uint8)

    # Gate: only keep non-zero classes where U-Net is confident there is damage
    label_map = np.where(post_prob_full > DAMAGE_THR, class_full, 0).astype(np.uint8)
    return label_map[:H, :W]


# ── Vectorisation ─────────────────────────────────────────────────────────────
def _label_map_to_zones(label_map: np.ndarray, transform: Affine) -> List[DamageZone]:
    zones = []
    for cls in (1, 2):
        binary = (label_map == cls).astype(np.uint8)
        if binary.sum() < 20:
            continue
        polys = [shape(g) for g, v in rasterio.features.shapes(binary, transform=transform) if v == 1]
        if not polys:
            continue
        merged = unary_union(polys)
        geoms  = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
        for geom in geoms:
            if geom.area < 1e-8:
                continue
            polygon = [[lat, lng] for lng, lat in geom.exterior.coords]
            px_mask = rasterio.features.rasterize(
                [(mapping(geom), 1)], out_shape=label_map.shape,
                transform=transform, fill=0, dtype=np.uint8,
            ).astype(bool)
            confidence = float((label_map[px_mask] == cls).mean()) if px_mask.any() else 0.9
            zones.append(DamageZone(
                polygon=polygon, damage_class=cls,
                label="Newly damaged" if cls == 1 else "Pre-existing",
                confidence=round(confidence, 3),
            ))
    return zones


# ── Temporal progression ──────────────────────────────────────────────────────
def _temporal_progression(aoi, start: str, end: str,
                           pre_arr: np.ndarray, pre_tf: Affine) -> List[TemporalPoint]:
    points, current = [], datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    while current <= e:
        last_day  = calendar.monthrange(current.year, current.month)[1]
        m_end     = min(datetime(current.year, current.month, last_day), e)
        try:
            comp           = _s2_composite(aoi, current.strftime("%Y-%m-%d"), m_end.strftime("%Y-%m-%d"))
            post_arr, _, _ = _download_as_array(comp, aoi, scale=30)
            damage_count   = int((_run_pipeline(pre_arr, post_arr) > 0).sum())
        except Exception as ex:
            print(f"[warn] temporal {current.strftime('%Y-%m')}: {ex}", file=sys.stderr)
            damage_count = 0
        points.append(TemporalPoint(date=current.strftime("%Y-%m"), damage_count=damage_count))
        current = datetime(current.year + 1, 1, 1) if current.month == 12 \
                  else datetime(current.year, current.month + 1, 1)
    return points


# ── Result cache ──────────────────────────────────────────────────────────────
def _cache_key(req: AnalyzeRequest) -> str:
    raw = f"{req.location}|{req.start_date}|{req.end_date}|{req.infrastructure_type}"
    return hashlib.md5(raw.encode()).hexdigest()


def _load_cache(key: str) -> Optional[dict]:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        print(f"[api] Cache hit: {key}", flush=True)
        return json.loads(path.read_text())
    return None


def _save_cache(key: str, data: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(data))
    print(f"[api] Result cached: {key}", flush=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":    "ok",
        "gee":       GEE_OK,
        "unet":      UNET_CKPT.exists(),
        "vit":       VIT_CKPT.exists(),
        "device":    str(DEVICE),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    # Return cached result immediately if available
    cache_key = _cache_key(req)
    cached = _load_cache(cache_key)
    if cached is not None:
        return AnalyzeResponse(**cached)

    try:
        lat, lng = [float(v.strip()) for v in req.location.split(",")]
    except Exception:
        raise HTTPException(422, "location must be 'lat,lng'")

    try:
        t_start = datetime.strptime(req.start_date, "%Y-%m-%d")
        t_end   = datetime.strptime(req.end_date,   "%Y-%m-%d")
    except ValueError:
        raise HTTPException(422, "dates must be YYYY-MM-DD")
    if t_end <= t_start:
        raise HTTPException(422, "end_date must be after start_date")

    pre_start = (t_start - timedelta(days=365)).strftime("%Y-%m-%d")
    pre_end   = (t_start - timedelta(days=1)  ).strftime("%Y-%m-%d")

    pre_arr = post_arr = pre_tf = post_tf = None
    gee_error = None
    aoi = None

    # Prefer local GeoTIFFs — full resolution, instant, no network.
    # GEE is only used when local files don't cover the requested location/period.
    local = _load_local_imagery(lat, lng, req.end_date)
    if local is not None:
        pre_arr, pre_tf, post_arr, post_tf = local
        print(f"[api] Using local GeoTIFFs for ({lat},{lng})", flush=True)

    if pre_arr is None and GEE_OK:
        try:
            delta = 0.15
            aoi = ee.Geometry.Rectangle([lng - delta, lat - delta, lng + delta, lat + delta])
            pre_arr,  pre_tf,  _ = _download_as_array(_s2_composite(aoi, pre_start, pre_end), aoi)
            post_arr, post_tf, _ = _download_as_array(_s2_composite(aoi, req.start_date, req.end_date), aoi)
        except Exception as e:
            gee_error = str(e)
            print(f"[warn] GEE fetch failed ({e})", file=sys.stderr, flush=True)

    if pre_arr is None:
        detail = f"GEE fetch failed ({gee_error}) and no local imagery for this location" \
                 if gee_error else "GEE not available and no local imagery for this location"
        raise HTTPException(502, detail)

    # Build AOI for temporal progression (needed even when imagery came from local files)
    if aoi is None and GEE_OK:
        delta = 0.15
        aoi = ee.Geometry.Rectangle([lng - delta, lat - delta, lng + delta, lat + delta])

    try:
        label_map = _run_pipeline(pre_arr, post_arr)
    except Exception as e:
        raise HTTPException(500, f"Inference pipeline failed: {e}")

    damage_zones = _label_map_to_zones(label_map, pre_tf)
    newly        = int((label_map == 1).sum())
    pre_exist    = int((label_map == 2).sum())

    # Encode imagery for the frontend map display
    try:
        pre_image_b64  = _arr_to_b64_png(pre_arr)
        post_image_b64 = _arr_to_b64_png(post_arr)
        mask_b64       = _mask_to_b64_png(label_map)
    except Exception as e:
        print(f"[warn] image encoding failed: {e}", file=sys.stderr)
        pre_image_b64 = post_image_b64 = mask_b64 = None

    if GEE_OK and gee_error is None:
        try:
            temporal = _temporal_progression(aoi, req.start_date, req.end_date, pre_arr, pre_tf)
        except Exception as e:
            print(f"[warn] temporal progression failed: {e}", file=sys.stderr)
            temporal = []
    else:
        temporal = []

    response = AnalyzeResponse(
        damage_zones=damage_zones,
        metrics={
            "zones_flagged":   len(damage_zones),
            "newly_damaged":   newly,
            "pre_existing":    pre_exist,
            "images_analyzed": 2,
            "total_damage_px": newly + pre_exist,
        },
        temporal_progression=temporal,
        pre_image_b64=pre_image_b64,
        post_image_b64=post_image_b64,
        mask_b64=mask_b64,
    )
    try:
        _save_cache(cache_key, response.model_dump())
    except Exception as e:
        print(f"[warn] cache write failed: {e}", file=sys.stderr)
    return response
