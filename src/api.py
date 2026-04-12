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

Run with:
    uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000 --loop asyncio
"""

import io
import sys
import zipfile
import calendar
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import ee
import numpy as np
import requests
import rasterio
import rasterio.features
import rasterio.transform
import torch
from affine import Affine
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# ── Paths & constants ─────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
MODEL_DIR  = ROOT / "models"
UNET_CKPT  = MODEL_DIR / "unet_resnet34_best.pth"
VIT_CKPT   = MODEL_DIR / "temporal_vit_best.pth"

GEE_PROJECT = "project-8232277e-d8ce-4a1f-bc6"
CLOUD_THR   = 20
PATCH_SIZE  = 256
STRIDE      = 192
MIN_VALID   = 0.05

# ── Lazy globals ──────────────────────────────────────────────────────────────
GEE_OK = False
DEVICE = None
UNET   = None
VIT    = None


def _get_device():
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

    DEVICE = _get_device()
    print(f"[api] Device: {DEVICE}", flush=True)

    try:
        ee.Initialize(project=GEE_PROJECT)
        GEE_OK = True
        print("[api] GEE initialised", flush=True)
    except Exception as e:
        print(f"[warn] GEE init failed: {e}", file=sys.stderr, flush=True)

    UNET = _load_unet(DEVICE)
    VIT  = _load_vit(DEVICE)
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
        pre_t = _rgb_to_tensor(pre_arr)
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
    ph, pw = max(0, h - arr.shape[0]), max(0, w - arr.shape[1])
    return np.pad(arr, ((0, ph), (0, pw), (0, 0))) if arr.ndim == 3 else np.pad(arr, ((0, ph), (0, pw)))


def _run_pipeline(pre_arr: np.ndarray, post_arr: np.ndarray) -> np.ndarray:
    H, W     = pre_arr.shape[:2]
    pre_arr  = _pad_to(pre_arr,  max(H, PATCH_SIZE), max(W, PATCH_SIZE))
    post_arr = _pad_to(post_arr, max(H, PATCH_SIZE), max(W, PATCH_SIZE))
    pH, pW   = pre_arr.shape[:2]
    label_map  = np.zeros((pH, pW), dtype=np.float32)
    weight_map = np.zeros((pH, pW), dtype=np.float32)
    for y in range(0, pH - PATCH_SIZE + 1, STRIDE):
        for x in range(0, pW - PATCH_SIZE + 1, STRIDE):
            pre_p = pre_arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            if pre_p.mean() < MIN_VALID:
                continue
            pre_prob, post_prob = _unet_prob(pre_p, post_arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE])
            label_map [y:y+PATCH_SIZE, x:x+PATCH_SIZE] += _vit_classify(pre_prob, post_prob).astype(np.float32)
            weight_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1.0
    weight_map = np.where(weight_map == 0, 1, weight_map)
    return np.round(label_map / weight_map).astype(np.uint8)[:H, :W]


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
            comp      = _s2_composite(aoi, current.strftime("%Y-%m-%d"), m_end.strftime("%Y-%m-%d"))
            post_arr, _, _ = _download_as_array(comp, aoi, scale=30)
            damage_count   = int((_run_pipeline(pre_arr, post_arr) > 0).sum())
        except Exception as ex:
            print(f"[warn] temporal {current.strftime('%Y-%m')}: {ex}", file=sys.stderr)
            damage_count = 0
        points.append(TemporalPoint(date=current.strftime("%Y-%m"), damage_count=damage_count))
        current = datetime(current.year + 1, 1, 1) if current.month == 12 \
                  else datetime(current.year, current.month + 1, 1)
    return points


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
    if not GEE_OK:
        raise HTTPException(503, "Google Earth Engine is not initialised")

    try:
        lat, lng = [float(v.strip()) for v in req.location.split(",")]
    except Exception:
        raise HTTPException(422, "location must be 'lat,lng'")

    delta = 0.15
    aoi = ee.Geometry.Rectangle([lng - delta, lat - delta, lng + delta, lat + delta])

    try:
        t_start = datetime.strptime(req.start_date, "%Y-%m-%d")
        t_end   = datetime.strptime(req.end_date,   "%Y-%m-%d")
    except ValueError:
        raise HTTPException(422, "dates must be YYYY-MM-DD")
    if t_end <= t_start:
        raise HTTPException(422, "end_date must be after start_date")

    pre_start = (t_start - timedelta(days=365)).strftime("%Y-%m-%d")
    pre_end   = (t_start - timedelta(days=1)  ).strftime("%Y-%m-%d")

    try:
        pre_arr,  pre_tf,  _ = _download_as_array(_s2_composite(aoi, pre_start, pre_end), aoi)
        post_arr, post_tf, _ = _download_as_array(_s2_composite(aoi, req.start_date, req.end_date), aoi)
    except Exception as e:
        raise HTTPException(502, f"GEE imagery fetch failed: {e}")

    try:
        label_map = _run_pipeline(pre_arr, post_arr)
    except Exception as e:
        raise HTTPException(500, f"Inference pipeline failed: {e}")

    damage_zones = _label_map_to_zones(label_map, pre_tf)
    newly        = int((label_map == 1).sum())
    pre_exist    = int((label_map == 2).sum())

    try:
        temporal = _temporal_progression(aoi, req.start_date, req.end_date, pre_arr, pre_tf)
    except Exception as e:
        print(f"[warn] temporal progression failed: {e}", file=sys.stderr)
        temporal = []

    return AnalyzeResponse(
        damage_zones=damage_zones,
        metrics={
            "zones_flagged":   len(damage_zones),
            "newly_damaged":   newly,
            "pre_existing":    pre_exist,
            "images_analyzed": 2,
            "total_damage_px": newly + pre_exist,
        },
        temporal_progression=temporal,
    )
