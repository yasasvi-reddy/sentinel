"""
export_geotiffs.py

Exports full-resolution (10 m) Sentinel-2 GeoTIFFs to Google Drive for
Kharkiv and Mariupol across three temporal windows, monitors the GEE tasks,
then downloads the results to data/imagery/geotiffs/.

Export CRS : EPSG:32637  (UTM zone 37 N – native metric CRS for Ukraine)
Scale      : 10 m        (Sentinel-2 native RGB resolution)
Bands      : B4, B3, B2  (Red, Green, Blue) — uint16 surface reflectance
"""

import ee
import io
import json
import time
from pathlib import Path

import geopandas as gpd
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build as gdrive_build
from googleapiclient.http import MediaIoBaseDownload

from ee.oauth import CLIENT_ID, CLIENT_SECRET

# ── Auth & init ────────────────────────────────────────────────────────────────
GEE_PROJECT = "project-8232277e-d8ce-4a1f-bc6"
ee.Initialize(project=GEE_PROJECT)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
SHP_DIR = ROOT / "data" / "damage_assessments"
OUT_DIR = ROOT / "data" / "imagery" / "geotiffs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Export config ──────────────────────────────────────────────────────────────
DRIVE_FOLDER  = "sentinel_geotiffs"
EXPORT_CRS    = "EPSG:32637"   # UTM zone 37 N
EXPORT_SCALE  = 10             # metres
CLOUD_THR     = 20             # scene-level pre-filter (%)
POLL_INTERVAL = 30             # seconds between status checks

# ── Cities and periods ─────────────────────────────────────────────────────────
CITIES = {
    "kharkiv":  SHP_DIR / "kharkiv_north" / "Kharkiv_23March2022_RDA.shp",
    "mariupol": SHP_DIR / "mariupol"      / "Damage_Point_All_No_Military_V2.shp",
}

PERIODS = {
    "prewar_oct_dec2021":          ("2021-10-01", "2021-12-31"),
    "postwar_early_mar_may2022":   ("2022-03-01", "2022-05-31"),
    "postwar_late_jun_aug2023":    ("2023-06-01", "2023-08-31"),
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_aoi(shp_path: Path) -> ee.Geometry:
    gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds
    return ee.Geometry.Rectangle([minx, miny, maxx, maxy])


def mask_s2_clouds(image):
    qa    = image.select("QA60")
    mask  = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask)


def build_composite(aoi: ee.Geometry, start: str, end: str) -> ee.Image:
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_THR))
        .map(mask_s2_clouds)
        .select(["B4", "B3", "B2"])
        .median()
        .clip(aoi)
    )


def get_drive_service():
    # Prefer Application Default Credentials (gcloud auth application-default login)
    try:
        import google.auth
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        creds.refresh(Request())
        return gdrive_build("drive", "v3", credentials=creds)
    except Exception:
        pass

    # Fall back to earthengine stored credentials with full drive scope
    creds_path = Path.home() / ".config" / "earthengine" / "credentials"
    stored     = json.loads(creds_path.read_text())
    creds = Credentials(
        token=None,
        refresh_token=stored["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/drive"],  # full drive, not readonly
    )
    creds.refresh(Request())
    return gdrive_build("drive", "v3", credentials=creds)


def download_file(drive, filename: str, out_path: Path) -> bool:
    """Find `filename` in Drive and download to `out_path`."""
    results = drive.files().list(
        q=f"name='{filename}' and trashed=false",
        fields="files(id, name, size)",
        spaces="drive",
    ).execute()
    files = results.get("files", [])
    if not files:
        print(f"  [warn] '{filename}' not found in Drive")
        return False

    file_id  = files[0]["id"]
    size_mb  = int(files[0].get("size", 0)) / 1e6
    request  = drive.files().get_media(fileId=file_id)
    buf      = io.BytesIO()
    dl       = MediaIoBaseDownload(buf, request, chunksize=10 * 1024 * 1024)
    done     = False
    while not done:
        status, done = dl.next_chunk()
        print(f"    {int(status.progress()*100):3d}%  ({size_mb:.0f} MB)", end="\r")
    out_path.write_bytes(buf.getvalue())
    print(f"  Saved {out_path.name}  ({out_path.stat().st_size/1e6:.1f} MB)      ")
    return True


# ── Phase 1: Submit tasks ──────────────────────────────────────────────────────
print("=" * 60)
print("Phase 1 — Submitting GEE export tasks")
print("=" * 60)

pending = {}   # task_name → {task, out_path}

for city, shp_path in CITIES.items():
    aoi = get_aoi(shp_path)
    for period, (start, end) in PERIODS.items():
        name     = f"{city}_{period}"
        out_path = OUT_DIR / f"{name}.tif"

        if out_path.exists():
            print(f"  [skip] {name}.tif already exists locally")
            continue

        composite = build_composite(aoi, start, end)
        task = ee.batch.Export.image.toDrive(
            image          = composite,
            description    = name,
            folder         = DRIVE_FOLDER,
            fileNamePrefix = name,
            region         = aoi,
            scale          = EXPORT_SCALE,
            crs            = EXPORT_CRS,
            fileFormat     = "GeoTIFF",
            maxPixels      = int(1e9),
        )
        task.start()
        pending[name] = {"task": task, "out_path": out_path}
        print(f"  Submitted: {name}")

if not pending:
    print("  All files already exist locally — skipping export.")
else:
    # ── Phase 2: Poll until all tasks complete ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 2 — Monitoring {len(pending)} tasks  (polling every {POLL_INTERVAL}s)")
    print(f"{'='*60}")
    terminal_states = {"COMPLETED", "FAILED", "CANCELLED"}

    while True:
        states = {n: d["task"].status()["state"] for n, d in pending.items()}
        done   = all(s in terminal_states for s in states.values())

        lines = [f"  {n}: {s}" for n, s in states.items()]
        print("\n".join(lines))

        if done:
            break
        print(f"  … waiting {POLL_INTERVAL}s\n")
        time.sleep(POLL_INTERVAL)

    failed = [n for n, s in states.items() if s != "COMPLETED"]
    if failed:
        print(f"\n[ERROR] Tasks failed: {failed}")
        print("Check https://code.earthengine.google.com/tasks for details.")

    completed = {n: d for n, d in pending.items() if states[n] == "COMPLETED"}

    # ── Phase 3: Download from Drive ───────────────────────────────────────────
    if completed:
        print(f"\n{'='*60}")
        print(f"Phase 3 — Downloading {len(completed)} files from Google Drive")
        print(f"{'='*60}")
        drive = get_drive_service()
        for name, info in completed.items():
            print(f"\n  {name}")
            download_file(drive, f"{name}.tif", info["out_path"])

print("\nDone.")
