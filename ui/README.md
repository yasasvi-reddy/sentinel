# Sentinel — Interface

The production interface is a React + FastAPI application. The FastAPI backend runs the full ML inference pipeline (U-Net + ViT); the React frontend visualises results on an interactive satellite map.

---

## Prerequisites

- Python 3.10+ with the project `venv` activated
- Node.js 18+ and npm

---

## 1. Start the FastAPI backend

```bash
# From the repo root
source venv/bin/activate
uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000 --loop asyncio
```

The backend will be available at `http://localhost:8000`.
Health check: `curl http://localhost:8000/health`

---

## 2. Start the React dev server

```bash
cd war-damage-ui
npm install        # first time only
npm run dev
```

The UI will be available at `http://localhost:5173`.

---

## 3. Usage

1. Open `http://localhost:5173` in your browser
2. Enter coordinates as `lat,lng` (e.g. `49.9935,36.2304` for Kharkiv)
3. Select a date range covering the conflict period (e.g. `2022-03-01` to `2022-08-31`)
4. Click **Analyze** — the loading screen shows pipeline progress (~13 min first run, ~1 s for cached locations)
5. The results dashboard shows:
   - Satellite imagery with damage mask overlay
   - Damage zone polygons (red = newly damaged, orange = pre-existing)
   - Metrics panel: zones flagged, pixel counts, images analyzed
   - Temporal progression chart (requires active GEE connection)

---

## Cached locations (instant response)

| Location | Coordinates | Date range |
|---|---|---|
| Kharkiv, Ukraine | `49.9935,36.2304` | `2022-03-01` → `2022-08-31` |
| Mariupol, Ukraine | `47.0966,37.5416` | `2022-03-01` → `2022-08-31` |

---

## API reference

### `GET /health`
Returns server status, model load state, GEE availability, and device.

### `POST /analyze`
```json
{
  "location": "49.9935,36.2304",
  "start_date": "2022-03-01",
  "end_date": "2022-08-31",
  "infrastructure_type": "all"
}
```
Returns damage zones, metrics, temporal progression, and base64-encoded satellite imagery.
