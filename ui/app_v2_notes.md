# Interface Evolution: v1 → v2

## v1 — Streamlit Prototype (`app_prototype_v1.py`)

The original interface was a single-file Streamlit app. It allowed basic location input and displayed a damage heatmap, but had several limitations:

- Blocking inference: the entire Streamlit process froze during the 13-minute sliding-window inference
- No satellite imagery display — only the raw segmentation mask
- No polygon-level zone detail or click-through inspection
- Limited to Streamlit's layout constraints; no custom map rendering

## v2 — React + FastAPI (current, `war-damage-ui/`)

The system was rebuilt as a two-process production stack:

| | v1 Streamlit | v2 React + FastAPI |
|---|---|---|
| Frontend | Streamlit (Python) | React 18 + Vite |
| Backend | In-process | FastAPI (async, port 8000) |
| Inference blocking | Yes — freezes UI | No — API call is async |
| Map rendering | Static PIL image | Canvas-based, interactive |
| Zone inspection | None | Click any zone for detail popup |
| Result caching | None | Disk-based JSON cache (instant repeat queries) |
| Satellite imagery | None | Base64 PNG overlay from GeoTIFF |

## Migration

No data or model changes were required. The FastAPI backend reuses the same `_run_pipeline()` inference code. The React frontend calls `POST /analyze` and renders the `AnalyzeResponse` schema directly.

The Streamlit prototype is preserved at `ui/app_prototype_v1.py` for reference.
