# Sentinel

Sentinel is an automated war damage detection system that quantifies destruction of civilian infrastructure using freely available Sentinel-2 satellite imagery and deep learning. Given a location and date range, Sentinel downloads pre- and post-conflict multispectral composites from Google Earth Engine, segments damage at the pixel level using a U-Net, classifies change temporally with a Vision Transformer, and serves the results through an interactive React map with polygon-level zone inspection. It was built to give humanitarian analysts, journalists, and aid organizations an objective, reproducible measure of conflict impact on hospitals, schools, and essential services.

Built for EGN 6217 Applied Deep Learning, University of Florida, Spring 2025.

---

## Architecture

Satellite imagery is fetched from the Copernicus Sentinel-2 archive via Google Earth Engine and preprocessed into overlapping 256 × 256 patches. A U-Net with a ResNet-34 encoder runs sliding-window inference across both pre- and post-war composites to produce full-resolution damage probability maps. A TemporalViT then compares the two probability maps patch-by-patch to classify each region as undamaged, newly damaged, or pre-existing damage. Results are vectorised into geo-referenced polygons and served through a FastAPI backend to a React frontend. See `docs/architecture.md` for full details.

---

## Performance

Evaluated on a held-out 20% test split (seed=42), balanced checkpoint `unet_resnet34_balanced.pth`:

| Metric | Kharkiv | Mariupol | Combined |
|---|---|---|---|
| Mean IoU | 0.849 | 0.673 | **0.838** |
| Overall Accuracy | 91.9% | 90.4% | **91.4%** |
| Damaged Precision | 87.5% | 50.0% | 83.7% |
| Damaged Recall | **97.6%** | **82.0%** | **96.5%** |
| Damaged F1 | 92.3% | 62.1% | 89.6% |

Full per-class metrics: `results/eval_compare_report.txt`
Confusion matrix: `results/confusion_matrix.png`
Loss curves: `results/loss_curves.png`

---

## Setup

**Requirements:** Python 3.10+, Node.js 18+

```bash
git clone https://github.com/yasasvi-reddy/sentinel
cd sentinel
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Optional — Google Earth Engine (for new locations):**
```bash
earthengine authenticate
```
Without GEE, the system falls back to pre-downloaded GeoTIFFs for Kharkiv and Mariupol.

---

## Running the FastAPI backend

```bash
source venv/bin/activate
uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000 --loop asyncio
```

Health check:
```bash
curl http://localhost:8000/health
```

---

## Running the React frontend

```bash
cd war-damage-ui
npm install   # first time only
npm run dev
```

Open `http://localhost:5173`. Enter coordinates as `lat,lng` and a date range, then click **Analyze**.

**Pre-cached locations (instant response, ~3 s loading screen):**

| Location | Coordinates | Date range |
|---|---|---|
| Kharkiv, Ukraine | `49.9935,36.2304` | `2022-03-01` → `2022-08-31` |
| Mariupol, Ukraine | `47.0966,37.5416` | `2022-03-01` → `2022-08-31` |

---

## Evaluation notebook

```bash
jupyter notebook notebooks/evaluation.ipynb
```

Pre-run with all outputs. Loads the balanced checkpoint, runs the full test evaluation, and displays all metrics, confusion matrix, loss curves, and probability maps inline.

---

## Project structure

```
sentinel/
├── src/                        # Python pipeline and API
│   ├── api.py                  # FastAPI backend (main entry point)
│   ├── temporal_vit.py         # TemporalViT architecture
│   ├── train.py                # U-Net training (baseline)
│   ├── train_balanced.py       # U-Net training (balanced, production)
│   ├── eval_unet.py            # Kharkiv evaluation
│   ├── eval_mariupol.py        # Mariupol full inference + evaluation
│   ├── eval_compare.py         # Side-by-side comparison of checkpoints
│   └── plot_loss_curves.py     # Training history visualisation
├── war-damage-ui/              # React + Vite frontend
├── models/                     # Saved checkpoints (Git LFS)
│   ├── unet_resnet34_best.pth          # Baseline U-Net
│   ├── unet_resnet34_balanced.pth      # Balanced U-Net (production)
│   └── temporal_vit_best.pth           # TemporalViT
├── data/
│   ├── imagery/geotiffs/       # Pre-downloaded Sentinel-2 GeoTIFFs
│   ├── patches/                # 256×256 training patches
│   └── cache/                  # Disk-cached API responses
├── results/                    # Evaluation outputs and figures
├── notebooks/                  # Jupyter evaluation notebook
├── docs/                       # Architecture notes
├── ui/                         # Legacy interface reference
└── requirements.txt
```

---

## Known issues

**GEE authentication and temporal chart:** Google Earth Engine authentication can time out in restricted network environments. When GEE is unavailable the system falls back to local GeoTIFFs for Kharkiv and Mariupol; other locations will fail with a 502 error. The temporal progression chart on the dashboard requires GEE for monthly composite generation and will render empty without it.

**ViT overfitting on small patch count:** The TemporalViT reached val_acc=1.000 after 13 epochs on 35 training patches, which almost certainly reflects overfitting rather than genuine generalisation. It operates as a patch-level classifier (one class per 256 × 256 patch) and spatial detail in the damage map comes entirely from the U-Net probability threshold, not the ViT.

**Kharkiv pre-existing pixel inflation:** The pre-existing damage class (17.1% of Kharkiv pixels) is notably high compared to external damage assessments. This likely reflects radiometric differences between the Oct–Dec 2021 pre-war composite and the post-war images that the U-Net incorrectly interprets as structural change, rather than actual pre-conflict damage at that scale.

**Geographic scope:** The system has only been validated on Kharkiv and Mariupol. Extending to new cities requires new local GeoTIFFs or an active GEE connection, and model performance on unseen geographies is untested.

---

## Contact

Yasasvi Kaipa
University of Florida
yasasvikaipa@ufl.edu
