# Sentinel

Automated detection of war damage to civilian infrastructure using satellite imagery and deep learning.

Sentinel ingests freely available multispectral satellite imagery and tracks destruction of hospitals, schools, and water systems in active conflict zones. It uses a U-Net CNN for pixel-level damage segmentation and a Vision Transformer for temporal change tracking across image sequences.

Built for EGN 6217 Applied Deep Learning, University of Florida, Spring 2025.

---

## What it does

- Pulls Sentinel-2 satellite imagery for any location via Google Earth Engine
- Segments damage at the pixel level using a trained U-Net model
- Tracks how damage progresses over time using a Vision Transformer
- Displays results as an interactive map with a damage progression report via Streamlit

---

## Project structure

```
sentinel/
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks for exploration and training
├── src/                   # Pipeline scripts (data loading, preprocessing, model)
├── ui/                    # Streamlit demo app
├── models/                # Saved model checkpoints
├── results/               # Output plots and evaluation results
├── docs/                  # Architecture diagrams and project visuals
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Dataset

**Phase 1 (current):** Kaggle Semantic Segmentation of Aerial Imagery dataset
- Source: `humansintheloop/semantic-segmentation-of-aerial-imagery`
- 8 aerial image tiles with paired pixel-level segmentation masks
- Used to train and validate the U-Net backbone

**Phase 2:** Sentinel-2 imagery via Google Earth Engine + Copernicus EMS damage labels
- Free access with a Google Earth Engine student account
- Damage labels from Copernicus Emergency Management Service and UNOSAT
- Covers Ukraine, Gaza, Turkey conflict activations

---

## Setup

**Requirements:** Python 3.10+, pip

```bash
git clone https://github.com/yasasvi-reddy/sentinel
cd sentinel
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**For Google Earth Engine access (Phase 2):**
```bash
earthengine authenticate
```
Follow the browser prompt. You will need a free GEE account registered at earthengine.google.com.

---

## Running the notebook

Open `notebooks/setup.ipynb` to verify environment setup, load the dataset, and view sample data:

```bash
jupyter notebook notebooks/setup.ipynb
```

The notebook covers:
- Dataset loading and verification
- Sample imagery and mask visualization
- Basic dataset statistics
- Environment check

---

## Running the demo

```bash
cd demo
streamlit run app.py
```

The demo lets you select a location, date range, and infrastructure type and returns a geo-referenced damage heatmap with a temporal progression chart.

---

## Technical approach

| Component | Architecture | Purpose |
|-----------|-------------|---------|
| Segmentation | U-Net with ResNet-34 encoder | Pixel-level damage detection |
| Temporal modeling | Vision Transformer (ViT) | Multi-date change tracking |
| Interface | Streamlit | Interactive demo |
| Geospatial | rasterio, geopandas | Imagery and label processing |
| Imagery | Sentinel-2 via GEE API | Free satellite data source |

---

## Author

Yasasvi Kaipa
Master's in AI Systems, University of Florida
yasasvikaipa7@gmail.com
