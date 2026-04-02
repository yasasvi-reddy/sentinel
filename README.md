# Sentinel

Sentinel is a satellite imagery analysis system for detecting war damage to civilian infrastructure. It uses deep learning and remote sensing techniques to identify and assess damage to buildings, roads, hospitals, and other critical infrastructure from multi-temporal satellite imagery.

## Project Structure

```
sentinel/
├── data/           # Raw and processed satellite imagery
├── models/         # Trained model weights and checkpoints
├── src/            # Core source code (preprocessing, training, inference)
├── demo/           # Demo scripts and assets
└── notebooks/      # Exploratory analysis and visualization notebooks
```

## Setup

```bash
pip install -r requirements.txt
```

## Dependencies

- **earthengine-api** — Access to Google Earth Engine satellite data
- **rasterio / geopandas** — Geospatial raster and vector data processing
- **torch / torchvision** — Deep learning model training and inference
- **scikit-learn** — Classical ML utilities and evaluation metrics
- **streamlit** — Interactive demo and visualization UI
- **numpy / pandas / matplotlib / pillow** — Data handling and visualization
