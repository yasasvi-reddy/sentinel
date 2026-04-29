# Sentinel — System Architecture

## Pipeline Overview

The Sentinel pipeline has four sequential stages:

```
Satellite Imagery Acquisition
        ↓
Preprocessing & Patch Extraction
        ↓
Inference (U-Net → ViT)
        ↓
Visualisation & API Response
```

---

## Stage 1 — Acquisition

**Source:** Sentinel-2 Level-2A multispectral imagery (10 m/pixel, bands B4/B3/B2)
**Access:** Google Earth Engine (GEE) via the `earthengine-api` Python client, or pre-downloaded GeoTIFFs stored at `data/imagery/geotiffs/`.

Two temporal composites are built per location:
- **Pre-war baseline:** median composite over Oct–Dec 2021
- **Post-war window:** median composite over the user-supplied date range (e.g. Mar–May 2022)

Cloud masking is applied using the Sentinel-2 QA60 band (bits 10 and 11: opaque clouds and cirrus). GEE exports are downloaded as GeoTIFFs at 10 m resolution; a 512-pixel thumbnail fallback is used if GEE is unavailable.

Local GeoTIFFs are checked first on every request; GEE is only called when local files do not cover the requested location.

---

## Stage 2 — Preprocessing & Patch Extraction

1. **Normalisation:** pixel values are divided by 3000 and clipped to [0, 1] (approximate Sentinel-2 reflectance scale)
2. **Patching:** the full image is sliced into overlapping 256 × 256 patches using a configurable stride (64 px during inference)
3. **Channel stacking:** each patch is a 6-channel tensor: `[pre_R, pre_G, pre_B, post_R, post_G, post_B]`
4. **Training augmentation:** horizontal flip, vertical flip, 90° rotation (applied only during training)

Training patches were extracted at stride = 128 from the full GeoTIFFs. The dataset contains 35 patches: 28 from Kharkiv and 7 from Mariupol. The balanced training configuration oversamples Mariupol to 23 patches to match Kharkiv.

---

## Stage 3 — Inference

### U-Net Segmentation (Stage 1 of inference)
- **Architecture:** U-Net with ResNet-34 encoder (`segmentation-models-pytorch`)
- **Input:** 6-channel patch (256 × 256)
- **Output:** sigmoid probability map — probability that each pixel is damaged
- **Loss:** BCE with logits (pos_weight=3) + soft Dice
- **Sliding-window blending:** Hanning window weights prevent seam artifacts at patch boundaries; accumulated probability maps are normalised by the sum of weights
- Two passes are run per full image: `pre_prob` (pre vs pre) and `post_prob` (pre vs post)

### ViT Temporal Classifier (Stage 2 of inference)
- **Architecture:** custom Vision Transformer (`temporal_vit.py`)
  - img_size=256, patch_size=16, in_channels=2, num_classes=3
  - embed_dim=192, depth=4, num_heads=6, mlp_ratio=4.0
- **Input:** 2-channel patch (`pre_prob` + `post_prob`)
- **Output:** single patch-level class — 0 (undamaged), 1 (newly damaged), 2 (pre-existing)
- The ViT class label is broadcast across the entire patch; spatial detail is preserved via the U-Net probability gate (`post_prob > 0.35`)

### Final label map
```
label_map = where(post_prob > 0.35, vit_class, 0)
```

Damage zones are vectorised with `rasterio.features.shapes` and merged with `shapely.ops.unary_union`. Each zone polygon is converted to lat/lng coordinates.

---

## Stage 4 — Visualisation & API Response

The FastAPI `/analyze` endpoint returns:
- **`damage_zones`:** list of polygons with class label, confidence, and lat/lng coordinates
- **`metrics`:** zone count, newly damaged pixels, pre-existing pixels
- **`post_image_b64`:** base64-encoded 512 × 512 PNG of the post-war composite
- **`mask_b64`:** base64-encoded RGBA PNG of the segmentation mask (red = newly damaged, orange = pre-existing)
- **`temporal_progression`:** monthly damage counts (requires active GEE connection)

Results are cached to disk at `data/cache/<md5>.json` keyed by `(location, start_date, end_date, infrastructure_type)`. Cache hits return in < 1 second.

The React frontend renders satellite imagery + mask on an HTML canvas with polygon overlays. Zone click events open a detail popup. A confidence threshold slider filters zones client-side.

---

## Training Setup

| | U-Net (baseline) | U-Net (balanced) | ViT |
|---|---|---|---|
| Epochs | 30 | 35 | 40 |
| Batch size | 8 | 8 | 8 |
| Optimizer | AdamW | AdamW | AdamW |
| LR | 3e-4 | 3e-4 | 1e-4 |
| LR schedule | Cosine | Cosine | Cosine |
| Training patches | 28 (unbalanced) | 46 (23 K + 23 M) | 28 |
| Device | Apple MPS | Apple MPS | Apple MPS |
| Best val loss | 0.638 | 0.413 | — |
| Best val IoU | 0.799 | 0.848 | — |
| Best val accuracy | — | — | 1.000 |

---

## Model Checkpoints

| File | Description |
|---|---|
| `models/unet_resnet34_best.pth` | Baseline U-Net (unbalanced training, 30 epochs) |
| `models/unet_resnet34_balanced.pth` | Balanced U-Net (oversampled Mariupol, 35 epochs) — **used in production** |
| `models/temporal_vit_best.pth` | TemporalViT (40 epochs, val_acc=1.000) |
