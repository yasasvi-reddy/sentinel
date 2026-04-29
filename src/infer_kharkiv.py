"""
infer_kharkiv.py — Full end-to-end inference for Kharkiv

1. Load pre-war + post-war imagery (GeoTIFF preferred, PNG fallback)
2. Sliding-window U-Net → full-resolution pre_prob + post_prob maps
3. Sliding-window ViT → dense damage classification (0/1/2)
4. Save overlay and side-by-side PNGs to results/
5. Print damage statistics
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
import segmentation_models_pytorch as smp

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from temporal_vit import TemporalViT

UNET_CKPT  = ROOT / "models" / "unet_resnet34_best.pth"
VIT_CKPT   = ROOT / "models" / "temporal_vit_best.pth"
TIFF_DIR   = ROOT / "data" / "imagery" / "geotiffs"
PNG_DIR    = ROOT / "data" / "imagery"
RESULTS    = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

PATCH_SIZE   = 256
INFER_STRIDE = 64
DAMAGE_THR   = 0.35
MIN_VALID    = 0.05

MASK_COLORS = {
    1: np.array([226,  75,  74, 180], dtype=np.uint8),  # newly damaged  → red
    2: np.array([230, 160,  70, 180], dtype=np.uint8),  # pre-existing   → orange
}


# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


# ── Model loading ─────────────────────────────────────────────────────────────
def load_unet(device):
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None,
        in_channels=6, classes=1, activation=None,
    ).to(device)
    ckpt = torch.load(UNET_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_vit(device):
    model = TemporalViT(
        img_size=256, patch_size=16, in_channels=2,
        num_classes=3, embed_dim=192, depth=4,
        num_heads=6, mlp_ratio=4.0, dropout=0.1,
    ).to(device)
    ckpt = torch.load(VIT_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ── Image loading (GeoTIFF preferred, PNG fallback) ───────────────────────────
def load_image(city: str, period: str):
    tif = TIFF_DIR / f"{city}_{period}.tif"
    png = PNG_DIR  / f"{city}_{period}.png"
    if tif.exists():
        import rasterio
        with rasterio.open(tif) as src:
            data = src.read([1, 2, 3]).astype(np.float32).transpose(1, 2, 0)
        arr = np.clip(np.nan_to_num(data) / 3000.0, 0.0, 1.0)
        print(f"  Loaded GeoTIFF : {tif.name}  {arr.shape[1]}×{arr.shape[0]} px")
    elif png.exists():
        arr = np.array(Image.open(png).convert("RGB"), dtype=np.float32) / 255.0
        print(f"  Loaded PNG     : {png.name}  {arr.shape[1]}×{arr.shape[0]} px  [fallback]")
    else:
        raise FileNotFoundError(f"No imagery found for {city}/{period}")
    return arr


# ── Sliding-window helpers ────────────────────────────────────────────────────
def pad_to(arr, h, w):
    ph = max(0, h - arr.shape[0])
    pw = max(0, w - arr.shape[1])
    if arr.ndim == 3:
        return np.pad(arr, ((0, ph), (0, pw), (0, 0)))
    return np.pad(arr, ((0, ph), (0, pw)))


def make_positions(total, patch, stride):
    positions = list(range(0, total - patch, stride))
    if not positions or positions[-1] + patch < total:
        positions.append(max(0, total - patch))
    return positions


def hanning_window():
    return np.outer(
        np.hanning(PATCH_SIZE + 2)[1:-1],
        np.hanning(PATCH_SIZE + 2)[1:-1],
    ).astype(np.float32)


# ── Stage 1: sliding-window U-Net → full-res probability maps ─────────────────
@torch.no_grad()
def unet_prob_patch(model, pre_p, post_p, device):
    def to_t(a):
        return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float().to(device)
    pre_t  = to_t(pre_p)
    post_t = to_t(post_p)
    pre_prob  = torch.sigmoid(model(torch.cat([pre_t,  pre_t],  1))).squeeze().cpu().numpy()
    post_prob = torch.sigmoid(model(torch.cat([pre_t, post_t],  1))).squeeze().cpu().numpy()
    return pre_prob.astype(np.float32), post_prob.astype(np.float32)


def run_unet_sliding(unet, pre_arr, post_arr, device):
    H, W   = pre_arr.shape[:2]
    pH, pW = max(H, PATCH_SIZE), max(W, PATCH_SIZE)
    pre_p  = pad_to(pre_arr,  pH, pW)
    post_p = pad_to(post_arr, pH, pW)
    han    = hanning_window()

    ys = make_positions(pH, PATCH_SIZE, INFER_STRIDE)
    xs = make_positions(pW, PATCH_SIZE, INFER_STRIDE)
    n  = len(ys) * len(xs)
    print(f"  U-Net: {n} patches  ({len(ys)}r × {len(xs)}c,  stride={INFER_STRIDE})")

    pre_acc  = np.zeros((pH, pW), np.float32)
    post_acc = np.zeros((pH, pW), np.float32)
    wt_acc   = np.zeros((pH, pW), np.float32)

    done = 0
    for y in ys:
        for x in xs:
            pp = pre_p [y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            qp = post_p[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            done += 1
            if pp.mean() < MIN_VALID:
                continue
            pre_prob, post_prob = unet_prob_patch(unet, pp, qp, device)
            pre_acc [y:y+PATCH_SIZE, x:x+PATCH_SIZE] += pre_prob  * han
            post_acc[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += post_prob * han
            wt_acc  [y:y+PATCH_SIZE, x:x+PATCH_SIZE] += han
            if done % 20 == 0 or done == n:
                print(f"    {done}/{n}", end="\r", flush=True)

    print()
    safe_w = np.where(wt_acc == 0, 1.0, wt_acc)
    return (pre_acc / safe_w)[:H, :W], (post_acc / safe_w)[:H, :W]


# ── Stage 2: sliding-window ViT → dense class map ────────────────────────────
@torch.no_grad()
def vit_classify_patch(vit, pre_prob, post_prob, device):
    x = torch.stack([
        torch.from_numpy(pre_prob),
        torch.from_numpy(post_prob),
    ], 0).unsqueeze(0).float().to(device)
    return int(vit(x).argmax(dim=1).item())


def run_vit_sliding(vit, pre_prob_full, post_prob_full, device):
    H, W   = pre_prob_full.shape
    pH, pW = max(H, PATCH_SIZE), max(W, PATCH_SIZE)
    pre_p  = pad_to(pre_prob_full,  pH, pW)
    post_p = pad_to(post_prob_full, pH, pW)
    han    = hanning_window()

    ys = make_positions(pH, PATCH_SIZE, INFER_STRIDE)
    xs = make_positions(pW, PATCH_SIZE, INFER_STRIDE)
    n  = len(ys) * len(xs)
    print(f"  ViT : {n} patches  ({len(ys)}r × {len(xs)}c,  stride={INFER_STRIDE})")

    cls_acc = np.zeros((pH, pW), np.float32)
    cls_wt  = np.zeros((pH, pW), np.float32)

    done = 0
    for y in ys:
        for x in xs:
            pp  = pre_p [y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            qp  = post_p[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            cls = float(vit_classify_patch(vit, pp, qp, device))
            cls_acc[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += cls * han
            cls_wt [y:y+PATCH_SIZE, x:x+PATCH_SIZE] += han
            done += 1
            if done % 20 == 0 or done == n:
                print(f"    {done}/{n}", end="\r", flush=True)

    print()
    safe_wt    = np.where(cls_wt == 0, 1.0, cls_wt)
    class_full = np.round(cls_acc / safe_wt).astype(np.uint8)[:H, :W]
    return np.where(post_prob_full > DAMAGE_THR, class_full, 0).astype(np.uint8)


# ── Visualisation ─────────────────────────────────────────────────────────────
def build_rgba_mask(label_map):
    H, W = label_map.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for cls, color in MASK_COLORS.items():
        rgba[label_map == cls] = color
    return rgba


def overlay_mask(rgb_arr, label_map):
    rgb8 = (np.clip(rgb_arr, 0, 1) * 255).astype(np.uint8)
    base = Image.fromarray(rgb8, "RGB").convert("RGBA")
    mask = Image.fromarray(build_rgba_mask(label_map), "RGBA")
    return Image.alpha_composite(base, mask).convert("RGB")


def make_comparison(pre_arr, post_arr, label_map):
    pre8     = (np.clip(pre_arr, 0, 1) * 255).astype(np.uint8)
    pre_img  = Image.fromarray(pre8, "RGB")
    post_img = overlay_mask(post_arr, label_map)

    H, W    = pre8.shape[:2]
    LABEL_H = 36
    GAP     = 8
    canvas  = Image.new("RGB", (W * 2 + GAP, H + LABEL_H), (15, 17, 20))
    canvas.paste(pre_img,  (0,        LABEL_H))
    canvas.paste(post_img, (W + GAP,  LABEL_H))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, W - 1, LABEL_H - 1],           fill=(30, 35, 45))
    draw.rectangle([W + GAP, 0, W * 2 + GAP, LABEL_H - 1], fill=(30, 35, 45))
    draw.text((8,       10), "PRE-WAR  (Oct–Dec 2021)",                    fill=(180, 178, 170))
    draw.text((W + GAP + 8, 10), "POST-WAR (Mar–May 2022) + damage mask", fill=(180, 178, 170))

    # Legend bottom-right of post panel
    lx = W + GAP + 12
    ly = H + LABEL_H - 26
    draw.rectangle([lx,       ly + 4, lx + 14,  ly + 17], fill=(226,  75,  74))
    draw.text(     (lx + 18, ly + 5), "Newly damaged",    fill=(200, 198, 192))
    draw.rectangle([lx + 130, ly + 4, lx + 144, ly + 17], fill=(230, 160,  70))
    draw.text(     (lx + 148, ly + 5), "Pre-existing",    fill=(200, 198, 192))

    return canvas


def save_prob_maps(pre_prob, post_prob, out_path):
    H, W = pre_prob.shape

    def to_u8(p):
        return (np.clip(p, 0, 1) * 255).astype(np.uint8)

    GAP    = 8
    LABEL_H = 28
    canvas = Image.new("RGB", (W * 2 + GAP, H + LABEL_H), (15, 17, 20))
    canvas.paste(Image.fromarray(to_u8(pre_prob),  "L").convert("RGB"), (0,        LABEL_H))
    canvas.paste(Image.fromarray(to_u8(post_prob), "L").convert("RGB"), (W + GAP, LABEL_H))
    draw = ImageDraw.Draw(canvas)
    draw.text((8,       8), "pre_prob  (U-Net baseline — pre vs pre)",  fill=(140, 138, 130))
    draw.text((W + GAP + 8, 8), "post_prob (U-Net damage — pre vs post)", fill=(140, 138, 130))
    canvas.save(out_path)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print(f"\nDevice : {device}")

    print("\nLoading models …")
    unet = load_unet(device)
    vit  = load_vit(device)
    print("  U-Net  ✓")
    print("  ViT    ✓")

    print("\nLoading imagery …")
    pre_arr  = load_image("kharkiv", "prewar_oct_dec2021")
    post_arr = load_image("kharkiv", "postwar_early_mar_may2022")

    # Align post to pre dimensions if they differ
    if pre_arr.shape != post_arr.shape:
        H, W = pre_arr.shape[:2]
        post_arr = np.array(
            Image.fromarray((post_arr * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
            dtype=np.float32,
        ) / 255.0
        print(f"  Resized post → {W}×{H}")

    H, W = pre_arr.shape[:2]
    print(f"  Final image size : {W} × {H} px")

    # ── Stage 1: U-Net ────────────────────────────────────────────────────────
    print("\nStage 1 — U-Net sliding-window inference …")
    pre_prob, post_prob = run_unet_sliding(unet, pre_arr, post_arr, device)
    print(f"  pre_prob  range [{pre_prob.min():.3f}, {pre_prob.max():.3f}]  mean={pre_prob.mean():.3f}")
    print(f"  post_prob range [{post_prob.min():.3f}, {post_prob.max():.3f}]  mean={post_prob.mean():.3f}")

    # ── Stage 2: ViT ──────────────────────────────────────────────────────────
    print("\nStage 2 — ViT sliding-window classification …")
    label_map = run_vit_sliding(vit, pre_prob, post_prob, device)

    # ── Statistics ────────────────────────────────────────────────────────────
    total      = H * W
    n_undamaged = int((label_map == 0).sum())
    n_new       = int((label_map == 1).sum())
    n_existing  = int((label_map == 2).sum())

    print()
    print("=" * 58)
    print("DAMAGE STATISTICS — Kharkiv (post-war Mar–May 2022 vs pre)")
    print("=" * 58)
    print(f"  Total pixels      : {total:>12,}")
    print(f"  Undamaged         : {n_undamaged:>12,}  ({100*n_undamaged/total:5.1f} %)")
    print(f"  Newly damaged     : {n_new:>12,}  ({100*n_new/total:5.1f} %)  ← class 1")
    print(f"  Pre-existing dmg  : {n_existing:>12,}  ({100*n_existing/total:5.1f} %)  ← class 2")
    print(f"  Total damage      : {n_new+n_existing:>12,}  ({100*(n_new+n_existing)/total:5.1f} %)")
    print("=" * 58)

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\nSaving outputs …")

    out_overlay = RESULTS / "kharkiv_damage_overlay.png"
    overlay_mask(post_arr, label_map).save(out_overlay)
    print(f"  [1] Damage overlay  → {out_overlay.relative_to(ROOT)}")

    out_compare = RESULTS / "kharkiv_prepost_comparison.png"
    make_comparison(pre_arr, post_arr, label_map).save(out_compare)
    print(f"  [2] Pre/post compare → {out_compare.relative_to(ROOT)}")

    out_probs = RESULTS / "kharkiv_prob_maps.png"
    save_prob_maps(pre_prob, post_prob, out_probs)
    print(f"  [3] Prob maps       → {out_probs.relative_to(ROOT)}")

    print("\nDone.")
    return out_overlay, out_compare, out_probs


if __name__ == "__main__":
    main()
