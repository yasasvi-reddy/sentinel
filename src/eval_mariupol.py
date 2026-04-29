"""
eval_mariupol.py — Full evaluation pipeline for Mariupol

Part 1 — Full-image inference (same as infer_kharkiv.py):
  Sliding-window U-Net → prob maps → ViT → damage label map
  Saves: mariupol_damage_overlay.png
         mariupol_prepost_comparison.png
         mariupol_prob_maps.png

Part 2 — Patch-level evaluation (same as eval_unet.py):
  20% test split (seed=42) on Mariupol patches only
  Metrics: IoU, Precision, Recall, F1 per class + Mean IoU + Overall Acc
  Saves: mariupol_confusion_matrix.png
         mariupol_eval_report.txt

Run:
    python src/eval_mariupol.py
"""

import os, sys, random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import segmentation_models_pytorch as smp

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from temporal_vit import TemporalViT

# ── Paths ──────────────────────────────────────────────────────────────────────
PATCHES_DIR = ROOT / "data" / "patches" / "mariupol"
TIFF_DIR    = ROOT / "data" / "imagery" / "geotiffs"
UNET_CKPT   = ROOT / "models" / "unet_resnet34_best.pth"
VIT_CKPT    = ROOT / "models" / "temporal_vit_best.pth"
RESULTS     = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))

PATCH_SIZE   = 256
INFER_STRIDE = 64
DAMAGE_THR   = 0.35
MIN_VALID    = 0.05
THRESHOLD    = 0.5
SEED         = 42
TEST_SPLIT   = 0.20

CLASS_NAMES = ["Undamaged", "Damaged"]

MASK_COLORS = {
    1: np.array([226,  75,  74, 180], dtype=np.uint8),
    2: np.array([230, 160,  70, 180], dtype=np.uint8),
}


# ── Model loading ──────────────────────────────────────────────────────────────
def load_unet(device):
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None,
        in_channels=6, classes=1, activation=None,
    ).to(device)
    ckpt = torch.load(UNET_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[eval] U-Net loaded  (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    return model, ckpt


def load_vit(device):
    model = TemporalViT(
        img_size=256, patch_size=16, in_channels=2,
        num_classes=3, embed_dim=192, depth=4,
        num_heads=6, mlp_ratio=4.0, dropout=0.1,
    ).to(device)
    ckpt = torch.load(VIT_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[eval] ViT loaded    (epoch {ckpt['epoch']})")
    return model


# ── Image loading ──────────────────────────────────────────────────────────────
def load_tiff(path: Path) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as src:
        data = src.read([1, 2, 3]).astype(np.float32).transpose(1, 2, 0)
    return np.clip(np.nan_to_num(data) / 3000.0, 0.0, 1.0)


# ── Sliding-window helpers ────────────────────────────────────────────────────
def pad_to(arr, h, w):
    ph = max(0, h - arr.shape[0]); pw = max(0, w - arr.shape[1])
    return (np.pad(arr, ((0, ph), (0, pw), (0, 0))) if arr.ndim == 3
            else np.pad(arr, ((0, ph), (0, pw))))


def make_positions(total, patch, stride):
    pos = list(range(0, total - patch, stride))
    if not pos or pos[-1] + patch < total:
        pos.append(max(0, total - patch))
    return pos


def hanning_window():
    return np.outer(
        np.hanning(PATCH_SIZE + 2)[1:-1],
        np.hanning(PATCH_SIZE + 2)[1:-1],
    ).astype(np.float32)


@torch.no_grad()
def unet_prob_patch(model, pre_p, post_p):
    def to_t(a):
        return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    pre_t = to_t(pre_p); post_t = to_t(post_p)
    pre_prob  = torch.sigmoid(model(torch.cat([pre_t,  pre_t],  1))).squeeze().cpu().numpy()
    post_prob = torch.sigmoid(model(torch.cat([pre_t, post_t],  1))).squeeze().cpu().numpy()
    return pre_prob.astype(np.float32), post_prob.astype(np.float32)


def run_unet_sliding(unet, pre_arr, post_arr):
    H, W   = pre_arr.shape[:2]
    pH, pW = max(H, PATCH_SIZE), max(W, PATCH_SIZE)
    pre_p  = pad_to(pre_arr,  pH, pW)
    post_p = pad_to(post_arr, pH, pW)
    han    = hanning_window()
    ys     = make_positions(pH, PATCH_SIZE, INFER_STRIDE)
    xs     = make_positions(pW, PATCH_SIZE, INFER_STRIDE)
    n      = len(ys) * len(xs)
    print(f"  U-Net: {n} patches ({len(ys)}r × {len(xs)}c, stride={INFER_STRIDE})")

    pre_acc = np.zeros((pH, pW), np.float32)
    post_acc= np.zeros((pH, pW), np.float32)
    wt_acc  = np.zeros((pH, pW), np.float32)
    done    = 0

    for y in ys:
        for x in xs:
            pp = pre_p [y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            qp = post_p[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            done += 1
            if pp.mean() < MIN_VALID:
                continue
            pre_prob, post_prob = unet_prob_patch(unet, pp, qp)
            pre_acc [y:y+PATCH_SIZE, x:x+PATCH_SIZE] += pre_prob  * han
            post_acc[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += post_prob * han
            wt_acc  [y:y+PATCH_SIZE, x:x+PATCH_SIZE] += han
            if done % 20 == 0 or done == n:
                print(f"    {done}/{n}", end="\r", flush=True)

    print()
    safe_w = np.where(wt_acc == 0, 1.0, wt_acc)
    return (pre_acc / safe_w)[:H, :W], (post_acc / safe_w)[:H, :W]


@torch.no_grad()
def vit_classify_patch(vit, pre_prob, post_prob):
    x = torch.stack([torch.from_numpy(pre_prob), torch.from_numpy(post_prob)],
                    0).unsqueeze(0).float().to(DEVICE)
    return int(vit(x).argmax(dim=1).item())


def run_vit_sliding(vit, pre_prob_full, post_prob_full):
    H, W   = pre_prob_full.shape
    pH, pW = max(H, PATCH_SIZE), max(W, PATCH_SIZE)
    pre_p  = pad_to(pre_prob_full,  pH, pW)
    post_p = pad_to(post_prob_full, pH, pW)
    han    = hanning_window()
    ys     = make_positions(pH, PATCH_SIZE, INFER_STRIDE)
    xs     = make_positions(pW, PATCH_SIZE, INFER_STRIDE)
    n      = len(ys) * len(xs)
    print(f"  ViT : {n} patches ({len(ys)}r × {len(xs)}c, stride={INFER_STRIDE})")

    cls_acc = np.zeros((pH, pW), np.float32)
    cls_wt  = np.zeros((pH, pW), np.float32)
    done    = 0

    for y in ys:
        for x in xs:
            pp  = pre_p [y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            qp  = post_p[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            cls = float(vit_classify_patch(vit, pp, qp))
            cls_acc[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += cls * han
            cls_wt [y:y+PATCH_SIZE, x:x+PATCH_SIZE] += han
            done += 1
            if done % 20 == 0 or done == n:
                print(f"    {done}/{n}", end="\r", flush=True)

    print()
    safe_wt    = np.where(cls_wt == 0, 1.0, cls_wt)
    class_full = np.round(cls_acc / safe_wt).astype(np.uint8)[:H, :W]
    return np.where(post_prob_full > DAMAGE_THR, class_full, 0).astype(np.uint8)


# ── Visualisation ──────────────────────────────────────────────────────────────
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
    H, W     = pre8.shape[:2]
    LABEL_H  = 36; GAP = 8
    canvas   = Image.new("RGB", (W * 2 + GAP, H + LABEL_H), (15, 17, 20))
    canvas.paste(pre_img,  (0,       LABEL_H))
    canvas.paste(post_img, (W + GAP, LABEL_H))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, W - 1, LABEL_H - 1],              fill=(30, 35, 45))
    draw.rectangle([W + GAP, 0, W * 2 + GAP, LABEL_H - 1],  fill=(30, 35, 45))
    draw.text((8,            10), "PRE-WAR  (Oct–Dec 2021)",                     fill=(180, 178, 170))
    draw.text((W + GAP + 8,  10), "POST-WAR (Mar–May 2022) + damage mask",       fill=(180, 178, 170))
    lx = W + GAP + 12; ly = H + LABEL_H - 26
    draw.rectangle([lx,       ly + 4, lx + 14,  ly + 17], fill=(226,  75,  74))
    draw.text(     (lx + 18,  ly + 5), "Newly damaged",    fill=(200, 198, 192))
    draw.rectangle([lx + 130, ly + 4, lx + 144, ly + 17], fill=(230, 160,  70))
    draw.text(     (lx + 148, ly + 5), "Pre-existing",     fill=(200, 198, 192))
    return canvas


def save_prob_maps(pre_prob, post_prob, out_path):
    H, W = pre_prob.shape
    def to_u8(p): return (np.clip(p, 0, 1) * 255).astype(np.uint8)
    GAP = 8; LABEL_H = 28
    canvas = Image.new("RGB", (W * 2 + GAP, H + LABEL_H), (15, 17, 20))
    canvas.paste(Image.fromarray(to_u8(pre_prob),  "L").convert("RGB"), (0,       LABEL_H))
    canvas.paste(Image.fromarray(to_u8(post_prob), "L").convert("RGB"), (W + GAP, LABEL_H))
    draw = ImageDraw.Draw(canvas)
    draw.text((8,         8), "pre_prob  (U-Net baseline — pre vs pre)",  fill=(140, 138, 130))
    draw.text((W + GAP + 8, 8), "post_prob (U-Net damage — pre vs post)", fill=(140, 138, 130))
    canvas.save(out_path)


# ── Patch evaluation helpers ───────────────────────────────────────────────────
def collect_mariupol_patches():
    samples = []
    pre_dir  = PATCHES_DIR / "pre"
    post_dir = PATCHES_DIR / "post"
    mask_dir = PATCHES_DIR / "masks"
    if not pre_dir.exists():
        return samples
    for fname in sorted(os.listdir(pre_dir)):
        pre_p  = pre_dir  / fname
        post_p = post_dir / fname
        mask_p = mask_dir / fname
        if post_p.exists() and mask_p.exists():
            samples.append((pre_p, post_p, mask_p))
    return samples


def load_patch(pre_p, post_p, mask_p):
    pre  = np.array(Image.open(pre_p),  dtype=np.float32) / 255.0
    post = np.array(Image.open(post_p), dtype=np.float32) / 255.0
    mask = (np.array(Image.open(mask_p), dtype=np.float32) > 127).astype(np.float32)
    x = np.concatenate([pre.transpose(2,0,1), post.transpose(2,0,1)], axis=0)
    return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(mask)


def compute_metrics(all_preds, all_targets):
    p = all_preds.astype(bool); t = all_targets.astype(bool)
    metrics = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        pred_cls   = ~p if cls_idx == 0 else p
        target_cls = ~t if cls_idx == 0 else t
        tp = (pred_cls &  target_cls).sum()
        fp = (pred_cls & ~target_cls).sum()
        fn = (~pred_cls & target_cls).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        metrics[cls_name] = {
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "iou": float(iou), "support": int(target_cls.sum()),
        }
    metrics["overall_accuracy"] = float((p == t).mean())
    metrics["mean_iou"]         = float(np.mean([metrics[c]["iou"] for c in CLASS_NAMES]))
    return metrics


def build_confusion_matrix(all_preds, all_targets):
    p = all_preds.astype(bool); t = all_targets.astype(bool)
    tn = (~p & ~t).sum(); fp = ( p & ~t).sum()
    fn = (~p &  t).sum(); tp = ( p &  t).sum()
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0d0f14"); ax.set_facecolor("#161b27")
    total = cm.sum(); norm = cm / total
    im = ax.imshow(norm, cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{norm[i,j]*100:.1f}%\n({cm[i,j]:,})",
                    ha="center", va="center", fontsize=11,
                    color="white" if norm[i,j] < 0.6 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES, color="white", fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, color="white", fontsize=11)
    ax.set_xlabel("Predicted", color="white", fontsize=12)
    ax.set_ylabel("Actual",    color="white", fontsize=12)
    ax.set_title("U-Net Confusion Matrix — Mariupol (Test Set)", color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#2d3248")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[eval] Saved → {save_path}")


def format_report(metrics, test_size, total_patches, cm, ckpt_meta):
    lines = []
    lines.append("=" * 60)
    lines.append("U-NET EVALUATION REPORT — MARIUPOL")
    lines.append(f"Checkpoint   : {UNET_CKPT.name}")
    lines.append(f"Ckpt epoch   : {ckpt_meta['epoch']}  val_loss={ckpt_meta['val_loss']:.4f}  val_iou={ckpt_meta['val_iou']:.3f}")
    lines.append(f"Dataset      : Mariupol patches only ({total_patches} total)")
    lines.append(f"Test split   : {test_size} patches  (seed={SEED}, {int(TEST_SPLIT*100)}% holdout)")
    lines.append(f"Device       : {DEVICE}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"{'Class':<16} {'IoU':>7} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>10}")
    lines.append("-" * 60)
    for cls in CLASS_NAMES:
        m = metrics[cls]
        lines.append(f"{cls:<16} {m['iou']:>7.4f} {m['precision']:>10.4f} "
                     f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['support']:>10,}")
    lines.append("-" * 60)
    lines.append(f"{'Mean IoU':<16} {metrics['mean_iou']:>7.4f}")
    lines.append(f"{'Overall Acc':<16} {metrics['overall_accuracy']:>7.4f}")
    lines.append("")
    lines.append("Confusion Matrix (rows=Actual, cols=Predicted):")
    lines.append(f"               Undamaged    Damaged")
    lines.append(f"  Undamaged  {cm[0,0]:>10,} {cm[0,1]:>10,}")
    lines.append(f"  Damaged    {cm[1,0]:>10,} {cm[1,1]:>10,}")
    lines.append("")
    lines.append(f"Note: only {total_patches} Mariupol patches available; "
                 f"test set is {test_size} patch(es).")
    lines.append("=" * 60)
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n[eval] Device: {DEVICE}\n")

    # Load models once
    unet, unet_ckpt = load_unet(DEVICE)
    vit             = load_vit(DEVICE)

    # ── Part 1: Full-image inference ──────────────────────────────────────────
    pre_tif  = TIFF_DIR / "mariupol_prewar_oct_dec2021.tif"
    post_tif = TIFF_DIR / "mariupol_postwar_early_mar_may2022.tif"

    print("\n--- Part 1: Full-image inference ---")
    print(f"Loading {pre_tif.name}  …")
    pre_arr  = load_tiff(pre_tif)
    print(f"  {pre_arr.shape[1]} × {pre_arr.shape[0]} px")

    print(f"Loading {post_tif.name}  …")
    post_arr = load_tiff(post_tif)
    print(f"  {post_arr.shape[1]} × {post_arr.shape[0]} px")

    # Align to same dims
    if pre_arr.shape != post_arr.shape:
        H, W = pre_arr.shape[:2]
        post_arr = np.array(
            Image.fromarray((post_arr * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
            dtype=np.float32,
        ) / 255.0
        print(f"  Resized post → {W}×{H}")

    H, W = pre_arr.shape[:2]
    print(f"  Final size: {W} × {H} px")

    print("\nStage 1 — U-Net sliding-window …")
    pre_prob, post_prob = run_unet_sliding(unet, pre_arr, post_arr)
    print(f"  pre_prob  [{pre_prob.min():.3f}, {pre_prob.max():.3f}]  mean={pre_prob.mean():.3f}")
    print(f"  post_prob [{post_prob.min():.3f}, {post_prob.max():.3f}]  mean={post_prob.mean():.3f}")

    print("\nStage 2 — ViT sliding-window …")
    label_map = run_vit_sliding(vit, pre_prob, post_prob)

    total = H * W
    n_und = int((label_map == 0).sum())
    n_new = int((label_map == 1).sum())
    n_pre = int((label_map == 2).sum())

    print()
    print("=" * 58)
    print("DAMAGE STATISTICS — Mariupol (post-war Mar–May 2022 vs pre)")
    print("=" * 58)
    print(f"  Total pixels      : {total:>12,}")
    print(f"  Undamaged         : {n_und:>12,}  ({100*n_und/total:5.1f} %)")
    print(f"  Newly damaged     : {n_new:>12,}  ({100*n_new/total:5.1f} %)  ← class 1")
    print(f"  Pre-existing dmg  : {n_pre:>12,}  ({100*n_pre/total:5.1f} %)  ← class 2")
    print(f"  Total damage      : {n_new+n_pre:>12,}  ({100*(n_new+n_pre)/total:5.1f} %)")
    print("=" * 58)

    print("\nSaving overlay PNGs …")
    overlay_mask(post_arr, label_map).save(RESULTS / "mariupol_damage_overlay.png")
    print(f"  mariupol_damage_overlay.png")
    make_comparison(pre_arr, post_arr, label_map).save(RESULTS / "mariupol_prepost_comparison.png")
    print(f"  mariupol_prepost_comparison.png")
    save_prob_maps(pre_prob, post_prob, RESULTS / "mariupol_prob_maps.png")
    print(f"  mariupol_prob_maps.png")

    # ── Part 2: Patch-level evaluation ────────────────────────────────────────
    print("\n--- Part 2: Patch-level evaluation ---")
    samples = collect_mariupol_patches()
    print(f"[eval] Mariupol patches: {len(samples)}")

    rng = random.Random(SEED)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n_test  = max(1, int(len(shuffled) * TEST_SPLIT))
    test_s  = shuffled[:n_test]
    print(f"[eval] Test patches: {len(test_s)}  (remaining {len(shuffled)-len(test_s)} excluded)")

    all_preds = []; all_targets = []
    print(f"[eval] Running inference on {len(test_s)} test patch(es) …")
    with torch.no_grad():
        for i, (pre_p, post_p, mask_p) in enumerate(test_s):
            x, mask = load_patch(pre_p, post_p, mask_p)
            x = x.to(DEVICE)
            logit = unet(x).squeeze().cpu()
            pred  = (torch.sigmoid(logit) > THRESHOLD).numpy().astype(np.uint8)
            tgt   = mask.numpy().astype(np.uint8)
            all_preds.append(pred.ravel())
            all_targets.append(tgt.ravel())
            print(f"  [{i+1}/{len(test_s)}] {Path(pre_p).name}  "
                  f"damage_px={tgt.sum()}  pred_px={pred.sum()}")

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    metrics = compute_metrics(all_preds, all_targets)
    cm      = build_confusion_matrix(all_preds, all_targets)

    plot_confusion_matrix(cm, RESULTS / "mariupol_confusion_matrix.png")

    report = format_report(metrics, len(test_s), len(samples), cm, unet_ckpt)
    report_path = RESULTS / "mariupol_eval_report.txt"
    report_path.write_text(report)
    print(f"[eval] Saved → {report_path}")

    print("\n" + report)
    print("\n[eval] Done. All outputs in results/")


if __name__ == "__main__":
    main()
