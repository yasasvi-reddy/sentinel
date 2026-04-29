"""
eval_unet.py — U-Net evaluation on held-out test split

Loads the saved checkpoint, creates a reproducible 20% test split (seed=42),
runs inference and computes per-class IoU, precision, recall, F1, overall
pixel accuracy, and mean IoU.

Saves to results/:
  confusion_matrix.png
  eval_report.txt

Run:
    python src/eval_unet.py
"""

import os, sys, random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import segmentation_models_pytorch as smp

# ── Paths ──────────────────────────────────────────────────────────────────────
PATCHES_DIR = ROOT / "data" / "patches"
CKPT_PATH   = ROOT / "models" / "unet_resnet34_best.pth"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))

SEED       = 42
TEST_SPLIT = 0.20
THRESHOLD  = 0.5

CLASS_NAMES = ["Undamaged", "Damaged"]


# ── Dataset helpers ────────────────────────────────────────────────────────────
def collect_samples():
    samples = []
    for city in ["kharkiv", "mariupol"]:
        city_dir = PATCHES_DIR / city
        pre_dir  = city_dir / "pre"
        post_dir = city_dir / "post"
        mask_dir = city_dir / "masks"
        if not pre_dir.exists():
            continue
        for fname in sorted(os.listdir(pre_dir)):
            pre_p  = pre_dir  / fname
            post_p = post_dir / fname
            mask_p = mask_dir / fname
            if post_p.exists() and mask_p.exists():
                samples.append((pre_p, post_p, mask_p))
    return samples


def load_sample(pre_p, post_p, mask_p):
    pre  = np.array(Image.open(pre_p),  dtype=np.float32) / 255.0
    post = np.array(Image.open(post_p), dtype=np.float32) / 255.0
    mask = (np.array(Image.open(mask_p), dtype=np.float32) > 127).astype(np.float32)
    x = np.concatenate([pre.transpose(2,0,1), post.transpose(2,0,1)], axis=0)
    return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(mask)


# ── Model ──────────────────────────────────────────────────────────────────────
def load_model():
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None,
        in_channels=6, classes=1, activation=None,
    ).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[eval] Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(val_loss={ckpt['val_loss']:.4f}  val_iou={ckpt['val_iou']:.3f})")
    return model


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(all_preds, all_targets):
    """
    all_preds, all_targets: flat numpy bool/int arrays.
    Returns dict of per-class and overall metrics.
    """
    p = all_preds.astype(bool)
    t = all_targets.astype(bool)
    metrics = {}

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        if cls_idx == 0:
            pred_cls   = ~p
            target_cls = ~t
        else:
            pred_cls   = p
            target_cls = t

        tp = (pred_cls &  target_cls).sum()
        fp = (pred_cls & ~target_cls).sum()
        fn = (~pred_cls & target_cls).sum()
        tn = (~pred_cls & ~target_cls).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        metrics[cls_name] = {
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
            "iou":       float(iou),
            "support":   int(target_cls.sum()),
        }

    overall_acc = (p == t).mean()
    mean_iou    = np.mean([metrics[c]["iou"] for c in CLASS_NAMES])

    metrics["overall_accuracy"] = float(overall_acc)
    metrics["mean_iou"]         = float(mean_iou)
    return metrics


def build_confusion_matrix(all_preds, all_targets):
    """Returns 2x2 confusion matrix [[TN, FP], [FN, TP]]."""
    p = all_preds.astype(bool)
    t = all_targets.astype(bool)
    tn = (~p & ~t).sum()
    fp = ( p & ~t).sum()
    fn = (~p &  t).sum()
    tp = ( p &  t).sum()
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0d0f14")
    ax.set_facecolor("#161b27")

    total = cm.sum()
    norm  = cm / total

    im = ax.imshow(norm, cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(2):
        for j in range(2):
            pct   = norm[i, j] * 100
            count = cm[i, j]
            ax.text(j, i, f"{pct:.1f}%\n({count:,})",
                    ha="center", va="center", fontsize=11,
                    color="white" if norm[i, j] < 0.6 else "black")

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES, color="white", fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, color="white", fontsize=11)
    ax.set_xlabel("Predicted", color="white", fontsize=12)
    ax.set_ylabel("Actual",    color="white", fontsize=12)
    ax.set_title("U-Net Confusion Matrix (Test Set)", color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3248")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[eval] Saved confusion matrix → {save_path}")


# ── Report ─────────────────────────────────────────────────────────────────────
def format_report(metrics, test_size, cm):
    lines = []
    lines.append("=" * 60)
    lines.append("U-NET EVALUATION REPORT")
    lines.append(f"Checkpoint : {CKPT_PATH.name}")
    lines.append(f"Test split : {test_size} patches  (seed={SEED}, {int(TEST_SPLIT*100)}% holdout)")
    lines.append(f"Device     : {DEVICE}")
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
    lines.append("=" * 60)
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # 1. Collect and split samples
    samples = collect_samples()
    print(f"[eval] Total patches: {len(samples)}")

    rng = random.Random(SEED)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n_test  = max(1, int(len(shuffled) * TEST_SPLIT))
    test_s  = shuffled[:n_test]
    train_s = shuffled[n_test:]
    print(f"[eval] Test: {len(test_s)}  Train (excluded): {len(train_s)}")

    # 2. Load model
    model = load_model()

    # 3. Run inference
    all_preds   = []
    all_targets = []

    print(f"[eval] Running inference on {len(test_s)} test patches …")
    with torch.no_grad():
        for i, (pre_p, post_p, mask_p) in enumerate(test_s):
            x, mask = load_sample(pre_p, post_p, mask_p)
            x = x.to(DEVICE)
            logit = model(x).squeeze().cpu()
            pred  = (torch.sigmoid(logit) > THRESHOLD).numpy().astype(np.uint8)
            tgt   = mask.numpy().astype(np.uint8)
            all_preds.append(pred.ravel())
            all_targets.append(tgt.ravel())
            print(f"  [{i+1}/{len(test_s)}] {Path(pre_p).name}  "
                  f"damage_px={tgt.sum()}  pred_px={pred.sum()}")

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # 4. Compute metrics
    metrics = compute_metrics(all_preds, all_targets)
    cm      = build_confusion_matrix(all_preds, all_targets)

    # 5. Save confusion matrix
    plot_confusion_matrix(cm, RESULTS_DIR / "confusion_matrix.png")

    # 6. Format and save report
    report = format_report(metrics, len(test_s), cm)
    report_path = RESULTS_DIR / "eval_report.txt"
    report_path.write_text(report)
    print(f"[eval] Saved report → {report_path}")

    # 7. Print summary
    print("\n" + report)


if __name__ == "__main__":
    main()
