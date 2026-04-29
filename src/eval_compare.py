"""
eval_compare.py — Side-by-side U-Net evaluation: original vs balanced checkpoint

Uses the identical test split (seed=42, 20% holdout) as eval_unet.py.
Reports per-class and overall metrics for:
  - Kharkiv test patches
  - Mariupol test patches
  - Combined

Saves:
  results/eval_compare_report.txt

Run:
    python src/eval_compare.py
"""

import os, sys, random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import segmentation_models_pytorch as smp

PATCHES_DIR  = ROOT / "data" / "patches"
CKPT_ORIG    = ROOT / "models" / "unet_resnet34_best.pth"
CKPT_NEW     = ROOT / "models" / "unet_resnet34_balanced.pth"
RESULTS      = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE     = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
SEED       = 42
TEST_SPLIT = 0.20
THRESHOLD  = 0.5
CLASS_NAMES = ["Undamaged", "Damaged"]


# ── Helpers ───────────────────────────────────────────────────────────────────
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
                samples.append((city, pre_p, post_p, mask_p))
    return samples


def get_test_split(samples):
    rng = random.Random(SEED)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * TEST_SPLIT))
    return shuffled[:n_test]


def load_model(ckpt_path):
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None,
        in_channels=6, classes=1, activation=None,
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def load_patch(pre_p, post_p, mask_p):
    pre  = np.array(Image.open(pre_p),  dtype=np.float32) / 255.0
    post = np.array(Image.open(post_p), dtype=np.float32) / 255.0
    mask = (np.array(Image.open(mask_p), dtype=np.float32) > 127).astype(np.float32)
    x = np.concatenate([pre.transpose(2,0,1), post.transpose(2,0,1)], axis=0)
    return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(mask)


def run_inference(model, samples):
    preds, targets = [], []
    with torch.no_grad():
        for _, pre_p, post_p, mask_p in samples:
            x, mask = load_patch(pre_p, post_p, mask_p)
            logit = model(x.to(DEVICE)).squeeze().cpu()
            pred  = (torch.sigmoid(logit) > THRESHOLD).numpy().astype(np.uint8)
            preds.append(pred.ravel())
            targets.append(mask.numpy().astype(np.uint8).ravel())
    return np.concatenate(preds), np.concatenate(targets)


def compute_metrics(preds, targets):
    p = preds.astype(bool); t = targets.astype(bool)
    metrics = {}
    for idx, name in enumerate(CLASS_NAMES):
        pc = ~p if idx == 0 else p
        tc = ~t if idx == 0 else t
        tp = (pc &  tc).sum(); fp = (pc & ~tc).sum(); fn = (~pc & tc).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2*precision*recall / (precision+recall)) if (precision+recall) > 0 else 0.0
        iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        metrics[name] = dict(precision=float(precision), recall=float(recall),
                             f1=float(f1), iou=float(iou))
    metrics["mean_iou"]         = float(np.mean([metrics[c]["iou"] for c in CLASS_NAMES]))
    metrics["overall_accuracy"] = float((p == t).mean())
    return metrics


# ── Report formatting ─────────────────────────────────────────────────────────
def fmt(v): return f"{v:.4f}"

def delta(new_v, old_v):
    d = new_v - old_v
    sign = "+" if d >= 0 else ""
    return f"({sign}{d:.4f})"

def section_header(title, n_patches):
    return [
        "",
        f"{'─'*70}",
        f"  {title}  [{n_patches} test patches]",
        f"{'─'*70}",
        f"  {'Metric':<22} {'Original':>10} {'Balanced':>10} {'Δ':>10}",
        f"  {'─'*52}",
    ]

def section_rows(m_old, m_new):
    rows = []
    for cls in CLASS_NAMES:
        for metric in ["iou", "precision", "recall", "f1"]:
            label = f"{cls} {metric.upper()}"
            o = m_old[cls][metric]; n = m_new[cls][metric]
            rows.append(f"  {label:<22} {fmt(o):>10} {fmt(n):>10} {delta(n,o):>10}")
    rows.append(f"  {'─'*52}")
    rows.append(f"  {'Mean IoU':<22} {fmt(m_old['mean_iou']):>10} {fmt(m_new['mean_iou']):>10} {delta(m_new['mean_iou'],m_old['mean_iou']):>10}")
    rows.append(f"  {'Overall Accuracy':<22} {fmt(m_old['overall_accuracy']):>10} {fmt(m_new['overall_accuracy']):>10} {delta(m_new['overall_accuracy'],m_old['overall_accuracy']):>10}")
    return rows


def build_report(results, ckpt_orig_meta, ckpt_new_meta):
    lines = []
    lines.append("=" * 70)
    lines.append("U-NET EVALUATION: ORIGINAL vs BALANCED CHECKPOINT")
    lines.append("=" * 70)
    lines.append(f"  Original  : {CKPT_ORIG.name}  (epoch {ckpt_orig_meta['epoch']}, val_loss={ckpt_orig_meta['val_loss']:.4f}, val_iou={ckpt_orig_meta['val_iou']:.3f})")
    lines.append(f"  Balanced  : {CKPT_NEW.name}   (epoch {ckpt_new_meta['epoch']}, val_loss={ckpt_new_meta['val_loss']:.4f}, val_iou={ckpt_new_meta['val_iou']:.3f})")
    lines.append(f"  Test split: seed={SEED}, {int(TEST_SPLIT*100)}% holdout (same split for both)")
    lines.append(f"  Device    : {DEVICE}")

    for title, (n, m_old, m_new) in results.items():
        lines.extend(section_header(title, n))
        lines.extend(section_rows(m_old, m_new))

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    samples  = collect_samples()
    test_all = get_test_split(samples)

    kharkiv_test  = [s for s in test_all if s[0] == "kharkiv"]
    mariupol_test = [s for s in test_all if s[0] == "mariupol"]

    print(f"[eval] Test split — Kharkiv: {len(kharkiv_test)}, Mariupol: {len(mariupol_test)}, Total: {len(test_all)}")
    print(f"[eval] Loading checkpoints …")

    model_orig, meta_orig = load_model(CKPT_ORIG)
    model_new,  meta_new  = load_model(CKPT_NEW)

    print(f"[eval] Original : epoch {meta_orig['epoch']}, val_loss={meta_orig['val_loss']:.4f}")
    print(f"[eval] Balanced : epoch {meta_new['epoch']},  val_loss={meta_new['val_loss']:.4f}")

    subsets = {
        "KHARKIV":  kharkiv_test,
        "MARIUPOL": mariupol_test,
        "COMBINED": test_all,
    }

    results = {}
    for name, subset in subsets.items():
        if not subset:
            continue
        print(f"\n[eval] Running inference — {name} ({len(subset)} patches) …")
        p_orig, t_orig = run_inference(model_orig, subset)
        p_new,  t_new  = run_inference(model_new,  subset)
        m_old = compute_metrics(p_orig, t_orig)
        m_new = compute_metrics(p_new,  t_new)
        results[name] = (len(subset), m_old, m_new)
        print(f"  Original  — Mean IoU={m_old['mean_iou']:.4f}  Acc={m_old['overall_accuracy']:.4f}")
        print(f"  Balanced  — Mean IoU={m_new['mean_iou']:.4f}  Acc={m_new['overall_accuracy']:.4f}")

    report = build_report(results, meta_orig, meta_new)
    out_path = RESULTS / "eval_compare_report.txt"
    out_path.write_text(report)
    print(f"\n[eval] Saved → {out_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
