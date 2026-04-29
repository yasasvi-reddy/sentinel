"""
plot_loss_curves.py — Replay both training runs to capture loss history, then plot.

Reruns original (30 epochs, unbalanced) and balanced (35 epochs, oversampled)
training configurations with the same seeds. Does NOT overwrite existing checkpoints.
Saves loss history to results/loss_history.json and plot to results/loss_curves.png.

Run:
    python src/plot_loss_curves.py
"""

import os, sys, random, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
import segmentation_models_pytorch as smp

PATCHES_DIR = ROOT / "data" / "patches"
RESULTS     = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))

SEED       = 42
TEST_SPLIT = 0.20
BATCH      = 8
LR         = 3e-4


# ── Dataset ───────────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        _, pre_p, post_p, mask_p = self.samples[idx]
        pre  = np.array(Image.open(pre_p),  dtype=np.float32) / 255.0
        post = np.array(Image.open(post_p), dtype=np.float32) / 255.0
        mask = (np.array(Image.open(mask_p), dtype=np.float32) > 127).astype(np.float32)
        if self.augment:
            if random.random() > 0.5:
                pre = pre[:, ::-1, :].copy(); post = post[:, ::-1, :].copy(); mask = mask[:, ::-1].copy()
            if random.random() > 0.5:
                pre = pre[::-1, :, :].copy(); post = post[::-1, :, :].copy(); mask = mask[::-1, :].copy()
            k = random.randint(0, 3)
            if k:
                pre = np.rot90(pre, k).copy(); post = np.rot90(post, k).copy(); mask = np.rot90(mask, k).copy()
        x = np.concatenate([pre.transpose(2,0,1), post.transpose(2,0,1)], axis=0)
        return torch.from_numpy(x), torch.from_numpy(mask).unsqueeze(0)


def collect_all():
    samples = []
    for city in ["kharkiv", "mariupol"]:
        city_dir = PATCHES_DIR / city
        pre_dir = city_dir / "pre"; post_dir = city_dir / "post"; mask_dir = city_dir / "masks"
        if not pre_dir.exists(): continue
        for fname in sorted(os.listdir(pre_dir)):
            pre_p = pre_dir / fname; post_p = post_dir / fname; mask_p = mask_dir / fname
            if post_p.exists() and mask_p.exists():
                samples.append((city, pre_p, post_p, mask_p))
    return samples


def split_samples(samples):
    rng = random.Random(SEED)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * TEST_SPLIT))
    return shuffled[n_test:], shuffled[:n_test]   # train, test


def balance(train_s):
    kharkiv_s  = [s for s in train_s if s[0] == "kharkiv"]
    mariupol_s = [s for s in train_s if s[0] == "mariupol"]
    target     = len(kharkiv_s)
    oversampled = (mariupol_s * (target // len(mariupol_s) + 1))[:target]
    balanced    = kharkiv_s + oversampled
    random.shuffle(balanced)
    return balanced


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    fp = pred.view(-1); ft = target.view(-1)
    inter = (fp * ft).sum()
    return 1.0 - (2.0 * inter + smooth) / (fp.sum() + ft.sum() + smooth)


# ── Single training run — returns (train_losses, val_losses) ──────────────────
def run_training(train_s, val_s, epochs, label):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    train_dl = DataLoader(PatchDataset(train_s, augment=True),
                          batch_size=BATCH, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(PatchDataset(val_s,   augment=False),
                          batch_size=BATCH, shuffle=False, num_workers=0)

    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None,
        in_channels=6, classes=1, activation=None,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bce_fn    = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(DEVICE))

    train_losses, val_losses, val_ious = [], [], []

    print(f"\n[loss] Training '{label}' for {epochs} epochs on {DEVICE} …")
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = bce_fn(pred, y) + dice_loss(pred, y)
            loss.backward(); optimizer.step()
            t_loss += loss.item() * x.size(0); t_n += x.size(0)
        scheduler.step()

        model.eval()
        v_loss, v_n, v_iou = 0.0, 0, 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred  = model(x)
                loss  = bce_fn(pred, y) + dice_loss(pred, y)
                v_loss += loss.item() * x.size(0); v_n += x.size(0)
                p = (torch.sigmoid(pred) > 0.5).float()
                inter = (p * y).sum().item(); union = (p + y).clamp(0,1).sum().item()
                v_iou += inter / max(union, 1)

        tl = t_loss / t_n; vl = v_loss / v_n; vi = v_iou / len(val_dl)
        train_losses.append(tl); val_losses.append(vl); val_ious.append(vi)
        print(f"  Epoch {epoch:3d}/{epochs}  train={tl:.4f}  val={vl:.4f}  iou={vi:.3f}", flush=True)

    return train_losses, val_losses, val_ious


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_curves(history, save_path):
    DARK_BG   = "#0d0f14"
    PANEL_BG  = "#161b27"
    GRID_COL  = "#2d3248"
    TEXT_COL  = "#b0b8cc"
    MUTED     = "#606880"

    COLORS = {
        "orig_train": "#4a9eff",
        "orig_val":   "#1a5fa8",
        "bal_train":  "#4dcc88",
        "bal_val":    "#1a7a45",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("U-Net Training History: Baseline vs Balanced",
                 color=TEXT_COL, fontsize=14, y=1.01)

    orig_epochs = list(range(1, len(history["orig"]["train"]) + 1))
    bal_epochs  = list(range(1, len(history["bal"]["train"])  + 1))

    # ── Left panel: Loss curves ────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    ax.plot(orig_epochs, history["orig"]["train"], color=COLORS["orig_train"],
            lw=1.8, label="Baseline — train")
    ax.plot(orig_epochs, history["orig"]["val"],   color=COLORS["orig_val"],
            lw=1.8, linestyle="--", label="Baseline — val")
    ax.plot(bal_epochs,  history["bal"]["train"],  color=COLORS["bal_train"],
            lw=1.8, label="Balanced — train")
    ax.plot(bal_epochs,  history["bal"]["val"],    color=COLORS["bal_val"],
            lw=1.8, linestyle="--", label="Balanced — val")

    # Best val markers
    orig_best_ep = int(np.argmin(history["orig"]["val"])) + 1
    bal_best_ep  = int(np.argmin(history["bal"]["val"]))  + 1
    ax.axvline(orig_best_ep, color=COLORS["orig_val"],  lw=0.8, linestyle=":", alpha=0.7)
    ax.axvline(bal_best_ep,  color=COLORS["bal_val"],   lw=0.8, linestyle=":", alpha=0.7)
    ax.scatter([orig_best_ep], [min(history["orig"]["val"])],
               color=COLORS["orig_val"], s=60, zorder=5)
    ax.scatter([bal_best_ep],  [min(history["bal"]["val"])],
               color=COLORS["bal_val"],  s=60, zorder=5)

    ax.set_title("Loss (BCE + Dice)", color=TEXT_COL, fontsize=11)
    ax.set_xlabel("Epoch", color=MUTED, fontsize=10)
    ax.set_ylabel("Loss",  color=MUTED, fontsize=10)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(which="major", color=GRID_COL, lw=0.6)
    ax.grid(which="minor", color=GRID_COL, lw=0.3, alpha=0.5)
    for spine in ax.spines.values(): spine.set_edgecolor(GRID_COL)
    leg = ax.legend(fontsize=9, facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)

    # Annotation: best val loss
    ax.annotate(f"val={min(history['orig']['val']):.4f}",
                xy=(orig_best_ep, min(history["orig"]["val"])),
                xytext=(orig_best_ep + 1, min(history["orig"]["val"]) + 0.05),
                color=COLORS["orig_val"], fontsize=8, arrowprops=dict(arrowstyle="->", color=COLORS["orig_val"], lw=0.8))
    ax.annotate(f"val={min(history['bal']['val']):.4f}",
                xy=(bal_best_ep, min(history["bal"]["val"])),
                xytext=(bal_best_ep + 1, min(history["bal"]["val"]) + 0.05),
                color=COLORS["bal_val"], fontsize=8, arrowprops=dict(arrowstyle="->", color=COLORS["bal_val"], lw=0.8))

    # ── Right panel: Val IoU curves ────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(PANEL_BG)
    ax.plot(orig_epochs, history["orig"]["iou"], color=COLORS["orig_val"],
            lw=1.8, label=f"Baseline  (best {max(history['orig']['iou']):.3f})")
    ax.plot(bal_epochs,  history["bal"]["iou"],  color=COLORS["bal_val"],
            lw=1.8, label=f"Balanced  (best {max(history['bal']['iou']):.3f})")

    orig_best_iou_ep = int(np.argmax(history["orig"]["iou"])) + 1
    bal_best_iou_ep  = int(np.argmax(history["bal"]["iou"]))  + 1
    ax.scatter([orig_best_iou_ep], [max(history["orig"]["iou"])],
               color=COLORS["orig_val"], s=60, zorder=5)
    ax.scatter([bal_best_iou_ep],  [max(history["bal"]["iou"])],
               color=COLORS["bal_val"],  s=60, zorder=5)

    ax.set_title("Validation IoU", color=TEXT_COL, fontsize=11)
    ax.set_xlabel("Epoch", color=MUTED, fontsize=10)
    ax.set_ylabel("IoU",   color=MUTED, fontsize=10)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(which="major", color=GRID_COL, lw=0.6)
    ax.grid(which="minor", color=GRID_COL, lw=0.3, alpha=0.5)
    for spine in ax.spines.values(): spine.set_edgecolor(GRID_COL)
    ax.legend(fontsize=9, facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"\n[plot] Saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    history_path = RESULTS / "loss_history.json"

    # Load cached history if already computed
    if history_path.exists():
        print(f"[plot] Loading cached loss history from {history_path}")
        history = json.loads(history_path.read_text())
    else:
        samples = collect_all()
        train_s, _ = split_samples(samples)

        # ── Original: unbalanced, 30 epochs ───────────────────────────────
        orig_train, orig_val, orig_iou = run_training(
            train_s, train_s, epochs=30, label="Baseline (unbalanced)"
        )

        # ── Balanced: oversampled Mariupol, 35 epochs ──────────────────────
        bal_train_s = balance(train_s)
        bal_train, bal_val, bal_iou = run_training(
            bal_train_s, train_s, epochs=35, label="Balanced (oversampled)"
        )

        history = {
            "orig": {"train": orig_train, "val": orig_val, "iou": orig_iou},
            "bal":  {"train": bal_train,  "val": bal_val,  "iou": bal_iou},
        }
        history_path.write_text(json.dumps(history, indent=2))
        print(f"[plot] Saved loss history → {history_path}")

    plot_curves(history, RESULTS / "loss_curves.png")

    # Print summary table
    print("\nBest checkpoints:")
    print(f"  Baseline  epoch {int(np.argmin(history['orig']['val']))+1:3d}  "
          f"val_loss={min(history['orig']['val']):.4f}  "
          f"val_iou={max(history['orig']['iou']):.3f}")
    print(f"  Balanced  epoch {int(np.argmin(history['bal']['val']))+1:3d}  "
          f"val_loss={min(history['bal']['val']):.4f}  "
          f"val_iou={max(history['bal']['iou']):.3f}")


if __name__ == "__main__":
    main()
