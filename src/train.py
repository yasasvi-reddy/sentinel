"""
train.py

Training pipeline for Sentinel-2 war damage segmentation.

Model   : U-Net with pretrained ResNet-34 encoder (segmentation_models_pytorch)
Input   : 6-channel (pre-war RGB + post-war RGB), 256×256 patches
Target  : binary damage mask
Loss    : Focal loss + Dice loss (combined for class imbalance)
Device  : MPS (Apple Silicon) → CUDA → CPU
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import make_splits, DamageDataset

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
MODEL_DIR  = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

CKPT_PATH  = MODEL_DIR / "unet_resnet34_best.pth"
CURVE_PATH = RESULTS_DIR / "loss_curves.png"

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS     = 60
BATCH_SIZE = 8
LR         = 3e-4
VAL_FRAC   = 0.2
SEED       = 42

# Focal loss params
FOCAL_ALPHA = 0.75   # weight for positive class (damaged) — higher because it's minority
FOCAL_GAMMA = 2.0

# Loss blend
FOCAL_WEIGHT = 0.6
DICE_WEIGHT  = 0.4


# ── Device ─────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Loss functions ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt   = torch.exp(-bce)
        # alpha weights: alpha for positives, (1-alpha) for negatives
        at   = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = at * (1 - pt) ** self.gamma * bce
        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        dice = (2 * intersection + self.smooth) / \
               (probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
        self.dice  = DiceLoss()

    def forward(self, logits, targets):
        return FOCAL_WEIGHT * self.focal(logits, targets) + \
               DICE_WEIGHT  * self.dice(logits, targets)


# ── Metrics ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_metrics(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    iou       = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}


# ── Training loop ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_metrics = {"iou": [], "f1": [], "precision": [], "recall": []}
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss   = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)
        m = compute_metrics(logits, masks)
        for k, v in m.items():
            all_metrics[k].append(v)
    avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    return total_loss / len(loader.dataset), avg_metrics


# ── Plot loss curves ───────────────────────────────────────────────────────────
def plot_curves(train_losses, val_losses, val_ious, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label="Train loss", color="#2196F3")
    ax1.plot(epochs, val_losses,   label="Val loss",   color="#F44336")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss  (Focal + Dice)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_ious, label="Val IoU", color="#4CAF50")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("IoU")
    ax2.set_title("Validation IoU")
    ax2.legend(); ax2.grid(alpha=0.3)

    best_epoch = int(np.argmin(val_losses)) + 1
    ax1.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6,
                label=f"Best (ep {best_epoch})")
    ax1.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved loss curves → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ids, val_ids = make_splits(val_frac=VAL_FRAC, seed=SEED)
    print(f"Train: {len(train_ids)}  |  Val: {len(val_ids)}")

    train_ds = DamageDataset(train_ids, augment=True)
    val_ds   = DamageDataset(val_ids,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = "imagenet",
        in_channels     = 6,        # pre RGB + post RGB
        classes         = 1,        # binary damage mask
        activation      = None,     # raw logits — loss handles sigmoid
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Optimizer & scheduler ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    criterion = CombinedLoss().to(device)

    # ── Training ───────────────────────────────────────────────────────────────
    train_losses, val_losses, val_ious = [], [], []
    best_val_loss = float("inf")
    best_epoch    = 0

    print(f"\n{'Epoch':>5}  {'Train':>8}  {'Val':>8}  {'IoU':>6}  {'F1':>6}  {'LR':>8}  {'Time':>6}")
    print("─" * 60)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss           = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_metrics["iou"])

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        marker  = " ✓" if val_loss < best_val_loss else ""

        print(f"{epoch:>5}  {train_loss:>8.4f}  {val_loss:>8.4f}  "
              f"{val_metrics['iou']:>6.3f}  {val_metrics['f1']:>6.3f}  "
              f"{lr_now:>8.2e}  {elapsed:>5.1f}s{marker}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":   val_loss,
                "val_iou":    val_metrics["iou"],
                "val_f1":     val_metrics["f1"],
                "hparams": {
                    "encoder":     "resnet34",
                    "in_channels": 6,
                    "focal_alpha": FOCAL_ALPHA,
                    "focal_gamma": FOCAL_GAMMA,
                    "lr":          LR,
                    "batch_size":  BATCH_SIZE,
                },
            }, CKPT_PATH)

    print(f"\nBest checkpoint: epoch {best_epoch}  "
          f"val_loss={best_val_loss:.4f}  "
          f"val_iou={val_ious[best_epoch-1]:.3f}")
    print(f"Saved → {CKPT_PATH}")

    plot_curves(train_losses, val_losses, val_ious, CURVE_PATH)


if __name__ == "__main__":
    main()
