"""
train.py — U-Net binary damage segmentation

Input : concat(pre_rgb, post_rgb) normalised to [0,1]  →  (6, 256, 256)
Target: binary mask from masks/  →  0 = undamaged, 1 = damaged
Loss  : BCEWithLogitsLoss + soft-Dice
Saves : models/unet_resnet34_best.pth  (best val loss)

Run:
    python src/train.py
"""

import os, sys, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT      = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import segmentation_models_pytorch as smp

PATCHES_DIR = ROOT / "data" / "patches"
MODEL_OUT   = ROOT / "models" / "unet_resnet34_best.pth"
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

DEVICE = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))

EPOCHS     = 30
BATCH      = 8
LR         = 3e-4
VAL_SPLIT  = 0.15
SEED       = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pre_p, post_p, mask_p = self.samples[idx]
        pre  = np.array(Image.open(pre_p),  dtype=np.float32) / 255.0
        post = np.array(Image.open(post_p), dtype=np.float32) / 255.0
        mask = (np.array(Image.open(mask_p), dtype=np.float32) > 127).astype(np.float32)

        # Random augmentations
        if self.augment:
            if random.random() > 0.5:
                pre  = pre[:, ::-1, :].copy()
                post = post[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
            if random.random() > 0.5:
                pre  = pre[::-1, :, :].copy()
                post = post[::-1, :, :].copy()
                mask = mask[::-1, :].copy()
            k = random.randint(0, 3)
            if k:
                pre  = np.rot90(pre,  k).copy()
                post = np.rot90(post, k).copy()
                mask = np.rot90(mask, k).copy()

        # (6, H, W) input
        x = np.concatenate([pre.transpose(2,0,1), post.transpose(2,0,1)], axis=0)
        return torch.from_numpy(x), torch.from_numpy(mask).unsqueeze(0)


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
            stem = fname.replace(".png", "")
            pre_p  = pre_dir  / fname
            post_p = post_dir / fname
            mask_p = mask_dir / fname
            if post_p.exists() and mask_p.exists():
                samples.append((pre_p, post_p, mask_p))
    return samples


# ── Dice loss ─────────────────────────────────────────────────────────────────
def dice_loss(pred, target, smooth=1.0):
    pred   = torch.sigmoid(pred)
    flat_p = pred.view(-1)
    flat_t = target.view(-1)
    inter  = (flat_p * flat_t).sum()
    return 1.0 - (2.0 * inter + smooth) / (flat_p.sum() + flat_t.sum() + smooth)


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    samples = collect_samples()
    print(f"[train] Total patches: {len(samples)}")
    random.shuffle(samples)
    n_val  = max(1, int(len(samples) * VAL_SPLIT))
    val_s, train_s = samples[:n_val], samples[n_val:]
    print(f"[train] Train: {len(train_s)}  Val: {len(val_s)}  Device: {DEVICE}")

    train_dl = DataLoader(PatchDataset(train_s, augment=True),  batch_size=BATCH, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(PatchDataset(val_s,   augment=False), batch_size=BATCH, shuffle=False, num_workers=0)

    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None,
        in_channels=6, classes=1, activation=None,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    bce_fn    = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(DEVICE))

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # — Train —
        model.train()
        t_loss, t_n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = bce_fn(pred, y) + dice_loss(pred, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * x.size(0)
            t_n    += x.size(0)
        scheduler.step()

        # — Validate —
        model.eval()
        v_loss, v_n, v_iou = 0.0, 0, 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y  = x.to(DEVICE), y.to(DEVICE)
                pred  = model(x)
                loss  = bce_fn(pred, y) + dice_loss(pred, y)
                v_loss += loss.item() * x.size(0)
                v_n    += x.size(0)
                # IoU
                p = (torch.sigmoid(pred) > 0.5).float()
                inter = (p * y).sum().item()
                union = (p + y).clamp(0,1).sum().item()
                v_iou += inter / max(union, 1)
        val_loss = v_loss / v_n
        val_iou  = v_iou / len(val_dl)

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={t_loss/t_n:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_iou={val_iou:.3f}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": val_loss, "val_iou": val_iou}, MODEL_OUT)
            print(f"  ✓ saved best checkpoint (val_loss={val_loss:.4f})", flush=True)

    print(f"\n[train] Done. Best val_loss={best_val:.4f}  →  {MODEL_OUT}")


if __name__ == "__main__":
    train()
