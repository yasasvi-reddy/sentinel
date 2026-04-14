"""
train_vit.py — TemporalViT damage classification

Input : stack(pre_prob.npy, post_prob.npy)  →  (2, 256, 256)
Target: patch-level class
          0 = undamaged      (mask empty)
          1 = newly damaged  (mask non-empty, pre_prob low)
          2 = pre-existing   (mask non-empty, pre_prob elevated)
Loss  : weighted CrossEntropyLoss
Saves : models/temporal_vit_best.pth  (best val accuracy)

Run:
    python src/train_vit.py
"""

import os, sys, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT    = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from temporal_vit import TemporalViT

PATCHES_DIR = ROOT / "data" / "patches"
MODEL_OUT   = ROOT / "models" / "temporal_vit_best.pth"
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

DEVICE = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))

EPOCHS    = 40
BATCH     = 16
LR        = 1e-3
VAL_SPLIT = 0.15
SEED      = 42

# Thresholds for labelling
DAMAGE_PIXEL_THR = 0.05   # below this → undamaged (class 0)
PRE_EXIST_THR    = 0.10   # above this → pre-existing (class 2); between → newly damaged (class 1)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────
class ViTDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pre_prob_p, post_prob_p, mask_p, label = self.samples[idx]
        pre_prob  = np.load(pre_prob_p).astype(np.float32)
        post_prob = np.load(post_prob_p).astype(np.float32)

        if self.augment:
            if random.random() > 0.5:
                pre_prob  = pre_prob[:, ::-1].copy()
                post_prob = post_prob[:, ::-1].copy()
            if random.random() > 0.5:
                pre_prob  = pre_prob[::-1, :].copy()
                post_prob = post_prob[::-1, :].copy()
            k = random.randint(0, 3)
            if k:
                pre_prob  = np.rot90(pre_prob,  k).copy()
                post_prob = np.rot90(post_prob, k).copy()

        x = np.stack([pre_prob, post_prob], axis=0)   # (2, H, W)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)


def derive_label(mask_path: Path, pre_prob_path: Path) -> int:
    """Return 0 (undamaged), 1 (newly damaged), or 2 (pre-existing / heavy damage)."""
    mask = np.array(Image.open(mask_path), dtype=np.float32)
    damage_frac = (mask > 127).mean()
    if damage_frac < DAMAGE_PIXEL_THR:
        return 0
    return 2 if damage_frac > PRE_EXIST_THR else 1


def collect_samples():
    samples = []
    for city in ["kharkiv", "mariupol"]:
        city_dir    = PATCHES_DIR / city
        pre_prob_dir  = city_dir / "pre_prob"
        post_prob_dir = city_dir / "post_prob"
        mask_dir      = city_dir / "masks"
        if not pre_prob_dir.exists():
            continue
        for fname in sorted(os.listdir(pre_prob_dir)):
            stem = fname.replace(".npy", "")
            pre_prob_p  = pre_prob_dir  / fname
            post_prob_p = post_prob_dir / fname
            mask_p      = mask_dir / (stem + ".png")
            if post_prob_p.exists() and mask_p.exists():
                label = derive_label(mask_p, pre_prob_p)
                samples.append((pre_prob_p, post_prob_p, mask_p, label))
    return samples


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    samples = collect_samples()
    print(f"[train_vit] Total patches: {len(samples)}")

    counts = [0, 0, 0]
    for *_, lbl in samples:
        counts[lbl] += 1
    print(f"[train_vit] Class distribution — undamaged:{counts[0]}  "
          f"newly_damaged:{counts[1]}  pre_existing:{counts[2]}")

    random.shuffle(samples)
    n_val  = max(1, int(len(samples) * VAL_SPLIT))
    val_s, train_s = samples[:n_val], samples[n_val:]
    print(f"[train_vit] Train: {len(train_s)}  Val: {len(val_s)}  Device: {DEVICE}")

    train_dl = DataLoader(ViTDataset(train_s, augment=True),  batch_size=BATCH, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(ViTDataset(val_s,   augment=False), batch_size=BATCH, shuffle=False, num_workers=0)

    model = TemporalViT(
        img_size=256, patch_size=16, in_channels=2,
        num_classes=3, embed_dim=192, depth=4,
        num_heads=6, mlp_ratio=4.0, dropout=0.1,
    ).to(DEVICE)

    # Class weights to handle imbalance
    total = sum(counts)
    weights = torch.tensor([total / max(c, 1) for c in counts], dtype=torch.float32).to(DEVICE)
    weights = weights / weights.sum()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ce_fn     = nn.CrossEntropyLoss(weight=weights)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # — Train —
        model.train()
        t_loss, t_correct, t_n = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss   = ce_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss    += loss.item() * x.size(0)
            t_correct += (logits.argmax(1) == y).sum().item()
            t_n       += x.size(0)
        scheduler.step()

        # — Validate —
        model.eval()
        v_correct, v_n = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)
                v_correct += (pred == y).sum().item()
                v_n       += x.size(0)

        train_acc = t_correct / t_n
        val_acc   = v_correct / v_n

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={t_loss/t_n:.4f}  "
              f"train_acc={train_acc:.3f}  "
              f"val_acc={val_acc:.3f}", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": val_acc}, MODEL_OUT)
            print(f"  ✓ saved best checkpoint (val_acc={val_acc:.3f})", flush=True)

    print(f"\n[train_vit] Done. Best val_acc={best_acc:.3f}  →  {MODEL_OUT}")


if __name__ == "__main__":
    train()
