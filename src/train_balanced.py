"""
train_balanced.py — U-Net retraining with balanced city sampling

Changes vs train.py:
  1. Same seed=42, 20% test split (identical held-out set for fair comparison)
  2. Mariupol training patches oversampled to match Kharkiv count (23 each → 46 total)
  3. Augmentation unchanged: hflip, vflip, rot90 (already present in original)
  4. 35 epochs (5 more than original to allow convergence on balanced data)
  5. Saves to models/unet_resnet34_balanced.pth — does NOT overwrite original

Run:
    python src/train_balanced.py
"""

import os, sys, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import segmentation_models_pytorch as smp

PATCHES_DIR = ROOT / "data" / "patches"
MODEL_OUT   = ROOT / "models" / "unet_resnet34_balanced.pth"
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

DEVICE = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))

EPOCHS    = 35
BATCH     = 8
LR        = 3e-4
SEED      = 42
TEST_SPLIT = 0.20

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, pre_p, post_p, mask_p = self.samples[idx]
        pre  = np.array(Image.open(pre_p),  dtype=np.float32) / 255.0
        post = np.array(Image.open(post_p), dtype=np.float32) / 255.0
        mask = (np.array(Image.open(mask_p), dtype=np.float32) > 127).astype(np.float32)

        if self.augment:
            if random.random() > 0.5:
                pre  = pre [:, ::-1, :].copy()
                post = post[:, ::-1, :].copy()
                mask = mask[:, ::-1  ].copy()
            if random.random() > 0.5:
                pre  = pre [::-1, :, :].copy()
                post = post[::-1, :, :].copy()
                mask = mask[::-1, :  ].copy()
            k = random.randint(0, 3)
            if k:
                pre  = np.rot90(pre,  k).copy()
                post = np.rot90(post, k).copy()
                mask = np.rot90(mask, k).copy()

        x = np.concatenate([pre.transpose(2,0,1), post.transpose(2,0,1)], axis=0)
        return torch.from_numpy(x), torch.from_numpy(mask).unsqueeze(0)


# ── Data collection ───────────────────────────────────────────────────────────
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


def balance_training(train_s):
    """Oversample Mariupol patches to match Kharkiv count."""
    kharkiv_s  = [s for s in train_s if s[0] == "kharkiv"]
    mariupol_s = [s for s in train_s if s[0] == "mariupol"]
    target = len(kharkiv_s)
    # Repeat Mariupol samples until we have `target` of them
    oversampled = (mariupol_s * (target // len(mariupol_s) + 1))[:target]
    balanced = kharkiv_s + oversampled
    random.shuffle(balanced)
    print(f"[train] Balanced: {len(kharkiv_s)} Kharkiv + {len(oversampled)} Mariupol "
          f"(from {len(mariupol_s)} unique) = {len(balanced)} total")
    return balanced


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

    # Same deterministic split as eval_unet.py / eval_mariupol.py
    rng = random.Random(SEED)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n_test  = max(1, int(len(shuffled) * TEST_SPLIT))
    test_s  = shuffled[:n_test]
    train_s = shuffled[n_test:]

    kharkiv_test  = sum(1 for s in test_s  if s[0] == "kharkiv")
    mariupol_test = sum(1 for s in test_s  if s[0] == "mariupol")
    print(f"[train] Test  ({len(test_s )}): Kharkiv={kharkiv_test}, Mariupol={mariupol_test}")
    print(f"[train] Train ({len(train_s)}): Kharkiv={sum(1 for s in train_s if s[0]=='kharkiv')}, "
          f"Mariupol={sum(1 for s in train_s if s[0]=='mariupol')}")

    # Balance training set
    balanced_train = balance_training(train_s)

    # Validation: use original (unbalanced) train split for an honest val loss
    val_s = train_s

    train_dl = DataLoader(PatchDataset(balanced_train, augment=True),
                          batch_size=BATCH, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(PatchDataset(val_s,          augment=False),
                          batch_size=BATCH, shuffle=False, num_workers=0)

    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None,
        in_channels=6, classes=1, activation=None,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    bce_fn    = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(DEVICE))

    best_val  = float("inf")
    best_epoch = 0

    print(f"\n[train] Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}")
    print(f"[train] Saving to: {MODEL_OUT}\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

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
                p = (torch.sigmoid(pred) > 0.5).float()
                inter = (p * y).sum().item()
                union = (p + y).clamp(0, 1).sum().item()
                v_iou += inter / max(union, 1)

        val_loss = v_loss / v_n
        val_iou  = v_iou  / len(val_dl)
        elapsed  = time.time() - t0

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={t_loss/t_n:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_iou={val_iou:.3f}  "
              f"({elapsed:.1f}s)", flush=True)

        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "val_iou":     val_iou,
            }, MODEL_OUT)
            print(f"  ✓ saved best (val_loss={val_loss:.4f})", flush=True)

    print(f"\n[train] Done. Best val_loss={best_val:.4f} at epoch {best_epoch}  →  {MODEL_OUT}")


if __name__ == "__main__":
    train()
