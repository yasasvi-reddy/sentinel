"""
dataset.py

PyTorch Dataset for Sentinel-2 damage detection.
Loads pre/post image patch pairs and binary damage masks.
Input: 6-channel tensor (pre RGB + post RGB stacked).
Target: binary mask (0=undamaged, 1=damaged).
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


PATCH_DIR = Path(__file__).parent.parent / "data" / "patches"


def get_patch_ids(city: str):
    """Return sorted list of patch IDs that have pre, post, and mask files."""
    mask_dir = PATCH_DIR / city / "masks"
    ids = []
    for m in sorted(mask_dir.glob("*.png")):
        pid = m.stem
        if (PATCH_DIR / city / "pre"  / f"{pid}.png").exists() and \
           (PATCH_DIR / city / "post" / f"{pid}.png").exists():
            ids.append((city, pid))
    return ids


def build_transforms(augment: bool):
    if augment:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=15, p=0.4),
            A.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.1, hue=0.05, p=0.4),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            ToTensorV2(),
        ], additional_targets={"image2": "image"})
    else:
        return A.Compose([
            ToTensorV2(),
        ], additional_targets={"image2": "image"})


class DamageDataset(Dataset):
    """
    Returns:
        image  : float32 tensor (6, H, W)  — pre RGB + post RGB, [0, 1]
        mask   : float32 tensor (1, H, W)  — binary damage mask
    """

    def __init__(self, samples: list[tuple[str, str]], augment: bool = False):
        self.samples   = samples
        self.transform = build_transforms(augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        city, pid = self.samples[idx]

        pre  = np.array(Image.open(PATCH_DIR / city / "pre"   / f"{pid}.png").convert("RGB"),
                        dtype=np.float32) / 255.0
        post = np.array(Image.open(PATCH_DIR / city / "post"  / f"{pid}.png").convert("RGB"),
                        dtype=np.float32) / 255.0
        mask = np.array(Image.open(PATCH_DIR / city / "masks" / f"{pid}.png").convert("L"),
                        dtype=np.float32) / 255.0  # {0, 1}

        out  = self.transform(image=pre, image2=post, mask=mask)
        pre_t, post_t = out["image"], out["image2"]
        mask_t        = out["mask"]

        image = torch.cat([pre_t, post_t], dim=0).float()   # (6, H, W)
        mask  = mask_t.unsqueeze(0).float()                  # (1, H, W)
        return image, mask


def make_splits(val_frac: float = 0.2, seed: int = 42):
    """Collect all patch IDs, shuffle, split into train/val."""
    all_samples = []
    for city in ["kharkiv", "mariupol"]:
        all_samples.extend(get_patch_ids(city))

    rng = random.Random(seed)
    rng.shuffle(all_samples)

    n_val  = max(1, int(len(all_samples) * val_frac))
    val    = all_samples[:n_val]
    train  = all_samples[n_val:]
    return train, val
