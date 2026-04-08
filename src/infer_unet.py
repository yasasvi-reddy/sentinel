"""
infer_unet.py

Runs the trained U-Net on every patch to generate per-pixel damage
probability maps used as input to the temporal ViT.

Two inference passes per patch:
  pre_prob  — U-Net([pre_rgb, pre_rgb]): same image in both channels;
              the model has no temporal change signal → captures any
              pre-war structural artefacts that look like damage.
  post_prob — U-Net([pre_rgb, post_rgb]): normal inference (trained input);
              captures actual post-war damage.

Outputs saved as float32 .npy files (values in [0, 1]) to:
  data/patches/{city}/pre_prob/patch_NNNNN.npy
  data/patches/{city}/post_prob/patch_NNNNN.npy
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import segmentation_models_pytorch as smp

ROOT      = Path(__file__).parent.parent
PATCH_DIR = ROOT / "data" / "patches"
CKPT_PATH = ROOT / "models" / "unet_resnet34_best.pth"
CITIES    = ["kharkiv", "mariupol"]


def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


def load_model(device):
    ckpt  = torch.load(CKPT_PATH, map_location=device)
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,
        in_channels     = 6,
        classes         = 1,
        activation      = None,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_patch_rgb(path: Path) -> torch.Tensor:
    """Load a PNG patch as float32 (3, H, W) tensor in [0,1]."""
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


@torch.no_grad()
def infer_patch(model, t1_rgb: torch.Tensor, t2_rgb: torch.Tensor,
                device) -> np.ndarray:
    """
    Run model on stacked [t1, t2] → probability map (H, W) in [0,1].
    """
    x = torch.cat([t1_rgb, t2_rgb], dim=0).unsqueeze(0).to(device)  # (1,6,H,W)
    logits = model(x)                                                 # (1,1,H,W)
    prob   = torch.sigmoid(logits).squeeze().cpu().numpy()            # (H,W)
    return prob.astype(np.float32)


def run(city: str, model, device):
    pre_dir   = PATCH_DIR / city / "pre"
    post_dir  = PATCH_DIR / city / "post"
    pre_out   = PATCH_DIR / city / "pre_prob";  pre_out.mkdir(exist_ok=True)
    post_out  = PATCH_DIR / city / "post_prob"; post_out.mkdir(exist_ok=True)

    patch_ids = sorted(p.stem for p in pre_dir.glob("*.png"))
    print(f"  {city}: {len(patch_ids)} patches")

    for pid in patch_ids:
        pre_rgb  = load_patch_rgb(pre_dir  / f"{pid}.png")
        post_rgb = load_patch_rgb(post_dir / f"{pid}.png")

        # Pre-war: same image in both U-Net channels (no temporal change signal)
        pre_prob = infer_patch(model, pre_rgb, pre_rgb, device)
        # Post-war: normal [pre, post] inference (this is the trained setup)
        post_prob = infer_patch(model, pre_rgb, post_rgb, device)

        np.save(pre_out  / f"{pid}.npy", pre_prob)
        np.save(post_out / f"{pid}.npy", post_prob)


if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")
    model  = load_model(device)
    print(f"U-Net loaded from {CKPT_PATH.name}")

    for city in CITIES:
        run(city, model, device)

    print("Done.")
