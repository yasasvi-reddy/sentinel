"""
temporal_vit.py

Compact Vision Transformer for temporal damage change detection.

Architecture
────────────
Input  : (B, 2, 256, 256)  — [pre_prob, post_prob] damage maps from U-Net
Output : (B, 3, 256, 256)  — per-pixel class logits

Classes
  0 : Undamaged   — UNOSAT = 0
  1 : Newly damaged — UNOSAT = 1  AND  pre_prob < PRE_THRESH
  2 : Pre-existing  — UNOSAT = 1  AND  pre_prob ≥ PRE_THRESH
      (areas the U-Net was already activating on before the invasion;
       captures stable structural features that resemble damage signal)

Design choices
  • Patch size 16 → 16×16 = 256 sequence tokens per image
  • Embed dim 192, depth 4, heads 6  (compact for 391-patch dataset)
  • Dense prediction head: each token → 3 × 16 × 16 logits → reshape
  • Learnable 2D positional embeddings
"""

import math
import torch
import torch.nn as nn


PRE_THRESH = 0.40   # pre_prob threshold for "pre-existing" label derivation


# ── Label derivation ──────────────────────────────────────────────────────────
def derive_labels(pre_prob: torch.Tensor,
                  unosat_mask: torch.Tensor,
                  threshold: float = PRE_THRESH) -> torch.Tensor:
    """
    Args:
        pre_prob    : (H, W) float in [0,1] — U-Net output on pre-war imagery
        unosat_mask : (H, W) binary float  — UNOSAT damage ground truth
        threshold   : pre-war probability above which → 'pre-existing'
    Returns:
        labels : (H, W) long  — 0=undamaged, 1=newly damaged, 2=pre-existing
    """
    mask = unosat_mask.bool()
    pre_high = pre_prob >= threshold

    labels = torch.zeros_like(unosat_mask, dtype=torch.long)
    labels[mask & ~pre_high] = 1   # UNOSAT=1 & pre low  → newly damaged
    labels[mask &  pre_high] = 2   # UNOSAT=1 & pre high → pre-existing
    return labels


# ── Patch embedding ───────────────────────────────────────────────────────────
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=2, patch_size=16, embed_dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) → (B, embed_dim, H/P, W/P) → (B, N, embed_dim)
        x = self.proj(x)
        B, E, Hg, Wg = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B, N, E)
        return x, Hg, Wg


# ── Transformer block ─────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                            dropout=dropout, batch_first=True)
        self.norm2  = nn.LayerNorm(embed_dim)
        hidden      = int(embed_dim * mlp_ratio)
        self.mlp    = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        n = self.norm1(x)
        x = x + self.attn(n, n, n)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ── Dense prediction head ─────────────────────────────────────────────────────
class DenseHead(nn.Module):
    """
    Each patch token → 3 × patch_size^2 values → reshape to (B, 3, H, W).
    """
    def __init__(self, embed_dim, patch_size, num_classes=3):
        super().__init__()
        self.patch_size  = patch_size
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes * patch_size * patch_size),
        )

    def forward(self, tokens, Hg, Wg):
        # tokens: (B, N, E)
        B, N, _ = tokens.shape
        P        = self.patch_size
        C        = self.num_classes

        out = self.head(tokens)                         # (B, N, C*P*P)
        out = out.view(B, Hg, Wg, C, P, P)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
        out = out.view(B, C, Hg * P, Wg * P)           # (B, C, H, W)
        return out


# ── Full model ────────────────────────────────────────────────────────────────
class TemporalViT(nn.Module):
    """
    Vision Transformer for temporal damage change detection.
    Takes stacked [pre_prob, post_prob] maps and outputs 3-class logits.
    """

    def __init__(self,
                 img_size    = 256,
                 patch_size  = 16,
                 in_channels = 2,
                 num_classes = 3,
                 embed_dim   = 192,
                 depth       = 4,
                 num_heads   = 6,
                 mlp_ratio   = 4.0,
                 dropout     = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, patch_size, embed_dim)

        n_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.head = DenseHead(embed_dim, patch_size, num_classes)

    def forward(self, x):
        tokens, Hg, Wg = self.patch_embed(x)
        tokens = self.dropout(tokens + self.pos_embed)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.head(tokens, Hg, Wg)   # (B, 3, H, W)
