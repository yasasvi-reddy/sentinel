"""
temporal_vit.py

Vision Transformer for temporal damage classification.
Takes a 2-channel input (pre-war U-Net probability map + post-war U-Net probability map)
and outputs per-image class logits (3 classes: 0=undamaged, 1=newly damaged, 2=pre-existing).
"""

import math
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) → (B, num_patches, embed_dim)
        x = self.proj(x)                        # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)        # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1  = nn.Linear(embed_dim, hidden)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = Attention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalViT(nn.Module):
    """
    Vision Transformer for temporal war-damage classification.

    Parameters
    ----------
    img_size    : spatial size of input patches (square)
    patch_size  : size of each patch token
    in_channels : number of input channels (2 = pre_prob + post_prob)
    num_classes : number of output classes (3)
    embed_dim   : token embedding dimension
    depth       : number of transformer blocks
    num_heads   : attention heads per block
    mlp_ratio   : MLP hidden-dim multiplier
    dropout     : dropout rate
    """

    def __init__(
        self,
        img_size:    int   = 256,
        patch_size:  int   = 16,
        in_channels: int   = 2,
        num_classes: int   = 3,
        embed_dim:   int   = 192,
        depth:       int   = 4,
        num_heads:   int   = 6,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 2, H, W) — stacked pre/post probability maps
        B = x.shape[0]
        x = self.patch_embed(x)                                  # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)                   # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)                         # (B, N+1, D)
        x   = self.pos_drop(x + self.pos_embed)
        x   = self.blocks(x)
        x   = self.norm(x)
        return self.head(x[:, 0])                                 # (B, num_classes)
