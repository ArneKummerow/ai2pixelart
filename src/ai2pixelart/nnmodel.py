"""Restoration network: palette-conditioned per-pixel classification U-Net.

The network never regresses RGB. It produces a feature vector per input
pixel and the palette's colors are embedded as keys; per-pixel logits are
the scaled dot products (a pointer head). The output is therefore
structurally incapable of off-palette colors, and one model handles any
palette size up to K_MAX (padding masked out of the softmax).

Output lives at INPUT resolution: a "regularized" palette-indexed image in
which every fake pixel should become crisp, axis-aligned, and uniform. The
true-resolution image is obtained afterwards by per-cell majority vote
(known grid during training, estimated grid at inference) — this keeps the
network fully convolutional and sidesteps variable output sizes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    def gn(c):
        return nn.GroupNorm(min(8, c), c)

    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), gn(cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), gn(cout), nn.SiLU(),
    )


class PixelCleanNet(nn.Module):
    DOWN_FACTOR = 8  # input H/W must be divisible by this (see pad_to_multiple)

    def __init__(self, base: int = 48, feat_dim: int = 64):
        super().__init__()
        self.stem = _block(3, base)
        self.down1 = nn.Sequential(nn.Conv2d(base, base * 2, 3, stride=2, padding=1), _block(base * 2, base * 2))
        self.down2 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1), _block(base * 4, base * 4))
        self.down3 = nn.Sequential(nn.Conv2d(base * 4, base * 4, 3, stride=2, padding=1), _block(base * 4, base * 4))
        self.up3 = _block(base * 4 + base * 4, base * 4)
        self.up2 = _block(base * 4 + base * 2, base * 2)
        self.up1 = _block(base * 2 + base, base)
        self.head = nn.Conv2d(base, feat_dim, 1)
        self.palette_mlp = nn.Sequential(
            nn.Linear(3, feat_dim), nn.SiLU(), nn.Linear(feat_dim, feat_dim)
        )
        self.feat_dim = feat_dim

    def forward(
        self, img: torch.Tensor, palette: torch.Tensor, pal_mask: torch.Tensor
    ) -> torch.Tensor:
        """img (B,3,H,W) in [-1,1]; palette (B,K,3) normalized Lab;
        pal_mask (B,K) bool. Returns logits (B,K,H,W)."""
        s0 = self.stem(img)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        x = self.down3(s2)
        x = self.up3(torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), s2], dim=1))
        x = self.up2(torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), s1], dim=1))
        x = self.up1(torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), s0], dim=1))
        feats = self.head(x)  # B,C,H,W

        keys = self.palette_mlp(palette)  # B,K,C
        logits = torch.einsum("bchw,bkc->bkhw", feats, keys) / self.feat_dim**0.5
        return logits.masked_fill(~pal_mask[:, :, None, None], -1e4)


def pad_to_multiple(img: torch.Tensor, m: int = PixelCleanNet.DOWN_FACTOR):
    """Replicate-pad (B,3,H,W) so H and W divide m; returns (padded, (H, W))."""
    h, w = img.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        img = F.pad(img, (0, pw, 0, ph), mode="replicate")
    return img, (h, w)
