"""Dataset over corruption-pair output for the restoration net.

The training target is exact by construction: pairs.jsonl records the scale
and phase of each corruption, and the nearest-upscale mapping (matching
corrupt.upscale) assigns every corrupted pixel its source cell — hence its
palette index. No estimation is involved anywhere in the supervision path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ai2pixelart.palette import image_palette, rgb_to_lab

# Palette slots per training sample. Raised from 16: samples now include
# DECOY entries (near-miss colors not present in the image) so the pointer
# head learns dense-palette discrimination — user assets bring 256-color
# palettes, and a net that only ever chose among <=16 real entries
# recolors badly there (measured: orange->green at K=256).
K_MAX = 96


def make_decoys(
    palette: np.ndarray, n: int, rng: np.random.Generator, min_de: float = 5.0
) -> np.ndarray:
    """Up to n decoy colors: half random, half near-miss jitters of real
    entries — kept only if >= min_de from EVERY real entry, so the training
    target stays unambiguous."""
    from ai2pixelart.palette import delta_e

    lab_used = rgb_to_lab(palette)
    out: list[np.ndarray] = []
    for _ in range(n * 4):
        if len(out) >= n:
            break
        if rng.random() < 0.5:
            cand = rng.integers(0, 256, 3)
        else:
            base = palette[int(rng.integers(len(palette)))].astype(np.float64)
            cand = np.clip(np.round(base + rng.normal(0, 18, 3)), 0, 255)
        cand = cand.astype(np.uint8)
        if float(delta_e(rgb_to_lab(cand), lab_used).min()) >= min_de:
            out.append(cand)
    return np.array(out, dtype=np.uint8).reshape(-1, 3)


def normalize_palette_lab(palette: np.ndarray) -> np.ndarray:
    """uint8 (K,3) RGB -> roughly unit-range Lab float32."""
    lab = rgb_to_lab(palette).astype(np.float32)
    lab[:, 0] = lab[:, 0] / 50.0 - 1.0
    lab[:, 1:] = lab[:, 1:] / 110.0
    return lab


def source_cell_maps(meta: dict, out_h: int, out_w: int, src_h: int, src_w: int):
    """Corrupted pixel -> source cell index per axis (matches corrupt.upscale)."""

    def axis(n_out, n_src, scale, phase):
        src = (np.arange(n_out) + 0.5 + phase) / scale - 0.5
        return np.clip(np.floor(src + 0.5).astype(np.int64), 0, n_src - 1)

    return (
        axis(out_h, src_h, meta["scale_y"], meta["phase_y"]),
        axis(out_w, src_w, meta["scale_x"], meta["phase_x"]),
    )


def index_map(clean: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Exact palette index per clean pixel."""
    eq = (clean[:, :, None, :] == palette[None, None, :, :]).all(axis=-1)
    if not eq.any(axis=-1).all():
        raise ValueError("clean image contains colors missing from its palette")
    return eq.argmax(axis=-1).astype(np.int64)


class PairDataset:
    """Torch-style dataset yielding fixed-size crops with per-pixel targets."""

    def __init__(
        self,
        pairs_dir: str | Path | list,
        crop: int = 88,
        split: str = "train",
        val_frac: float = 0.05,
        seed: int = 0,
        detail_weight: float = 8.0,
        decoy_max: int = 48,
    ):
        self.decoy_max = decoy_max
        roots = [Path(p) for p in (pairs_dir if isinstance(pairs_dir, (list, tuple)) else [pairs_dir])]
        self.dir = roots[0]  # kept for single-dir callers
        self.crop = crop
        self.train = split == "train"
        self.detail_weight = detail_weight
        items = []
        for root in roots:  # mixed corpora: each item remembers its root
            for line in (root / "pairs.jsonl").read_text().splitlines():
                if line.strip():
                    meta = json.loads(line)
                    meta["_root"] = str(root)
                    items.append(meta)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(items))
        n_val = max(1, int(len(items) * val_frac))
        picked = order[n_val:] if self.train else order[:n_val]
        self.items = [items[i] for i in picked]
        # drop pairs smaller than the crop (small sprite x low scale); PNG
        # header reads only, so this is cheap even for thousands of pairs
        kept = []
        for meta in self.items:
            with Image.open(self.path_of(meta, "corrupt")) as im:
                w, h = im.size
            if h >= crop and w >= crop:
                kept.append(meta)
        self.n_dropped_small = len(self.items) - len(kept)
        self.items = kept
        self._rng = np.random.default_rng(seed + (0 if self.train else 1))

    def path_of(self, meta: dict, key: str) -> Path:
        return Path(meta.get("_root", self.dir)) / meta[key]

    def __len__(self) -> int:
        return len(self.items)

    def load_pair(self, meta: dict):
        """(corrupt uint8, per-pixel target idx, palette uint8) full-size."""
        corrupt, target, palette, _ = self._load_full(meta)
        return corrupt, target, palette

    def _load_full(self, meta: dict):
        clean = np.array(Image.open(self.path_of(meta, "clean")).convert("RGB"))
        corrupt = np.array(Image.open(self.path_of(meta, "corrupt")).convert("RGB"))
        palette = image_palette(clean)
        if len(palette) > K_MAX:
            raise ValueError(f"palette larger than K_MAX: {len(palette)}")
        idx = index_map(clean, palette)
        rows, cols = source_cell_maps(
            meta, corrupt.shape[0], corrupt.shape[1], clean.shape[0], clean.shape[1]
        )
        target = idx[rows[:, None], cols[None, :]]

        # loss weights: pixels of isolated single-cell details are upweighted —
        # a 1-px eye is ~25 of 7700 crop pixels, invisible to uniform CE (and
        # exactly the thing the model must never trade away)
        weight = np.ones(target.shape, dtype=np.float32)
        if self.detail_weight != 1.0:
            from ai2pixelart.metrics import isolated_details

            isolated = isolated_details(clean)
            weight += (self.detail_weight - 1.0) * isolated[rows[:, None], cols[None, :]]
        if meta.get("mask"):
            # rail-guarded pairs (img2img): cells whose content drifted are
            # poison — zero their loss (see rails.validity_mask)
            valid = np.array(Image.open(self.path_of(meta, "mask")).convert("L")) > 127
            weight *= valid[rows[:, None], cols[None, :]]
        return corrupt, target, palette, weight

    def __getitem__(self, i: int):
        import torch

        meta = self.items[i]
        corrupt, target, palette, weight = self._load_full(meta)

        h, w = corrupt.shape[:2]
        if h < self.crop or w < self.crop:
            raise ValueError(f"pair smaller than crop: {h}x{w} < {self.crop}")
        if self.train:
            y0 = int(self._rng.integers(0, h - self.crop + 1))
            x0 = int(self._rng.integers(0, w - self.crop + 1))
        else:
            y0, x0 = (h - self.crop) // 2, (w - self.crop) // 2
        corrupt = corrupt[y0 : y0 + self.crop, x0 : x0 + self.crop]
        target = target[y0 : y0 + self.crop, x0 : x0 + self.crop]
        weight = weight[y0 : y0 + self.crop, x0 : x0 + self.crop]

        if self.train and self.decoy_max > 0:
            room = K_MAX - len(palette)
            n = int(self._rng.integers(0, min(self.decoy_max, room) + 1))
            decoys = make_decoys(palette, n, self._rng)
            if len(decoys):
                palette = np.concatenate([palette, decoys])

        pal_lab = np.zeros((K_MAX, 3), dtype=np.float32)
        pal_lab[: len(palette)] = normalize_palette_lab(palette)
        pal_mask = np.zeros(K_MAX, dtype=bool)
        pal_mask[: len(palette)] = True

        return {
            "img": torch.from_numpy(corrupt.transpose(2, 0, 1)).float() / 127.5 - 1.0,
            "target": torch.from_numpy(target),
            "weight": torch.from_numpy(weight),
            "palette": torch.from_numpy(pal_lab),
            "pal_mask": torch.from_numpy(pal_mask),
        }


def majority_vote_cells(
    pred_idx: np.ndarray, rows: np.ndarray, cols: np.ndarray, src_h: int, src_w: int, k: int,
    allowed: np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate per-pixel predictions into per-cell indices (majority vote).

    `allowed` ((src_h, src_w, k) bool) restricts each cell's outcome to a
    candidate set — the cell-level color leash: per-pixel noise on flat
    regions can genuinely straddle adjacent palette entries, and the vote
    then mottles; constraining it to entries near the cell's denoised
    sample keeps flat regions committed."""
    flat_cell = (rows[:, None] * src_w + cols[None, :]).ravel()
    counts = np.zeros((src_h * src_w, k), dtype=np.int64)
    np.add.at(counts, (flat_cell, pred_idx.ravel()), 1)
    if allowed is not None:
        counts = np.where(allowed.reshape(-1, k), counts, -1)
    cell_idx = counts.argmax(axis=1).reshape(src_h, src_w)
    empty = np.maximum(counts, 0).sum(axis=1).reshape(src_h, src_w) == 0
    if empty.any():
        # sub-pixel cells (granularity 2 on a ~1px grid) can contain no
        # pixel center; argmax over zero votes would silently paint palette
        # entry 0 — grid-shaped dark lines. Take the nearest voted cell.
        from scipy import ndimage

        _, (iy, ix) = ndimage.distance_transform_edt(empty, return_indices=True)
        cell_idx = cell_idx[iy, ix]
    return cell_idx
