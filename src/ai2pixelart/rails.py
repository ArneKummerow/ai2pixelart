"""Per-cell validity rails for generatively-corrupted training pairs.

A generative editor (img2img) adds realistic AI-rendering artifacts that the
VAE roundtrip cannot produce — but unlike the VAE it may also MOVE content.
The rails separate the two per source cell, using the exact scale/phase
metadata of the pair (no estimation):

- a cell whose corrupted region still reads as the source color (robust
  cell color within `keep_de` CIE76) is VALID — this includes cells where
  the editor added interior detail, which is exactly the gold signal the
  restoration net must learn to undo;
- a cell whose color drifted further is POISON (content moved or was
  hallucinated) and is masked out of the loss.
"""

from __future__ import annotations

import numpy as np

from ai2pixelart.nndata import source_cell_maps
from ai2pixelart.palette import delta_e, rgb_to_lab


def cell_colors(corrupt: np.ndarray, meta: dict, src_h: int, src_w: int) -> np.ndarray:
    """Mean corrupted color per source cell, via the exact pair mapping."""
    rows, cols = source_cell_maps(meta, corrupt.shape[0], corrupt.shape[1], src_h, src_w)
    flat = (rows[:, None] * src_w + cols[None, :]).ravel()
    counts = np.bincount(flat, minlength=src_h * src_w).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    out = np.empty((src_h * src_w, 3))
    pix = corrupt.reshape(-1, 3).astype(np.float64)
    for c in range(3):
        out[:, c] = np.bincount(flat, weights=pix[:, c], minlength=src_h * src_w) / counts
    return np.clip(np.round(out), 0, 255).astype(np.uint8).reshape(src_h, src_w, 3)


def validity_mask(
    clean: np.ndarray, corrupt: np.ndarray, meta: dict, keep_de: float = 14.0
) -> np.ndarray:
    """(src_h, src_w) bool: True where the corrupted cell still reads as its
    source color and may supervise the net."""
    cc = cell_colors(corrupt, meta, clean.shape[0], clean.shape[1])
    return delta_e(rgb_to_lab(cc), rgb_to_lab(clean)) <= keep_de
