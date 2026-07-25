"""Inference helpers for the restoration net."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ai2pixelart.nndata import majority_vote_cells, normalize_palette_lab

# re-exported so existing callers (and test monkeypatches) keep importing it
# from here; the format-aware implementation lives in models.py
from ai2pixelart.models import load_checkpoint  # noqa: F401


def net_logits(model, corrupt: np.ndarray, palette: np.ndarray, device: str):
    """Raw per-pixel class logits (K, H, W) on `device` — the net forward.

    Independent of leash/smooth/consensus, so it is the reusable unit the
    viewer caches: tweaking those downstream params re-runs only the cheap
    post-processing, not this."""
    import torch

    from ai2pixelart.nnmodel import pad_to_multiple

    x = torch.from_numpy(corrupt.transpose(2, 0, 1)).float()[None] / 127.5 - 1.0
    x, (h, w) = pad_to_multiple(x.to(device))
    pal = normalize_palette_lab(palette)
    mask = np.ones(len(palette), dtype=bool)
    with torch.no_grad():
        logits = model(
            x,
            torch.from_numpy(pal)[None].to(device),
            torch.from_numpy(mask)[None].to(device),
        )
    return logits[0, :, :h, :w]


def indices_from_logits(logits, corrupt: np.ndarray, palette: np.ndarray,
                        device: str, leash: float | None = None) -> np.ndarray:
    """Per-pixel indices from logits, optionally leashed. Never mutates the
    input logits (so cached logits stay reusable across leash values)."""
    if leash:
        logits = logits.clone()
        _apply_leash(logits, corrupt, palette, leash, device)
    return logits.argmax(0).cpu().numpy()


def predict_indices(
    model, corrupt: np.ndarray, palette: np.ndarray, device: str,
    leash: float | None = None,
) -> np.ndarray:
    """Per-pixel palette indices at input resolution (net forward + leash).

    The pointer head embeds each palette COLOR, not a slot, so any palette
    size works at inference (K_MAX only bounds training batches). Palettes
    much larger than the training range are extrapolation — usable, but
    discrimination between very close entries degrades. `leash` guards
    that: each pixel may only choose entries at most `leash` ΔE FURTHER
    than its nearest entry (d <= d_min + leash). Relative, not absolute:
    on a dense 256-color palette an absolute radius still admits dozens of
    near-greens (measured — the net then flip-flops among them), while the
    relative margin keeps only near-ties in play and makes recoloring
    orange into green impossible at any palette size.
    """
    return indices_from_logits(
        net_logits(model, corrupt, palette, device), corrupt, palette, device, leash=leash
    )


def _apply_leash(logits, corrupt: np.ndarray, palette: np.ndarray, leash: float,
                 device: str, chunk_rows: int = 256) -> None:
    """Forbid, in-place, every logit whose palette entry is more than `leash`
    ΔE beyond the pixel's nearest entry. The per-pixel (H,W,K) color-distance
    was the dominant cost of neural cleaning when done in host numpy; here it
    runs on the same device as the logits (the GPU), chunked over rows to
    bound memory. delta_e is Euclidean in Lab, i.e. exactly torch.cdist."""
    import torch

    from ai2pixelart.palette import rgb_to_lab

    lab_img = torch.from_numpy(rgb_to_lab(corrupt).astype(np.float32)).to(device)  # (H,W,3)
    lab_pal = torch.from_numpy(rgb_to_lab(palette).astype(np.float32)).to(device)  # (K,3)
    h, w = corrupt.shape[:2]
    for y0 in range(0, h, chunk_rows):
        lab = lab_img[y0 : y0 + chunk_rows]                        # (c,W,3)
        d = torch.cdist(lab.reshape(-1, 3), lab_pal).reshape(lab.shape[0], w, -1)
        allowed = d <= d.amin(dim=2, keepdim=True) + leash         # (c,W,K)
        logits[:, y0 : y0 + chunk_rows, :].masked_fill_(
            ~allowed.permute(2, 0, 1), float("-inf")
        )


def leash_mask(
    corrupt: np.ndarray, palette: np.ndarray, leash: float, chunk_rows: int = 64
) -> np.ndarray:
    """(H, W, K) bool: entries at most `leash` ΔE further from the pixel's
    color than its nearest entry (the nearest itself always qualifies)."""
    from ai2pixelart.palette import delta_e, rgb_to_lab

    lab_pal = rgb_to_lab(palette).astype(np.float32)
    h, w = corrupt.shape[:2]
    out = np.empty((h, w, len(palette)), dtype=bool)
    for y0 in range(0, h, chunk_rows):  # chunked: HxWxK distances get large
        lab = rgb_to_lab(corrupt[y0 : y0 + chunk_rows]).astype(np.float32)
        d = delta_e(lab[:, :, None, :], lab_pal[None, None, :, :])
        out[y0 : y0 + chunk_rows] = d <= (d.min(axis=2, keepdims=True) + leash)
    return out


def nn_clean_image(
    img: np.ndarray,
    model,
    device: str = "cuda",
    max_colors: int | None = None,
    palette: np.ndarray | None = None,
    leash: float | None = None,
    **proposal_params,
):
    """Full inference on an arbitrary image (no metadata) -> CleanResult.

    The classical pipeline proposes the palette (or an explicit `palette`
    replaces it) and the grid; the net does the per-pixel classification;
    per-cell majority vote produces the true-resolution output. All other
    classical parameters (merge_de, absorb_de/frac, pitch, granularity, ...)
    pass through to the proposal. An explicit `max_colors` is a target as
    well as a cap (see extract_palette), exactly as for the classical
    method; left unset, the proposal keeps its natural size. Palettes beyond
    K_MAX are extrapolation past the net's training range, but MEASURED to
    fit better than force-capping: on a 113-color sheet the v3 net scored
    fit 7.8 uncapped vs 10.1 capped at 16 — a richer palette lets imperfect
    discrimination land closer, and capping starves color-rich images (a
    16-color output from a 113-color sheet). Returning a CleanResult keeps
    the NN a drop-in sibling of the classical cleaner (same info/residual
    reporting everywhere).
    """
    prop = nn_propose(img, model, device, max_colors=max_colors, palette=palette, **proposal_params)
    return nn_finalize(
        img, prop, device, leash=leash,
        smooth=proposal_params.get("smooth", True),
        consensus=proposal_params.get("consensus", False),
    )


def nn_propose(img: np.ndarray, model, device: str,
               max_colors: int | None = None, palette: np.ndarray | None = None,
               **proposal_params) -> dict:
    """The expensive, cacheable half of neural cleaning: the classical
    palette/grid/cell proposal plus the net's raw logits. Independent of
    leash/smooth/consensus, so the viewer can reuse it while those change.

    Returns {palette, grid, raw_cells, logits} (logits a device tensor)."""
    from ai2pixelart.pipeline import clean as classical_clean

    # leash/smooth/consensus don't shape the proposal's palette/grid/cells
    # (only its DISCARDED indices) — drop them so the proposal is a pure
    # function of the remaining params (and cache-keyable without them)
    for k in ("leash", "smooth", "consensus"):
        proposal_params.pop(k, None)
    proposal = classical_clean(
        img,
        max_colors=None if palette is not None else max_colors,
        palette=palette,
        **proposal_params,
    )
    return {
        "palette": proposal.palette,
        "grid": proposal.grid,
        "raw_cells": proposal.raw_cells,
        "logits": net_logits(model, img, proposal.palette, device),
    }


def nn_finalize(img: np.ndarray, prop: dict, device: str,
                leash: float | None = None, smooth: bool = True,
                consensus: bool = False):
    """The cheap, param-sensitive half: leash the cached logits, vote per
    cell, then smooth/consensus. Re-runnable at slider speed on a cached
    proposal."""
    from ai2pixelart.pipeline import CleanResult, consensus_indices, smooth_indices

    palette, grid, raw_cells = prop["palette"], prop["grid"], prop["raw_cells"]
    pred_idx = indices_from_logits(prop["logits"], img, palette, device, leash=leash)

    if grid is None:
        # the net's votes get the same detail-guarded smoothing as the
        # classical path (the guard only joins a local majority the cell's
        # RAW color fits as well — real details stay)
        if smooth:
            pred_idx = smooth_indices(pred_idx, img, palette)
        return CleanResult(
            image=palette[pred_idx], palette=palette, indices=pred_idx,
            grid=None, raw_cells=img.copy(),
        )

    def cell_of(edges, n):
        return np.clip(np.searchsorted(edges, np.arange(n) + 0.5, side="right") - 1, 0, len(edges) - 2)

    rows = cell_of(grid.y.edges, img.shape[0])
    cols = cell_of(grid.x.edges, img.shape[1])
    src_h, src_w = grid.shape
    cell_allowed = None
    if leash:
        # cell-level leash: per-pixel noise on flat regions straddles
        # adjacent entries and mottles the vote — constrain each cell's
        # outcome to entries near its DENOISED sample color
        cell_allowed = leash_mask(raw_cells, palette, leash)
    cell_idx = majority_vote_cells(
        pred_idx, rows, cols, src_h, src_w, len(palette), allowed=cell_allowed
    )
    if smooth and cell_idx.shape == raw_cells.shape[:2]:
        cell_idx = smooth_indices(cell_idx, raw_cells, palette)
    if consensus and cell_idx.shape == raw_cells.shape[:2]:
        cell_idx = consensus_indices(cell_idx, raw_cells, palette)
    return CleanResult(
        image=palette[cell_idx], palette=palette, indices=cell_idx,
        grid=grid, raw_cells=raw_cells,
    )
