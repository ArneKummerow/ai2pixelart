"""Inference helpers for the restoration net."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ai2pixelart.nndata import K_MAX, majority_vote_cells, normalize_palette_lab


def load_checkpoint(path: str | Path, device: str = "cuda"):
    import torch

    from ai2pixelart.nnmodel import PixelCleanNet

    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = PixelCleanNet(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.ckpt_step = ckpt.get("step")  # surfaced in run info (in-training ckpts)
    return model


def predict_indices(
    model, corrupt: np.ndarray, palette: np.ndarray, device: str,
    leash: float | None = None,
) -> np.ndarray:
    """Per-pixel palette indices at input resolution.

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
    logits = logits[0, :, :h, :w]
    if leash:
        allowed = torch.from_numpy(leash_mask(corrupt, palette, leash)).to(device)
        logits = logits.masked_fill(~allowed.permute(2, 0, 1), float("-inf"))
    return logits.argmax(0).cpu().numpy()


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
    from ai2pixelart.pipeline import (
        CleanResult,
        clean as classical_clean,
        consensus_indices,
        smooth_indices,
    )

    # the net's votes get the same detail-guarded smoothing as the classical
    # path: without it, near-duplicate entries speckle across homogeneous
    # backgrounds (the guard only allows joining a local majority the cell's
    # RAW color fits as well — real details stay)
    smooth = proposal_params.get("smooth", True)
    # popped, not passed through: the proposal's own indices are discarded,
    # so running consensus there would be wasted work — it applies to the
    # net's cells below instead
    consensus = proposal_params.pop("consensus", False)
    proposal = classical_clean(
        img,
        max_colors=None if palette is not None else max_colors,
        palette=palette,
        **proposal_params,
    )
    palette = proposal.palette
    pred_idx = predict_indices(model, img, palette, device, leash=leash)

    if proposal.grid is None:
        if smooth:
            pred_idx = smooth_indices(pred_idx, img, palette)
        return CleanResult(
            image=palette[pred_idx], palette=palette, indices=pred_idx,
            grid=None, raw_cells=img.copy(),
        )

    def cell_of(edges, n):
        return np.clip(np.searchsorted(edges, np.arange(n) + 0.5, side="right") - 1, 0, len(edges) - 2)

    rows = cell_of(proposal.grid.y.edges, img.shape[0])
    cols = cell_of(proposal.grid.x.edges, img.shape[1])
    src_h, src_w = proposal.grid.shape
    cell_allowed = None
    if leash:
        # cell-level leash: per-pixel noise on flat regions straddles
        # adjacent entries and mottles the vote — constrain each cell's
        # outcome to entries near its DENOISED sample color
        cell_allowed = leash_mask(proposal.raw_cells, palette, leash)
    cell_idx = majority_vote_cells(
        pred_idx, rows, cols, src_h, src_w, len(palette), allowed=cell_allowed
    )
    if smooth and cell_idx.shape == proposal.raw_cells.shape[:2]:
        cell_idx = smooth_indices(cell_idx, proposal.raw_cells, palette)
    if consensus and cell_idx.shape == proposal.raw_cells.shape[:2]:
        cell_idx = consensus_indices(cell_idx, proposal.raw_cells, palette)
    return CleanResult(
        image=palette[cell_idx], palette=palette, indices=cell_idx,
        grid=proposal.grid, raw_cells=proposal.raw_cells,
    )
