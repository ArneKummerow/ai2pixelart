"""No-ground-truth quality assessment of a cleanup result.

Real AI images have no ground truth, so these metrics judge a CleanResult
against the input itself, each targeting one observed failure mode:

- grid: `boundary_snr` (gradient mass on cell edges vs interiors — is the
  detected grid where the real fake-pixel boundaries are?) and
  `pitch_consistency` (tile-wise re-estimation vs the global pitch — a
  single global grid on mixed-pitch content is the pipeline's known
  structural limitation, this measures how much an image violates it).
- palette representation: `cell_fit_mean` / `cell_fit_p95` (ΔE between each
  sampled cell color and the palette entry it received — high values mean
  the palette is too small/flattened for this image, e.g. absorbed shade
  families).
- pixel-art flatness: `speckle_rate` (isolated single-cell index flips that
  a detail-guarded mode filter would remove) and `shade_flicker` (adjacent
  cells assigned near-duplicate entries <4 ΔE apart — soft-gradient banding
  quantized into mottle; true pixel art places deliberately distinct colors
  next to each other).
- detail: `detail_survival` (sampled cells that are strong chromatic
  outliers vs their 4-neighborhood must remain outliers in the output — the
  1-px-detail promise, measured without ground truth).

All ΔE are CIE76.
"""

from __future__ import annotations

import numpy as np

from ai2pixelart.grid import estimate_grid
from ai2pixelart.palette import delta_e, rgb_to_lab
from ai2pixelart.pipeline import CleanResult


def boundary_snr(img: np.ndarray, result: CleanResult) -> float | None:
    """Gradient mass on detected cell boundaries / interior average (per
    axis, averaged). > 1.5 means the grid sits on real edges."""
    if result.grid is None:
        return None
    ratios = []
    for axis, ax_grid in ((0, result.grid.y), (1, result.grid.x)):
        g = np.abs(np.diff(img.astype(np.float64), axis=axis)).sum(axis=2).mean(axis=1 - axis)
        edge = np.round(ax_grid.edges[1:-1]).astype(int) - 1
        edge = edge[(edge >= 0) & (edge < len(g))]
        if not len(edge):
            return None
        on = np.zeros(len(g), dtype=bool)
        on[edge] = True
        if on.all() or not on.any():
            return None
        ratios.append(g[on].mean() / max(g[~on].mean(), 1e-9))
    return float(np.mean(ratios))


def pitch_consistency(img: np.ndarray, result: CleanResult, tiles: int = 3) -> float | None:
    """Fraction of image tiles whose locally-estimated pitch agrees (±10%)
    with the global grid, among tiles where estimation succeeds. Low values
    flag mixed-pitch content (out of scope for the single global grid)."""
    if result.grid is None:
        return None
    h, w = img.shape[:2]
    if min(h, w) < tiles * 64:
        return None
    agree = total = 0
    for ty in range(tiles):
        for tx in range(tiles):
            tile = img[ty * h // tiles : (ty + 1) * h // tiles, tx * w // tiles : (tx + 1) * w // tiles]
            g = estimate_grid(tile)
            if g is None:
                continue
            total += 1
            ok_y = abs(g.y.pitch / result.grid.y.pitch - 1) <= 0.1
            ok_x = abs(g.x.pitch / result.grid.x.pitch - 1) <= 0.1
            agree += ok_y and ok_x
    return float(agree / total) if total else None


def cell_fit(result: CleanResult) -> tuple[float, float]:
    """(mean, p95) ΔE between sampled cell colors and their palette entry."""
    d = delta_e(rgb_to_lab(result.raw_cells), rgb_to_lab(result.palette[result.indices]))
    return float(d.mean()), float(np.percentile(d, 95))


def _neighbor_stacks(a: np.ndarray) -> np.ndarray:
    """The 4-neighborhood of each entry, edge-padded: (4, H, W, ...)."""
    pad = np.pad(a, [(1, 1), (1, 1)] + [(0, 0)] * (a.ndim - 2), mode="edge")
    return np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])


def speckle_rate(result: CleanResult, tol: float = 3.0) -> float:
    """Fraction of cells that disagree with a >=5/8 neighborhood majority
    while fitting the majority's entry within `tol` ΔE of their own — i.e.
    quantization speckle a detail-guarded mode filter would remove."""
    from ai2pixelart.pipeline import smooth_indices

    smoothed = smooth_indices(result.indices, result.raw_cells, result.palette, tol=tol, passes=1)
    return float((smoothed != result.indices).mean())


def shade_flicker(result: CleanResult, close_de: float = 4.0) -> float:
    """Fraction of adjacent cell pairs whose (different) palette entries are
    nearly the same color — soft-gradient banding rendered as mottle."""
    lab_pal = rgb_to_lab(result.palette)
    idx = result.indices
    pairs = flicker = 0
    for a, b in ((idx[:, 1:], idx[:, :-1]), (idx[1:, :], idx[:-1, :])):
        differ = a != b
        close = delta_e(lab_pal[a], lab_pal[b]) < close_de
        flicker += int((differ & close).sum())
        pairs += differ.size
    return float(flicker / max(pairs, 1))


def detail_survival(result: CleanResult, outlier_de: float = 20.0) -> tuple[float, int]:
    """(survival rate, n details): sampled cells whose color is far from all
    4 neighbors (>= outlier_de) must stay distinct from all 4 neighbors in
    the output."""
    lab_raw = rgb_to_lab(result.raw_cells)
    raw_min = delta_e(_neighbor_stacks(lab_raw), lab_raw[None]).min(axis=0)
    details = raw_min >= outlier_de
    n = int(details.sum())
    if n == 0:
        return 1.0, 0
    lab_out = rgb_to_lab(result.palette)[result.indices]
    out_min = delta_e(_neighbor_stacks(lab_out), lab_out[None]).min(axis=0)
    return float((out_min[details] >= outlier_de * 0.5).mean()), n


def assess(img: np.ndarray, result: CleanResult) -> dict:
    """All no-GT metrics for one cleanup run."""
    fit_mean, fit_p95 = cell_fit(result)
    survival, n_details = detail_survival(result)
    return {
        "boundary_snr": _r(boundary_snr(img, result)),
        "pitch_consistency": _r(pitch_consistency(img, result)),
        "cell_fit_mean": round(fit_mean, 2),
        "cell_fit_p95": round(fit_p95, 2),
        "speckle_rate": round(speckle_rate(result), 4),
        "shade_flicker": round(shade_flicker(result), 4),
        "detail_survival": round(survival, 3),
        "n_details": n_details,
    }


def _r(v, nd: int = 2):
    return None if v is None else round(v, nd)
