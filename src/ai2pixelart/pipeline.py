"""Classical cleanup pipeline: grid estimation -> cell downsample -> palette quantization.

This is the baseline and, more importantly, the labeling/measurement
infrastructure for the learned pipeline (alignment scoring, validity masks,
metrics). It assumes a single (mildly wobbly) global grid and will lose to a
learned model exactly where detail density varies locally — by design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ai2pixelart.grid import Grid, estimate_grid, regrain, uniform_axis_grid
from ai2pixelart.palette import dedupe_palette, delta_e, extract_palette, quantize, rgb_to_lab


@dataclass
class CleanResult:
    image: np.ndarray  # (rows, cols, 3) uint8, palette-quantized true-resolution art
    palette: np.ndarray  # (K, 3) uint8
    indices: np.ndarray  # (rows, cols) palette index per cell
    grid: Grid | None  # None => input treated as already true-resolution
    raw_cells: np.ndarray  # cell colors before palette quantization


def downsample_cells(img: np.ndarray, grid: Grid, keep_frac: float = 0.34) -> np.ndarray:
    """One robust color per grid cell.

    Per-channel median over a centered window covering ~`keep_frac` of the
    cell — a single center pixel at typical pitches. Resampling mixes
    neighboring cells into everything off-center (at pitch ~3 a pixel 0.4
    cells off-center can carry under half of its own cell's color), so a
    tight center crop beats a wide interior median for 1-px features like
    outlines and eye highlights.

    Vectorized as one gather: the window size is uniform per axis (from the
    median cell width — cell widths only vary by the rasterization +-1), so
    all cells' windows stack into a (rows, cols, ky, kx, 3) block.
    """
    yy, _ = _center_indices(grid.y.edges, img.shape[0], keep_frac)
    xx, _ = _center_indices(grid.x.edges, img.shape[1], keep_frac)
    block = img[yy[:, None, :, None], xx[None, :, None, :]]
    return np.round(np.median(block, axis=(2, 3))).astype(np.uint8)


def _center_indices(edges: np.ndarray, size: int, keep_frac: float) -> tuple[np.ndarray, int]:
    widths = np.diff(edges)
    keep = max(1, int(round(float(np.median(widths)) * keep_frac)))
    keep = min(keep, size)
    centers = (edges[:-1] + edges[1:]) / 2.0
    lo = np.floor(centers - keep / 2.0 + 0.5).astype(int)
    lo = np.clip(lo, 0, size - keep)
    return lo[:, None] + np.arange(keep)[None, :], keep


def smooth_indices(
    indices: np.ndarray,
    raw_cells: np.ndarray,
    palette: np.ndarray,
    tol: float = 3.0,
    passes: int = 2,
) -> np.ndarray:
    """Kill palette-index speckle in flat regions, keeping real details.

    AI art shades "flat" regions with soft gradients + noise, so neighboring
    cells quantize to different near-identical palette entries and the
    output mottles. Pixel-art prior: a cell surrounded by one dominant index
    should join it — UNLESS its own sampled color genuinely fits its current
    entry better by more than `tol` ΔE. That guard is what protects 1-px
    details and outlines: their raw colors fit the neighborhood's entry far
    worse, while speckle (near-duplicate entries) fits within a hair.
    """
    lab_pal = rgb_to_lab(palette)
    lab_raw = rgb_to_lab(raw_cells)
    idx = indices.copy()
    ii, jj = np.indices(idx.shape)
    for _ in range(passes):
        # neighborhood mode over the 8-neighborhood, per palette entry
        pad = np.pad(idx, 1, mode="edge")
        votes = np.zeros((len(palette),) + idx.shape, dtype=np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = pad[1 + dy : pad.shape[0] - 1 + dy, 1 + dx : pad.shape[1] - 1 + dx]
                votes[shifted, ii, jj] += 1  # (i, j) unique per shift: no collisions
        mode = votes.argmax(axis=0)
        count = votes.max(axis=0)
        d_cur = delta_e(lab_raw, lab_pal[idx])
        d_mode = delta_e(lab_raw, lab_pal[mode])
        switch = (mode != idx) & (count >= 5) & (d_mode - d_cur <= tol)
        if not switch.any():
            break
        idx[switch] = mode[switch]
    return idx


def consensus_indices(
    indices: np.ndarray,
    raw_cells: np.ndarray,
    palette: np.ndarray,
    merge_de: float = 2.5,
    guard_de: float = 2.0,
) -> np.ndarray:
    """Global flat-color consensus: near-identical observed regions get the
    same palette entry, image-wide.

    smooth_indices coordinates only across adjacent cells, and the net only
    within its receptive field — two areas of the SAME color far apart are
    decided independently, and when the observed color sits near-equidistant
    between palette entries, noise flips the winner per area ("almost
    identical areas get different colors", measured on a 2048px tile sheet
    with a dense 256-color palette: 49-65% background dominance).

    Connected components of equal index are clustered by their mean observed
    (raw cell) color — components within `merge_de` ΔE are declared "the
    same intended color" — and each cluster is forced to its mass-weighted
    winning entry. A component flips if it genuinely belongs to the cluster
    (observed mean within `merge_de` of the cluster's mass centroid) or if
    the winner fits its observed mean within `guard_de` ΔE of its current
    entry's fit (the relative-guard idea of smooth_indices/leash, rescuing
    borderline members). Details and legitimate distinct shades are safe on
    both counts: they sit in their own clusters — or, when chained in by
    single-linkage, far from the mass centroid with a fit gap beyond the
    guard.
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree

    lab_pal = rgb_to_lab(palette)
    lab_raw = rgb_to_lab(raw_cells)

    labels = np.zeros(indices.shape, dtype=np.int64)
    n_total = 0
    for e in np.unique(indices):
        m = indices == e
        lab_e, n_e = ndimage.label(m)
        labels[m] = lab_e[m] + n_total
        n_total += n_e
    flat = labels.ravel() - 1
    mass = np.bincount(flat, minlength=n_total).astype(np.float64)
    mean_lab = np.stack(
        [np.bincount(flat, weights=lab_raw[..., c].ravel(), minlength=n_total) for c in range(3)],
        axis=1,
    ) / mass[:, None]
    comp_entry = np.zeros(n_total, dtype=np.int64)
    comp_entry[flat] = indices.ravel()

    # cluster component colors: quantize onto a merge_de Lab grid, then
    # union grid reps within merge_de (approximate single linkage — exact
    # enough because near-identical colors land in the same or an adjacent
    # bin, and exact linkage over ~10^4 mottle components is infeasible)
    q = np.round(mean_lab / merge_de).astype(np.int64)
    uniq, inv = np.unique(q, axis=0, return_inverse=True)
    rep_mass = np.bincount(inv, weights=mass, minlength=len(uniq))
    rep = np.stack(
        [np.bincount(inv, weights=mean_lab[:, c] * mass, minlength=len(uniq)) for c in range(3)],
        axis=1,
    ) / rep_mass[:, None]
    parent = np.arange(len(uniq))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in cKDTree(rep).query_pairs(merge_de):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    _, dense = np.unique([find(a) for a in range(len(uniq))], return_inverse=True)
    comp_cluster = dense[inv]

    k = len(palette)
    votes = np.bincount(
        comp_cluster * k + comp_entry, weights=mass, minlength=(comp_cluster.max() + 1) * k
    ).reshape(-1, k)
    winner = votes.argmax(axis=1)[comp_cluster]
    cl_mass = np.bincount(comp_cluster, weights=mass)
    centroid = np.stack(
        [np.bincount(comp_cluster, weights=mean_lab[:, c] * mass) for c in range(3)], axis=1
    ) / cl_mass[:, None]
    d_cur = delta_e(mean_lab, lab_pal[comp_entry])
    d_new = delta_e(mean_lab, lab_pal[winner])
    member = delta_e(mean_lab, centroid[comp_cluster]) <= merge_de
    flip = (winner != comp_entry) & (member | (d_new <= d_cur + guard_de))
    if not flip.any():
        return indices
    remap = comp_entry.copy()
    remap[flip] = winner[flip]
    return remap[flat].reshape(indices.shape)


def clean(
    img: np.ndarray,
    merge_de: float = 3.0,
    max_colors: int | None = None,
    palette: np.ndarray | None = None,
    pitch: tuple[float, float] | None = None,
    keep_frac: float = 0.34,
    denoise: bool = True,
    absorb_de: float | None = None,
    absorb_frac: float = 0.05,
    granularity: float = 1.0,
    smooth: bool = True,
    consensus: bool = False,
) -> CleanResult:
    """Clean a pseudo-pixel-art image into palette-quantized true-resolution art.

    img: (H, W, 3) uint8.
    palette: if given, output is forced onto exactly these colors; otherwise a
        palette is extracted from the cell colors (merged below `merge_de`,
        with mixture absorption controlled by `absorb_de`/`absorb_frac`).
    absorb_de: mixture-drain radius; None = auto by pitch — 15 at pitch >= 3
        (bleed is low-mass and draining it recovers detail colors), 10 below
        (bleed is too massive to drain there, so a wide radius only eats the
        image's real shades — measured on a 2px-pitch sprite sheet the wide
        radius collapsed 113 real colors to 32).
    pitch: optional known (y_pitch, x_pitch), bypassing grid estimation.
    denoise: cross-median filter before cell sampling (auto-skipped below
        pitch 3 regardless).
    granularity: output resolution relative to the detected grid — 2 samples
        each cell as 2x2, 0.5 merges 2x2 cells (see grid.regrain).
    smooth: detail-guarded neighborhood-mode filter on the palette indices
        (see smooth_indices) — removes flat-region speckle.
    consensus: global flat-color consensus (see consensus_indices) — forces
        near-identical observed regions anywhere in the image onto the same
        palette entry. Off by default; most useful with dense forced
        palettes on noisy inputs.
    """
    if pitch is not None:
        grid = Grid(
            y=uniform_axis_grid(img.shape[0], pitch[0]),
            x=uniform_axis_grid(img.shape[1], pitch[1]),
        )
    else:
        grid = estimate_grid(img)
    if grid is not None and granularity != 1.0:
        grid = regrain(grid, granularity)

    if grid is None:
        cells = img.copy()
    else:
        sample_img = img
        if denoise and min(grid.y.pitch, grid.x.pitch) >= 3.0:
            # edge-preserving denoise before center sampling. A cross-shaped
            # (5-point) median, not a 3x3 square: resampling shrinks a 1-cell
            # detail's pure core to ~2x2 px, and a 9-point median there is
            # dominated by the mixed surround and erases the detail, while
            # the cross keeps 3 of 5 samples on the core (and likewise
            # follows 1-px bands like outlines). Skipped below pitch 3 where
            # even the cross would span neighboring cells.
            cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
            sample_img = ndimage.median_filter(img, footprint=cross[:, :, None])
        cells = downsample_cells(sample_img, grid, keep_frac=keep_frac)

    if absorb_de is None:
        fine = grid is not None and min(grid.y.pitch, grid.x.pitch) < 3.0
        absorb_de = 10.0 if fine else 15.0
    if palette is None:
        palette = extract_palette(
            cells,
            merge_de=merge_de,
            max_colors=max_colors,
            absorb_de=absorb_de,
            absorb_frac=absorb_frac,
        )
    else:
        # forced palettes may contain near-duplicate entries (measured
        # 0.4 ΔE floor on a real 256-color palette) that turn entry choice
        # into a noise-decided coin flip; indistinguishable twins are dropped
        palette = dedupe_palette(palette)
    quantized, indices = quantize(cells, palette)
    if smooth:
        indices = smooth_indices(indices, cells, palette)
        quantized = palette[indices].astype(np.uint8)
    if consensus and grid is not None:
        indices = consensus_indices(indices, cells, palette)
        quantized = palette[indices].astype(np.uint8)

    return CleanResult(
        image=quantized, palette=palette, indices=indices, grid=grid, raw_cells=cells
    )
