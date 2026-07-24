"""Palette extraction and palette-constrained quantization.

Color distances are CIE76 (Euclidean in CIELAB). All ΔE thresholds in this
package are calibrated for CIE76.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from skimage import color

# Above this many distinct colors, hierarchical clustering runs on one
# representative per occupied color-space bin (adaptive bin width until
# <= MAX_UNIQUE; scipy's average linkage on 4096 observations takes ~1 min,
# on 1024 about a second). Two properties both matter:
# - bins give COVERAGE: rare-but-distinct color families (a green sprite on
#   a mostly pink sheet) keep a representative — picking the globally most
#   frequent colors instead once starved them out and whole hues collapsed
#   into the dominant families;
# - the representative is the bin's most frequent EXACT color, not the bin
#   center: a bin center is a made-up color, and at the widths color-rich
#   images need (16-32) it tints dominant colors by several ΔE.
# Centroids are computed from all exact colors afterwards.
MAX_UNIQUE = 1024


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB (..., 3) -> CIELAB float (..., 3)."""
    return color.rgb2lab(rgb.astype(np.float64) / 255.0)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIELAB (..., 3) -> uint8 RGB, clipped to gamut."""
    rgb = color.lab2rgb(lab)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def delta_e(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """CIE76 ΔE between two Lab arrays (broadcasting)."""
    return np.sqrt(((lab_a - lab_b) ** 2).sum(axis=-1))


def _unique_colors(colors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """-> (unique colors, counts, linkage representatives, rep per unique).

    With few distinct colors the representatives ARE the unique colors;
    beyond MAX_UNIQUE there is one per occupied color-space bin — the bin's
    most frequent exact member — and every unique color maps to its bin's
    representative.
    """
    uniq, counts = np.unique(colors.reshape(-1, 3), axis=0, return_counts=True)
    if len(uniq) <= MAX_UNIQUE:
        return uniq, counts, uniq, np.arange(len(uniq))
    step = 2
    while True:
        bins, bin_of = np.unique(uniq // step, axis=0, return_inverse=True)
        if len(bins) <= MAX_UNIQUE:
            break
        step *= 2
    # representative = the bin's mode: iterate members by ascending count so
    # each bin's final write is its most frequent exact color
    order = np.argsort(counts, kind="stable")
    rep_idx = np.empty(len(bins), dtype=np.intp)
    rep_idx[bin_of[order]] = order
    reps = uniq[rep_idx]
    # assign by proximity, not bin membership: a wide bin would lump its
    # whole color box into one cluster and drag the median off the peak
    from scipy.spatial import cKDTree

    inv = cKDTree(rgb_to_lab(reps)).query(rgb_to_lab(uniq), workers=-1)[1]
    return uniq, counts, reps, inv


def extract_palette(
    colors: np.ndarray,
    merge_de: float = 3.0,
    max_colors: int | None = None,
    absorb_de: float = 15.0,
    absorb_frac: float = 0.05,
) -> np.ndarray:
    """Cluster observed colors into a palette.

    Three stages, all in Lab with pixel-count-weighted means:

    1. Agglomerative clustering with ΔE cutoff `merge_de`, so the palette
       size adapts to the image instead of being fixed up front. If
       `max_colors` asks for MORE clusters than the cutoff yields, the merge
       tree is cut by count instead: an explicit color count is a target,
       not only a cap.
    2. Absorption, smallest cluster first: any cluster under `absorb_frac`
       of the mass merges into its nearest neighbor within `absorb_de`.
       Noise fragments of one true color coalesce first (they are mutually
       closest) and rebuild that color above the size threshold, while
       boundary-bleed mixtures drain into the real color they sit next to.
       Genuine rare details — a two-cell white eye — are small but far from
       everything, and survive. This distinction is what keeps "denoise the
       palette" from also meaning "delete the rare details".
    3. Near-duplicate centroid merge (average linkage chaining can split one
       noisy color into adjacent clusters), then closest-pair merging down to
       `max_colors` if set. With `max_colors` set, stages 2 and 3 never go
       below it, so the requested count survives to the output.
    4. Usage check: the anti-drag merge rule (below) can leave "shadowed"
       entries that no observed color is nearest to — they would silently
       vanish from the quantized output. Each such entry is reseeded to the
       observed color with the largest count-weighted error.

    Returns (K, 3) uint8, most frequently used color first.
    """
    uniq, counts, reps, inv = _unique_colors(colors)
    if len(uniq) == 1:
        return uniq.astype(np.uint8)

    tree = linkage(rgb_to_lab(reps), method="average")
    rep_labels = fcluster(tree, t=merge_de, criterion="distance")
    if max_colors is not None and len(np.unique(rep_labels)) < max_colors:
        rep_labels = fcluster(tree, t=max_colors, criterion="maxclust")
    # centroids from the exact colors (reps served only the linkage). A
    # cluster whose most frequent exact color carries a real share of its
    # mass IS a flat-area color — take it verbatim (measured on the Gemini
    # set: flat-area dominants hold 9-29% in one exact color, noise-spread
    # clusters ~2%). Otherwise use the count-weighted MEDIAN: wobble/bleed
    # clouds have asymmetric tails, and the mean inherits a visible tint
    # from them while the median stays on the peak.
    labels = rep_labels[inv]
    lab = rgb_to_lab(uniq)
    w = counts.astype(np.float64)
    cents, ws = [], []
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        cw = w[mask]
        mode = int(np.argmax(cw))
        if cw[mode] >= 0.05 * cw.sum():
            cents.append(lab[mask][mode])
        else:
            cents.append(_weighted_median(lab[mask], cw))
        ws.append(float(cw.sum()))
    cents = np.asarray(cents)
    ws = np.asarray(ws)

    def merge(i: int, j: int) -> None:
        """Fold cluster i into cluster j.

        Comparably sized clusters are noise fragments of one color and their
        centroids average; a much smaller cluster is boundary-bleed
        contamination — a corrupted observation, not a sample — and must not
        drag the real color's centroid (with a dominant background even a
        ~1 ΔE drag pushes every one of its cells toward the tolerance edge).
        """
        nonlocal cents, ws
        if ws[i] >= 0.25 * ws[j]:
            cents[j] = (cents[i] * ws[i] + cents[j] * ws[j]) / (ws[i] + ws[j])
        ws[j] += ws[i]
        cents = np.delete(cents, i, axis=0)
        ws = np.delete(ws, i)

    total = ws.sum()
    changed = True
    while changed and len(ws) > (max_colors or 1):
        changed = False
        for i in np.argsort(ws):
            if ws[i] >= absorb_frac * total:
                break
            d = delta_e(cents[i], cents)
            d[i] = np.inf
            j = int(np.argmin(d))
            if d[j] <= absorb_de:
                merge(i, j)
                changed = True
                break

    def closest_pair() -> tuple[float, int, int]:
        d = delta_e(cents[:, None, :], cents[None, :, :])
        d[np.tril_indices(len(cents))] = np.inf
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        return float(d[i, j]), int(i), int(j)

    dupe_de = max(2.5, merge_de / 2)
    while len(ws) > (max_colors or 1):
        dist, i, j = closest_pair()
        if max_colors is None and dist > dupe_de:
            break
        merge(i, j)

    return _reseed_unused(lab_to_rgb(cents), colors)


def _weighted_median(vals: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-channel weighted median of (n, 3) values."""
    out = np.empty(3)
    for c in range(3):
        order = np.argsort(vals[:, c], kind="stable")
        cw = np.cumsum(w[order])
        out[c] = vals[order[np.searchsorted(cw, 0.5 * cw[-1])], c]
    return out


def _nearest(lab_pts: np.ndarray, lab_pal: np.ndarray) -> np.ndarray:
    """Index of the nearest palette entry per point (chunked, quantize's metric)."""
    chunk = max(256, (1 << 23) // max(len(lab_pal), 1))  # ~64MB per distance block
    idx = np.empty(len(lab_pts), dtype=np.intp)
    for i in range(0, len(lab_pts), chunk):
        d = delta_e(lab_pts[i : i + chunk, None, :], lab_pal[None, :, :])
        idx[i : i + chunk] = d.argmin(axis=1)
    return idx


def _reseed_unused(pal: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Make every palette entry win at least one observed color.

    merge() deliberately keeps a big cluster's centroid fixed when absorbing
    a small one, so the absorbed colors can end up nearest to a THIRD
    centroid; a centroid whose members all defect that way is never chosen
    by quantize() and the output has fewer colors than the palette promises
    (user report: requested 64, output had 60). Runs on the exact colors and
    metric quantize() will use (uint8 palette, Lab); entries that still find
    no color (image has fewer distinct colors than requested) are dropped.
    Used entries are never moved — this preserves the anti-drag calibration.
    """
    uniq, counts = np.unique(colors.reshape(-1, 3), axis=0, return_counts=True)
    lab_u = rgb_to_lab(uniq)
    for _ in range(4):
        lab_p = rgb_to_lab(pal)
        idx = _nearest(lab_u, lab_p)
        used = np.bincount(idx, minlength=len(pal)) > 0
        if used.all():
            break
        score = delta_e(lab_u, lab_p[idx]) * counts  # error mass a reseed removes
        for slot in np.where(~used)[0]:
            j = int(np.argmax(score))
            if score[j] <= 0:
                break
            pal[slot] = uniq[j]
            score[j] = -1.0
    idx = _nearest(lab_u, rgb_to_lab(pal))
    mass = np.bincount(idx, weights=counts.astype(np.float64), minlength=len(pal))
    keep = np.where(mass > 0)[0]
    return pal[keep[np.argsort(mass[keep])[::-1]]]


def quantize(img: np.ndarray, palette: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Snap every pixel to the nearest palette color (in Lab).

    Returns (quantized uint8 image, palette-index array).
    """
    lab_img = rgb_to_lab(img).reshape(-1, 1, 3)
    lab_pal = rgb_to_lab(palette).reshape(1, -1, 3)
    idx = delta_e(lab_img, lab_pal).argmin(axis=1)
    out = palette[idx].reshape(img.shape).astype(np.uint8)
    return out, idx.reshape(img.shape[:-1])


def image_palette(img: np.ndarray) -> np.ndarray:
    """All distinct colors in an image, most frequent first."""
    uniq, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    return uniq[np.argsort(counts)[::-1]].astype(np.uint8)


def dedupe_palette(palette: np.ndarray, min_de: float = 2.0) -> np.ndarray:
    """Drop palette entries closer than min_de to an earlier entry.

    Forced palettes in the wild contain near-duplicates (a real user palette
    repeats whole ramps with ~1-2 ΔE offsets; its global nearest-neighbor
    floor measured 0.4 ΔE). An observed color that falls between two such
    twins turns entry choice into a coin flip that per-pixel noise decides —
    adjacent regions of one flat color then land on different (visibly
    different-enough) entries. Entries this close are indistinguishable
    anyway; keeping the first of each twin group removes the tie class.
    """
    lab = rgb_to_lab(palette)
    keep: list[int] = []
    for i in range(len(palette)):
        if not keep or float(delta_e(lab[i], lab[keep]).min()) >= min_de:
            keep.append(i)
    return palette[keep]


def parse_hex_palette(spec: str) -> np.ndarray:
    """Parse '#rrggbb,#rrggbb,...' into a (K, 3) uint8 palette."""
    entries = [e.strip().lstrip("#") for e in spec.split(",") if e.strip()]
    if not entries:
        raise ValueError("empty palette spec")
    return np.array(
        [[int(e[i : i + 2], 16) for i in (0, 2, 4)] for e in entries], dtype=np.uint8
    )
