"""Grid pitch/phase estimation for pseudo-pixel-art images.

A pseudo-pixel-art image renders each "fake pixel" (cell) as a roughly
pitch x pitch block of real pixels. We estimate the pitch and phase per axis
from the gradient profile: cell boundaries produce elevated gradients at a
(mostly) periodic set of positions.

Coordinate convention: a vertical boundary between columns x-1 and x has
coordinate x, so boundary coordinates live in [0, W] (analogously [0, H]).
The gradient profile index i corresponds to boundary coordinate i + 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Minimum Rayleigh z-score (see _rayleigh) to accept a grid along an axis.
# Below it we assume the image has no fake-pixel grid (it may already be at
# true resolution). Calibrated on the diagnostic suite: true grids score
# >~6 even for sparse sprites on flat backgrounds; gridless images stay <~4.
Z_MIN = 5.0


@dataclass
class AxisGrid:
    pitch: float
    phase: float  # boundary coordinate in [0, pitch)
    score: float  # Rayleigh z-score of the detection (>= Z_MIN)
    edges: np.ndarray  # sorted boundary coordinates, includes 0 and the axis size


@dataclass
class Grid:
    y: AxisGrid  # horizontal boundaries (row edges)
    x: AxisGrid  # vertical boundaries (column edges)

    @property
    def shape(self) -> tuple[int, int]:
        """Output (rows, cols) of the true-resolution image."""
        return len(self.y.edges) - 1, len(self.x.edges) - 1


def gradient_profile(img: np.ndarray, axis: int, denoise: bool = True) -> np.ndarray:
    """Mean absolute gradient across each candidate boundary along `axis`.

    With denoise=True the per-pixel gradients are soft-thresholded at 1.5x
    their median first. For sparse content (a sprite on a flat background)
    the profile is otherwise dominated by sensor/generation noise, which
    contributes gradient everywhere; the median estimates that noise floor
    because boundary pixels are a minority.
    """
    g = np.abs(np.diff(img.astype(np.float64), axis=axis)).sum(axis=2)
    if denoise:
        g = np.maximum(g - 1.5 * np.median(g), 0.0)
    return g.mean(axis=1 - axis)


def _fft_pitch_candidates(
    profile: np.ndarray, min_pitch: float, max_pitch: float, n_peaks: int = 6
) -> list[float]:
    """Candidate pitches from the power spectrum of the gradient profile.

    A pitch-p boundary comb concentrates power at frequency 1/p (and its
    harmonics). Crucially the spectral peak width (1/L) matches the pitch
    accuracy achievable over a length-L profile, so unlike a coarse pitch
    scan this cannot step over a narrow fundamental. Each spectral peak at f
    is expanded into pitches k/f (the peak may be the k-th harmonic); all
    candidates are re-scored in the spatial domain afterwards.
    """
    d = profile - profile.mean()
    nfft = 8 * (1 << int(np.ceil(np.log2(max(len(d), 2)))))
    power = np.abs(np.fft.rfft(d, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0)

    band = (freqs >= 1.0 / max_pitch) & (freqs <= 0.5)
    idx = np.where(band)[0]
    if len(idx) < 3:
        return []
    p_band = power[idx]
    local_max = np.ones(len(idx), dtype=bool)
    local_max[1:] &= p_band[1:] >= p_band[:-1]
    local_max[:-1] &= p_band[:-1] >= p_band[1:]
    peaks = idx[local_max]
    peaks = peaks[np.argsort(power[peaks])[::-1][:n_peaks]]

    candidates: list[float] = []
    for f in freqs[peaks]:
        # a peak at f may be the k-th harmonic of the fundamental (pitch k/f),
        # or the super-period of alternating cell widths (k=0.5: pitch 3.5
        # rasterizes to 3,4,3,4 whose strongest period is 7)
        for k in (0.5, 1.0, 2.0, 3.0):
            pitch = k / f
            if min_pitch <= pitch <= max_pitch and not any(
                abs(pitch - c) < 0.02 for c in candidates
            ):
                candidates.append(float(pitch))
    return candidates


def _peak_impulses(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse each gradient bump to one subpixel impulse (non-max suppression).

    Resampling corruption smears a boundary's gradient over several columns —
    at small pitches over most of the cell — which spreads its phase around
    the circle and destroys Rayleigh concentration even for perfectly aligned
    evidence. NMS restores a sharp comb: one impulse per bump at its
    parabolically-refined peak, weighted by the local 3-tap mass (mass, not
    height, so broad real bumps outweigh sharp noise spikes).

    Returns (positions in profile-index space, weights).
    """
    p = profile
    n = len(p)
    if n < 3:
        return np.empty(0), np.empty(0)
    is_max = np.zeros(n, dtype=bool)
    is_max[1:-1] = (p[1:-1] >= p[:-2]) & (p[1:-1] > p[2:]) & (p[1:-1] > 0)
    is_max[0] = p[0] > p[1] and p[0] > 0
    is_max[-1] = p[-1] > p[-2] and p[-1] > 0
    idx = np.where(is_max)[0]
    if len(idx) == 0:
        return np.empty(0), np.empty(0)

    pos = idx.astype(np.float64)
    weights = p[idx].copy()
    interior = (idx > 0) & (idx < n - 1)
    i = idx[interior]
    if len(i):
        weights[interior] = p[i - 1] + p[i] + p[i + 1]
        denom = p[i - 1] - 2.0 * p[i] + p[i + 1]
        shift = np.zeros(len(i))
        concave = denom < 0
        shift[concave] = 0.5 * (p[i - 1] - p[i + 1])[concave] / denom[concave]
        pos[interior] = i + np.clip(shift, -0.5, 0.5)
    return pos, weights


def _rayleigh(pos: np.ndarray, weights: np.ndarray, pitch: float) -> tuple[float, float]:
    """Rayleigh test of boundary-evidence impulses against a pitch-p comb.

    Each impulse is placed on the phase circle 2*pi*x/pitch and we measure
    the concentration of the weighted resultant: z = n_eff * R^2 with R the
    resultant length and n_eff = (sum w)^2 / sum w^2 the effective number of
    independent evidence columns. Under the no-grid null z is ~Exp(1); a real
    grid concentrates all boundary energy at one phase and scores far higher.

    Why this beats scanning boundary positions directly:
    - sparse content (a sprite on a flat background) has boundary evidence in
      only a few columns; n_eff counts exactly that evidence instead of
      diluting it over every cell of the comb,
    - harmonics (2p) collapse: true boundaries land antiphase and cancel,
    - the phase estimate is free (the resultant angle) and continuous.

    Sub-harmonics (p/2) score as well as p and are handled by the
    largest-pitch selection rule (_largest) in estimate_grid.

    Returns (z, phase) with phase the boundary coordinate in [0, pitch).
    """
    total = float(weights.sum())
    if total <= 0 or len(pos) < 4:
        return 0.0, 0.0
    theta = 2.0 * np.pi * (pos + 1.0) / pitch  # profile index -> boundary coordinate
    c = float((weights * np.cos(theta)).sum())
    s = float((weights * np.sin(theta)).sum())
    resultant = np.hypot(c, s) / total
    n_eff = total * total / (float((weights * weights).sum()) + 1e-12)
    z = n_eff * resultant * resultant
    phase = (np.arctan2(s, c) / (2.0 * np.pi) * pitch) % pitch
    return float(z), float(phase)


def _refine_edges(
    pos: np.ndarray,
    weights: np.ndarray,
    pitch: float,
    phase: float,
    size: int,
    search_frac: float = 0.3,
    weight_floor: float = 0.25,
) -> np.ndarray:
    """Snap comb boundaries to nearby strong evidence impulses.

    Snapping absorbs mild grid wobble and accumulated pitch error. Only
    impulses carrying at least `weight_floor` of the maximum impulse mass
    count as evidence: flat regions provide none, and there the comb position
    is kept rather than chasing noise bumps. Border cells may legitimately be
    partial (the corruption's sub-cell phase offset crops them); anything
    narrower than 1 px is merged. Returns boundary coordinates including the
    outer edges 0 and size.
    """
    strong = pos[weights >= weight_floor * weights.max()] + 1.0 if len(pos) else np.empty(0)
    window = max(1.0, pitch * search_frac)
    edges: list[float] = [0.0]
    for coord in np.arange(phase, size + 1e-6, pitch):
        if coord < 1.0 or coord > size - 1.0:
            continue
        snapped = float(coord)
        if len(strong):
            j = int(np.argmin(np.abs(strong - coord)))
            if abs(strong[j] - coord) <= window:
                snapped = float(strong[j])
        if snapped - edges[-1] >= 1.0:
            edges.append(snapped)
    if size - edges[-1] < 1.0 and len(edges) > 1:
        edges[-1] = float(size)  # merge a sub-pixel trailing sliver
    else:
        edges.append(float(size))
    return np.asarray(edges)


def _axis_candidates(
    img: np.ndarray,
    axis: int,
    min_pitch: float = 1.6,
    max_pitch: float = 32.0,
    z_min: float = Z_MIN,
    rel_tol: float = 0.75,
):
    """Near-best pitch candidates for one axis.

    Candidate pitches come from the FFT of the gradient profile (a coarse
    pitch scan would step over the narrow fundamental peaks); each candidate
    is locally refined and scored with the Rayleigh test, then grouped into
    scale clusters (p and p/2 are distinct scales; neighbors within a
    refinement span are not) represented by their argmax-z member.

    Returns (near-best clusters as (pitch, z, phase), impulse positions,
    impulse weights, axis size) or None if no grid evidence.
    """
    profile = gradient_profile(img, axis)
    n = len(profile)
    size = n + 1
    max_pitch = min(max_pitch, size / 4)
    if max_pitch <= min_pitch or n < 8 or profile.max() <= 1e-9:
        return None

    pos, weights = _peak_impulses(profile)
    if len(pos) < 4:
        return None
    raster = np.zeros(n)
    np.add.at(raster, np.clip(np.round(pos).astype(int), 0, n - 1), weights)

    scored: list[tuple[float, float, float]] = []  # (pitch, z, phase)
    for cand in _fft_pitch_candidates(raster, min_pitch, max_pitch):
        # local refinement: a pitch-p comb over length n loses phase
        # coherence after ~p^2/n of pitch error, so the scan is that fine
        span = max(0.05, 3.0 * cand * cand / (8 * n))
        for p in np.linspace(max(min_pitch, cand - span), min(max_pitch, cand + span), 13):
            z, phase = _rayleigh(pos, weights, p)
            scored.append((float(p), z, phase))
    if not scored:
        return None

    z_best = max(z for _, z, _ in scored)
    if z_best < z_min:
        return None

    scored.sort(key=lambda t: t[0])
    clusters: list[tuple[float, float, float]] = []
    for entry in scored:
        if clusters and entry[0] < clusters[-1][0] * 1.3:
            if entry[1] > clusters[-1][1]:
                clusters[-1] = entry
        else:
            clusters.append(entry)
    near = [c for c in clusters if c[1] >= rel_tol * z_best]
    return near, pos, weights, size


def _axis_grid(cand, pick) -> AxisGrid:
    near, pos, weights, size = cand
    pitch, z, phase = pick
    return AxisGrid(
        pitch=pitch, phase=phase, score=z,
        edges=_refine_edges(pos, weights, pitch, phase, size),
    )


def _largest(near):
    """The classic per-axis rule: largest near-best pitch. A true pitch p
    and its sub-harmonic p/2 are phase-coherent and score alike, while the
    harmonic 2p self-cancels, so the largest near-best pitch is the
    fundamental."""
    return max(near, key=lambda t: t[0])


def _harmonic(a: float, b: float, tol: float = 0.05) -> bool:
    """True when a and b belong to one comb family (integer ratio, either
    direction — covers both the sub-harmonic p/2 and the harmonic 2p)."""
    r = max(a, b) / min(a, b)
    k = round(r)
    return k >= 1 and abs(r - k) <= tol * k


def _rescore(cand, pitch: float) -> tuple[float, float, float]:
    """Best (pitch, z, phase) near `pitch` on this axis's impulse evidence."""
    _, pos, weights, size = cand
    n = size - 1
    span = max(0.05, 3.0 * pitch * pitch / (8 * n))
    best = (pitch, 0.0, 0.0)
    for p in np.linspace(pitch - span, pitch + span, 13):
        z, phase = _rayleigh(pos, weights, p)
        if z > best[1]:
            best = (float(p), z, phase)
    return best


def estimate_grid(img: np.ndarray, **kwargs) -> Grid | None:
    """Estimate the 2D cell grid of a pseudo-pixel-art image (HxWx3 uint8).

    Axes are scored independently, but the pitch pair is chosen JOINTLY:
    fake pixels are square in practice, so if any near-best (y, x) pair is
    square-consistent, only such pairs compete — the top pair by combined z
    anchors a harmonic family, and the largest family member wins (the
    p vs p/2 sub-harmonic rule, applied to pairs). This lets one confident
    axis veto the other axis's spurious scales, which otherwise make the
    grid too coarse (detail loss: a near-best 4.32 beating a higher-scoring
    true 3.00 on blurry art) or distort the aspect. Without any square pair
    the axes fall back to the independent largest-near-best rule.

    Champion veto: decorative periodicity (scanlines, sprite-sheet frames)
    can own one axis's near-best set outright and drag the chosen pair to
    its scale (04.png: scanlines at y 3.58 paired with x 3.59 while the art
    grid at x 2.0 out-scored both). When an axis's top scorer is UNRELATED
    to the chosen scale and strictly stronger, and its pitch completes to a
    square pair on the other axis with detectable evidence (z >= z_min),
    the completed square pair wins.
    """
    cy = _axis_candidates(img, 0, **kwargs)
    cx = _axis_candidates(img, 1, **kwargs)
    if cy is None or cx is None:
        return None
    pairs = [
        (a, b) for a in cy[0] for b in cx[0] if abs(a[0] / b[0] - 1.0) <= 0.05
    ]
    if pairs:
        champion = max(pairs, key=lambda p: p[0][1] + p[1][1])
        base = (champion[0][0] + champion[1][0]) / 2.0

        def in_family(pair) -> bool:
            m = (pair[0][0] + pair[1][0]) / 2.0
            k = round(m / base)
            return k >= 1 and abs(m / base - k) <= 0.05 * k

        py, px = max(
            (p for p in pairs if in_family(p)), key=lambda p: p[0][0] + p[1][0]
        )
    else:
        py, px = _largest(cy[0]), _largest(cx[0])

    z_min = kwargs.get("z_min", Z_MIN)
    veto = None  # (champ z, own pick, completed pick, own axis is y?)
    for own, other, own_pick, is_y in ((cy, cx, py, True), (cx, cy, px, False)):
        champ = max(own[0], key=lambda t: t[1])
        if _harmonic(champ[0], own_pick[0]) or champ[1] <= own_pick[1]:
            continue
        completed = _rescore(other, champ[0])
        if completed[1] >= z_min and (veto is None or champ[1] > veto[0]):
            veto = (champ[1], champ, completed, is_y)
    if veto is not None:
        _, champ, completed, is_y = veto
        py, px = (champ, completed) if is_y else (completed, champ)

    return Grid(y=_axis_grid(cy, py), x=_axis_grid(cx, px))


def regrain(grid: Grid, factor: float) -> Grid:
    """Change output granularity relative to the detected grid.

    factor 2 subdivides every cell into 2x2 (finer output), factor 0.5
    merges 2x2 cells into one (coarser). Factors are rounded to integers —
    non-integer factors would place boundaries incommensurate with the
    detected grid and every output cell would mix two source cells — and
    subdivision is capped so cells keep >= 1 px per axis. The refined
    (wobbly) boundaries are preserved; sub-boundaries interpolate them.
    """
    if factor <= 0:
        return grid
    if factor >= 1:
        k = int(round(factor))
        k = min(k, int(min(grid.y.pitch, grid.x.pitch)))  # keep cells >= 1 px
        if k <= 1:
            return grid
        return Grid(y=_subdivide_axis(grid.y, k), x=_subdivide_axis(grid.x, k))
    m = int(round(1.0 / factor))
    if m <= 1:
        return grid
    return Grid(y=_merge_axis(grid.y, m), x=_merge_axis(grid.x, m))


def _subdivide_axis(axis: AxisGrid, k: int) -> AxisGrid:
    widths = np.diff(axis.edges)
    sub = axis.edges[:-1, None] + widths[:, None] * (np.arange(k) / k)
    edges = np.append(sub.ravel(), axis.edges[-1])
    return AxisGrid(pitch=axis.pitch / k, phase=axis.phase, score=axis.score, edges=edges)


def _merge_axis(axis: AxisGrid, m: int) -> AxisGrid:
    edges = axis.edges[::m]
    if edges[-1] != axis.edges[-1]:
        edges = np.append(edges, axis.edges[-1])
    return AxisGrid(pitch=axis.pitch * m, phase=axis.phase, score=axis.score, edges=edges)


def uniform_axis_grid(size: int, pitch: float, phase: float = 0.0) -> AxisGrid:
    """Build an AxisGrid with strictly uniform edges (for a known pitch)."""
    interior = np.arange(phase, size + 1e-6, pitch)
    interior = interior[(interior >= 1.0) & (interior <= size - 1.0)]
    edges = np.concatenate([[0.0], interior, [float(size)]])
    edges = edges[np.concatenate([[True], np.diff(edges) >= 1.0])]
    if edges[-1] != size:
        edges = np.append(edges, float(size))
    return AxisGrid(pitch=pitch, phase=phase, score=float("nan"), edges=edges)
