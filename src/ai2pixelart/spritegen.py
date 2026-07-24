"""Procedural clean pixel-art sprites for training data.

Real pixel-art collections are the better long-term source (no generator
bias), but a procedural generator gives unlimited, license-free clean ground
truth with guaranteed coverage of the structures the restoration model must
preserve: limited distinct palettes, 1-px outlines, shading ramps, dithered
transitions, and isolated single-pixel details (the "white eye" case).
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from scipy import ndimage

from ai2pixelart.palette import delta_e, rgb_to_lab

# Colors closer than this are ambiguous training targets. Deliberately low:
# real palettes contain close shades (several darks especially), and the net
# must learn to discriminate them — v1 sprites at min 7.0 never taught that,
# and the net then confused dark background entries on real images.
MIN_PALETTE_DE = 4.0


def _hsv(h: float, s: float, v: float) -> tuple[int, int, int]:
    return tuple(int(round(c * 255)) for c in colorsys.hsv_to_rgb(h % 1.0, s, v))


def _parts_ok(parts: dict) -> bool:
    pal = np.concatenate([np.atleast_2d(v) for v in parts.values() if len(v)])
    lab = rgb_to_lab(pal.astype(np.uint8))
    d = delta_e(lab[:, None, :], lab[None, :, :])
    d[np.diag_indices(len(pal))] = np.inf
    return d.min() >= MIN_PALETTE_DE


def random_parts(rng: np.random.Generator) -> dict:
    """Structured palette: bg (+ optional second dark), outline, 1-2 hue
    ramps, 1-2 accents. Up to ~15 colors, min pairwise MIN_PALETTE_DE."""
    for _ in range(32):
        hue = rng.uniform()
        hues = [hue]
        if rng.random() < 0.45:
            # second ramp: usually a distinct hue; sometimes a NEAR hue —
            # area-scale near-hue families (yellow-green skin vs yellow
            # hair) are what the net must discriminate on real images, and
            # v1/v2 sprites never contained them (measured hallucination:
            # nets recolored skin with the hair entry on 15.png)
            off = rng.uniform(0.04, 0.10) if rng.random() < 0.35 else rng.uniform(0.15, 0.55)
            hues.append((hue + off) % 1.0)
        ramps = []
        for h in hues:
            n = int(rng.integers(3, 6))
            ramps.append(np.array([
                _hsv(h + rng.normal(0, 0.02), rng.uniform(0.45, 0.9), v)
                for v in np.linspace(rng.uniform(0.28, 0.40), rng.uniform(0.72, 0.95), n)
            ], dtype=np.uint8))
        parts = {
            "bg": np.array([_hsv(hue + rng.uniform(0.25, 0.75), rng.uniform(0.15, 0.55),
                                 rng.uniform(0.08, 0.28))], dtype=np.uint8),
            # a second dark teaches dark-vs-dark discrimination (UI frames,
            # vignettes) — the exact failure seen on real images
            "dark2": np.array([_hsv(rng.uniform(), rng.uniform(0.2, 0.6),
                                    rng.uniform(0.10, 0.30))], dtype=np.uint8)
            if rng.random() < 0.55 else np.empty((0, 3), dtype=np.uint8),
            "outline": np.array([_hsv(hue + rng.normal(0, 0.1), rng.uniform(0.2, 0.6),
                                      rng.uniform(0.02, 0.12))], dtype=np.uint8),
            "ramps": np.concatenate(ramps),
            "accents": np.array(
                [_hsv(rng.uniform(), rng.uniform(0.0, 0.3), rng.uniform(0.88, 1.0))]
                + ([_hsv(hue + 0.5 + rng.normal(0, 0.05), rng.uniform(0.7, 1.0), rng.uniform(0.6, 0.9))]
                   if rng.random() < 0.5 else []),
                dtype=np.uint8),
        }
        parts["_ramp_sizes"] = [len(r) for r in ramps]
        if _parts_ok({k: v for k, v in parts.items() if not k.startswith("_")}):
            return parts
    raise RuntimeError("could not sample a well-separated palette")


def random_palette(rng: np.random.Generator) -> np.ndarray:
    parts = random_parts(rng)
    return np.concatenate([v for k, v in parts.items() if not k.startswith("_") and len(v)])


def load_palette_pool(path) -> list[np.ndarray]:
    """data/palettes_real.json -> list of (K, 3) uint8 palettes."""
    import json

    return [np.asarray(p, dtype=np.uint8) for p in json.loads(Path(path).read_text())]


def build_palette_pool(image_paths, out_json, progress=None) -> int:
    """Extract real-image palettes (the classical pipeline's own proposals,
    i.e. exactly the distribution the net sees at inference) into a pool."""
    import json

    from ai2pixelart.pipeline import clean

    pool = []
    for p in image_paths:
        from PIL import Image

        img = np.array(Image.open(p).convert("RGB"))
        result = clean(img, max_colors=16)
        pool.append([c.tolist() for c in result.palette])
        if progress:
            progress(f"{p}: {len(result.palette)} colors")
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(pool))
    return len(pool)


def _pool_parts(rng: np.random.Generator, palette: np.ndarray) -> dict | None:
    """Map a real-image palette onto the sprite part structure.

    Pool palettes intentionally BYPASS the MIN_PALETTE_DE filter: close
    shades from real images are the target distribution (supervision stays
    exact — targets are index maps over the drawn colors), only sub-2 ΔE
    near-duplicates are dropped as pointless.
    """
    pal = np.unique(palette, axis=0)
    lab = rgb_to_lab(pal)
    keep: list[int] = []
    for i in np.argsort(lab[:, 0]):  # dark -> light
        if all(float(delta_e(lab[i], lab[j])) >= 2.0 for j in keep):
            keep.append(i)
    pal = pal[keep]
    if len(pal) < 5:
        return None
    outline, bg = pal[0], pal[1]
    use_dark2 = len(pal) >= 7 and rng.random() < 0.7
    dark2 = pal[2:3] if use_dark2 else np.empty((0, 3), dtype=np.uint8)
    n_acc = 1 + int(len(pal) >= 8 and rng.random() < 0.5)
    mid = pal[2 + len(dark2) : len(pal) - n_acc]
    if len(mid) < 2:
        return None
    if len(mid) >= 6 and rng.random() < 0.6:  # two ramps, hue-partitioned
        import colorsys

        hues = [colorsys.rgb_to_hsv(*(c / 255.0))[0] for c in mid.astype(np.float64)]
        order = np.argsort(hues)
        half = len(mid) // 2
        ramps = [mid[np.sort(order[:half])], mid[np.sort(order[half:])]]
    else:
        ramps = [mid]
    ramps = [r[:5] for r in ramps if len(r) >= 2]
    if not ramps:
        return None
    return {
        "bg": bg[None],
        "dark2": dark2,
        "outline": outline[None],
        "ramps": np.concatenate(ramps),
        "accents": pal[len(pal) - n_acc :],
        "_ramp_sizes": [len(r) for r in ramps],
    }


def _sample_parts(rng: np.random.Generator, pool: list[np.ndarray] | None) -> dict:
    if pool and rng.random() < 0.5:
        parts = _pool_parts(rng, pool[int(rng.integers(len(pool)))])
        if parts is not None:
            return parts
    return random_parts(rng)


def _blob_mask(rng: np.random.Generator, size: int) -> np.ndarray:
    """Superellipse with radial wobble — an organic sprite body."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cy, cx = rng.uniform(0.42, 0.58, 2) * size
    ry, rx = rng.uniform(0.26, 0.40, 2) * size
    p = rng.uniform(1.6, 3.0)
    theta = np.arctan2(yy - cy, xx - cx)
    wobble = np.ones_like(theta)
    for k in range(2, int(rng.integers(3, 6))):
        wobble += rng.uniform(0.0, 0.09) * np.sin(k * theta + rng.uniform(0, 2 * np.pi))
    r = (np.abs((yy - cy) / ry) ** p + np.abs((xx - cx) / rx) ** p) ** (1.0 / p)
    return r <= wobble


def _interior(mask: np.ndarray, n: int = 1) -> np.ndarray:
    return ndimage.binary_erosion(mask, iterations=n)


def _draw_blob(img, rng, mask, ramp, outline) -> None:
    n_ramp = len(ramp)
    base = n_ramp // 2
    size = img.shape[0]
    img[mask] = ramp[base]

    # shading: darker crescent on one side, lighter on the other
    shift = max(2, size // 8)
    dy, dx = rng.integers(-shift, shift + 1, 2)
    shaded = mask & ~np.roll(mask, (dy, dx), axis=(0, 1))
    img[shaded] = ramp[max(0, base - 1)]
    lit = mask & ~np.roll(mask, (-dy, -dx), axis=(0, 1)) & ~shaded
    img[lit] = ramp[min(n_ramp - 1, base + 1)]

    # dithered transition between base and shaded regions
    if rng.random() < 0.6 and shaded.any():
        band = ndimage.binary_dilation(shaded, iterations=1) & mask & ~shaded
        yy, xx = np.mgrid[0:size, 0:size]
        img[band & ((yy + xx) % 2 == 0)] = ramp[max(0, base - 1)]

    img[mask & ~_interior(mask)] = outline


def sheet_sprite(
    rng: np.random.Generator, pool: list[np.ndarray] | None = None
) -> np.ndarray:
    """A 2x2 tile sheet sharing one palette: per-tile background colors and
    gutter lines between tiles — the layout of character-selection sheets
    (real user assets and several Gemini samples) that single-scene sprites
    never taught."""
    parts = _sample_parts(rng, pool)
    tile = int(rng.integers(28, 45))
    gutter = int(rng.integers(1, 3))
    # per-tile backgrounds drawn from the shared palette's area colors
    bg_pool = [parts["bg"][0]] + [r for r in parts["ramps"]] + list(parts["dark2"])
    gutter_color = parts["outline"][0]
    size = tile * 2 + gutter * 3
    img = np.empty((size, size, 3), dtype=np.uint8)
    img[:] = gutter_color
    for ty in range(2):
        for tx in range(2):
            sub = _scene(rng, tile, {**parts, "bg": np.array([bg_pool[int(rng.integers(len(bg_pool)))]])})
            y0 = gutter + ty * (tile + gutter)
            x0 = gutter + tx * (tile + gutter)
            img[y0 : y0 + tile, x0 : x0 + tile] = sub
    return img


def random_sprite(
    rng: np.random.Generator, size: int | None = None, pool: list[np.ndarray] | None = None
) -> np.ndarray:
    """One clean scene: background structure + 1-2 blobs + patterns + details.

    Occasionally a 2x2 tile sheet instead (see sheet_sprite)."""
    if size is None and rng.random() < 0.25:
        return sheet_sprite(rng, pool)
    size = size or int(rng.integers(32, 65))
    parts = _sample_parts(rng, pool)
    return _scene(rng, size, parts)


def _scene(rng: np.random.Generator, size: int, parts: dict) -> np.ndarray:
    bg, dark2, outline = parts["bg"][0], parts["dark2"], parts["outline"][0]
    accents = parts["accents"]
    ramp_sizes = parts["_ramp_sizes"]
    ramps = np.split(parts["ramps"], np.cumsum(ramp_sizes))[: len(ramp_sizes)]

    img = np.empty((size, size, 3), dtype=np.uint8)
    img[:] = bg

    # background structure in the second dark: border frame or split — the
    # kind of near-dark adjacency real UI-styled art has everywhere
    if len(dark2):
        yy, xx = np.mgrid[0:size, 0:size]
        kind = rng.integers(0, 3)
        if kind == 0:  # frame
            t = int(rng.integers(1, max(2, size // 10)))
            frame = (yy < t) | (yy >= size - t) | (xx < t) | (xx >= size - t)
            img[frame] = dark2[0]
        elif kind == 1:  # half split
            img[yy < int(rng.integers(size // 4, 3 * size // 4))] = dark2[0]
        else:  # horizontal stripes band
            y0 = int(rng.integers(0, size // 2))
            h = int(rng.integers(size // 6, size // 3))
            band = (yy >= y0) & (yy < y0 + h)
            img[band & (yy % 2 == 0)] = dark2[0]

    mask_all = np.zeros((size, size), dtype=bool)
    # when two ramps exist, usually draw both blobs so near-hue ramp pairs
    # actually appear as adjacent areas in the image
    n_blobs = 1 + (rng.random() < 0.7 and len(ramps) > 1)
    for b in range(n_blobs):
        mask = _blob_mask(rng, size)
        if mask.sum() < size * size * 0.08:  # degenerate draw — plain fallback blob
            yy, xx = np.mgrid[0:size, 0:size]
            mask = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= (size * 0.33) ** 2
        mask &= ~mask_all
        _draw_blob(img, rng, mask, ramps[min(b, len(ramps) - 1)], outline)
        mask_all |= mask
    mask = mask_all

    # isolated 1-px details on uniform interior ground (guaranteed >= 1)
    deep = _interior(mask, 3)
    candidates = np.argwhere(deep)
    n_details = int(rng.integers(1, 4))
    placed = 0
    rng.shuffle(candidates)
    for y, x in candidates:
        if placed >= n_details:
            break
        patch = img[y - 1 : y + 2, x - 1 : x + 2].reshape(-1, 3)
        if len(np.unique(patch, axis=0)) == 1:  # uniform 3x3 -> stays isolated
            img[y, x] = accents[placed % len(accents)]
            placed += 1
    if placed == 0:  # force one at the deepest interior point
        dist = ndimage.distance_transform_edt(mask)
        y, x = np.unravel_index(int(dist.argmax()), dist.shape)
        img[y, x] = accents[0]

    return img


def generate_sprites(outdir, n: int, seed: int = 0, progress=None, pool_path=None) -> list:
    """Write n random sprites as PNGs; returns the paths."""
    from PIL import Image

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pool = load_palette_pool(pool_path) if pool_path else None
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        img = random_sprite(rng, pool=pool)
        p = outdir / f"sprite_{i:05d}.png"
        Image.fromarray(img).save(p)
        paths.append(p)
        if progress and (i + 1) % 200 == 0:
            progress(f"{i + 1}/{n} sprites")
    return paths
