"""Training-pair generation: clean pixel art -> VAE-roundtripped corruption.

Each pair is (true-resolution clean art, corrupted rendering at a random
non-integer scale and sub-cell phase). Because the corruption is a pure
autoencoder roundtrip, alignment is exact: output pixel x belongs to source
cell floor-nearest((x + 0.5 + phase) / scale), reconstructible from the
metadata written to pairs.jsonl.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ai2pixelart.corrupt import upscale


def _shading_field(shape: tuple, rng: np.random.Generator, amp: float = 0.06) -> np.ndarray:
    """Random low-frequency brightness field (linear gradient + vignette).

    Real AI images shade their 'flat' backgrounds with subtle vignettes
    (measured ~14 RGB across a 1024px Gemini sample); nets trained only on
    perfectly flat sprite backgrounds faithfully SPLIT such gradients into
    multiple palette bands instead of collapsing them. Applying this field
    to the corruption (targets stay flat) teaches the pixel-art convention:
    a vignetted background is still one color.
    """
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    gx, gy = rng.uniform(-1, 1, 2)
    lin = gx * (xx / w - 0.5) + gy * (yy / h - 0.5)
    cy, cx = rng.uniform(0.3, 0.7, 2)
    rad = np.hypot(yy / h - cy, xx / w - cx)
    field = lin * rng.uniform(0.3, 1.0) - rad * rng.uniform(0.0, 1.0)
    field /= np.abs(field).max() + 1e-9
    return (field * amp * 255.0 * rng.uniform(0.3, 1.0))[:, :, None]


def _box_downscale2(img: np.ndarray) -> np.ndarray:
    """Exact 2x box downscale (crops odd trailing row/col first)."""
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    im = img[:h, :w].astype(np.float64)
    return np.round(im.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))).astype(np.uint8)


def generate_edit_pairs(
    src_paths: list[Path],
    outdir: Path,
    pipe,
    editor_name: str,
    n_per_image: int = 1,
    scale_range: tuple[float, float] = (3.0, 12.0),
    strength_range: tuple[float, float] = (0.15, 0.45),
    anisotropy: float = 0.03,
    seed: int = 0,
    max_side: int = 1408,
    keep_de: float = 14.0,
    min_valid: float = 0.7,
    downscale_frac: float = 0.0,
    fine_scale_range: tuple[float, float] = (4.2, 6.4),
    shade_prob: float = 0.0,
    shade_amp: float = 0.06,
    noise_prob: float = 0.0,
    noise_sigma: tuple[float, float] = (3.0, 14.0),
    corrupt_fn=None,
    progress=None,
) -> list[dict]:
    """Rail-guarded img2img pairs: clean -> upscale -> generative re-render.

    Unlike the VAE roundtrip, the editor may move content, so every pair
    carries a per-cell validity mask (rails.validity_mask): drifted cells
    are excluded from the loss, added interior detail stays as gold signal.
    Pairs with less than `min_valid` valid cells are rejected outright.

    downscale_frac: fraction of pairs corrupted at `fine_scale_range` and
    then box-downscaled 2x. Cells below ~3 px dissolve in f8 latents, so
    the fine-pitch regime (~2-3 px cells — where the nets recolor real
    images) cannot be corrupted directly; corrupting at 4.2-6.4 and halving
    afterwards yields realistic 2.1-3.2 px pairs, and the alignment
    metadata transforms exactly (scale/2, phase/2 — verified: the halved
    mapping reproduces the source cells bit-exactly for a 2x box filter).
    """
    from ai2pixelart.imgedit import edit_corrupt
    from ai2pixelart.rails import validity_mask

    corrupt_fn = corrupt_fn or edit_corrupt
    outdir = Path(outdir)
    (outdir / "imgs").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    metas: list[dict] = []
    rejected = 0

    with open(outdir / "pairs.jsonl", "a") as manifest:
        for path in src_paths:
            clean = np.array(Image.open(path).convert("RGB"))
            clean_rel = f"imgs/{path.stem}.clean.png"
            Image.fromarray(clean).save(outdir / clean_rel)

            for i in range(n_per_image):
                fine = rng.random() < downscale_frac
                s = float(rng.uniform(*(fine_scale_range if fine else scale_range)))
                a = float(rng.uniform(1.0 - anisotropy, 1.0 + anisotropy))
                sy, sx = s * a, s / a
                if max(clean.shape[0] * sy, clean.shape[1] * sx) > max_side:
                    shrink = max_side / max(clean.shape[0] * sy, clean.shape[1] * sx)
                    sy, sx = sy * shrink, sx * shrink
                # fine pairs keep >= 3 px border presence so the halved pair
                # keeps the usual >= 1.5 px cell-completeness margin
                margin = 3.0 if fine else 1.5
                phase = (
                    float(rng.uniform(0, max(sy - margin, 0.0))),
                    float(rng.uniform(0, max(sx - margin, 0.0))),
                )
                strength = float(rng.uniform(*strength_range))
                pair_seed = int(rng.integers(0, 2**31))

                big = upscale(clean, scale=(sy, sx), phase=phase, interp="nearest")
                shaded = rng.random() < shade_prob
                if shaded:
                    big = np.clip(
                        big.astype(np.float64) + _shading_field(big.shape, rng, shade_amp), 0, 255
                    ).astype(np.uint8)
                corrupted = corrupt_fn(big, pipe, strength=strength, seed=pair_seed)
                if fine:
                    corrupted = _box_downscale2(corrupted)
                    sy, sx = sy / 2.0, sx / 2.0
                    phase = (phase[0] / 2.0, phase[1] / 2.0)
                sigma = 0.0
                if rng.random() < noise_prob:
                    # real user assets carry heavy per-pixel noise on
                    # visually-flat regions (measured std ~16 RGB); targets
                    # stay clean — teaches noise collapse at this amplitude
                    sigma = float(rng.uniform(*noise_sigma))
                    corrupted = np.clip(
                        corrupted.astype(np.float64) + rng.normal(0, sigma, corrupted.shape),
                        0, 255,
                    ).astype(np.uint8)

                meta = {
                    "clean": clean_rel,
                    "corrupt": f"imgs/{path.stem}.{i:02d}.png",
                    "scale_y": sy,
                    "scale_x": sx,
                    "phase_y": phase[0],
                    "phase_x": phase[1],
                    "corruption": "img2img",
                    "editor": editor_name,
                    "strength": strength,
                    "downscaled": fine,
                    "shaded": shaded,
                    "noise_sigma": round(sigma, 2),
                    "seed": pair_seed,
                }
                mask = validity_mask(clean, corrupted, meta, keep_de=keep_de)
                valid_frac = float(mask.mean())
                if valid_frac < min_valid:
                    rejected += 1
                    continue
                meta["mask"] = f"imgs/{path.stem}.{i:02d}.mask.png"
                meta["valid_frac"] = round(valid_frac, 4)
                Image.fromarray(corrupted).save(outdir / meta["corrupt"])
                Image.fromarray((mask * 255).astype(np.uint8)).save(outdir / meta["mask"])
                manifest.write(json.dumps(meta) + "\n")
                metas.append(meta)
            if progress:
                progress(f"{path.stem}: {len(metas)} pairs ({rejected} rejected)")
    return metas


@dataclass
class PairMeta:
    clean: str  # path of the true-resolution ground truth, relative to outdir
    corrupt: str  # path of the corrupted rendering, relative to outdir
    scale_y: float
    scale_x: float
    phase_y: float
    phase_x: float
    vae: str
    seed: int


def generate_pairs(
    src_paths: list[Path],
    outdir: Path,
    vae,
    vae_name: str,
    n_per_image: int = 4,
    scale_range: tuple[float, float] = (3.0, 8.0),
    anisotropy: float = 0.03,
    seed: int = 0,
    max_side: int = 2048,
) -> list[PairMeta]:
    """Write (clean, corrupt) pairs plus a pairs.jsonl manifest to `outdir`.

    anisotropy: relative y/x scale mismatch, matching the slightly
    non-square fake pixels AI images exhibit.

    scale_range floor of 3.0: the f8 VAE cannot represent cells much below
    3 px — they dissolve rather than wobble (measured: grid undetectable at
    scale 2.6, exact at 3.3+).

    Phases are drawn from [0, scale - 1.5] so the first source row/column
    keeps at least ~1.5 px of presence in the corrupted image — pairs stay
    cell-complete (every GT cell appears), which downstream loaders and the
    classical evaluation both rely on.
    """
    from ai2pixelart.vae import vae_roundtrip

    outdir = Path(outdir)
    (outdir / "imgs").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    metas: list[PairMeta] = []

    with open(outdir / "pairs.jsonl", "a") as manifest:
        for path in src_paths:
            clean = np.array(Image.open(path).convert("RGB"))
            clean_rel = f"imgs/{path.stem}.clean.png"
            Image.fromarray(clean).save(outdir / clean_rel)

            for i in range(n_per_image):
                s = float(rng.uniform(*scale_range))
                a = float(rng.uniform(1.0 - anisotropy, 1.0 + anisotropy))
                sy, sx = s * a, s / a
                if max(clean.shape[0] * sy, clean.shape[1] * sx) > max_side:
                    shrink = max_side / max(clean.shape[0] * sy, clean.shape[1] * sx)
                    sy, sx = sy * shrink, sx * shrink
                phase = (
                    float(rng.uniform(0, max(sy - 1.5, 0.0))),
                    float(rng.uniform(0, max(sx - 1.5, 0.0))),
                )
                pair_seed = int(rng.integers(0, 2**31))

                big = upscale(clean, scale=(sy, sx), phase=phase, interp="nearest")
                corrupted = vae_roundtrip(big, vae, sample=True, seed=pair_seed)

                meta = PairMeta(
                    clean=clean_rel,
                    corrupt=f"imgs/{path.stem}.{i:02d}.png",
                    scale_y=sy,
                    scale_x=sx,
                    phase_y=phase[0],
                    phase_x=phase[1],
                    vae=vae_name,
                    seed=pair_seed,
                )
                Image.fromarray(corrupted).save(outdir / meta.corrupt)
                manifest.write(json.dumps(asdict(meta)) + "\n")
                metas.append(meta)
    return metas
