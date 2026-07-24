"""Smoke-test corruptions.

NOT the training degradation. These simple resampling corruptions exist to
exercise the classical pipeline and metrics with known ground truth. The
real data engine (VAE roundtrips, rail-guarded img2img) lives in a later
milestone and produces the semantically interesting artifacts these cannot.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def upscale(
    img: np.ndarray,
    scale: float | tuple[float, float] = 3.3,
    phase: tuple[float, float] = (0.0, 0.0),
    interp: str = "bilinear",
) -> np.ndarray:
    """Resample true-resolution art to a (possibly non-integer) fake-pixel scale.

    phase: sub-cell offset in output pixels (>= 0), produces misaligned grids.
    interp: 'nearest' (crisp cells) or 'bilinear' (mixed boundary colors).

    The output is cropped to the region fully covered by source cells:
    with a non-zero phase the last cell row/column would otherwise be
    clamp-replicated past its true extent, which adds a phantom partial
    cell to the image and breaks (corrupted -> ground truth) alignment.
    """
    sy, sx = (scale, scale) if isinstance(scale, (int, float)) else scale
    h, w = img.shape[:2]
    out_h = int(np.floor(h * sy - 0.5 - phase[0] - 1e-9)) + 1
    out_w = int(np.floor(w * sx - 0.5 - phase[1] - 1e-9)) + 1
    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float64)
    src_y = (yy + 0.5 + phase[0]) / sy - 0.5
    src_x = (xx + 0.5 + phase[1]) / sx - 0.5
    order = {"nearest": 0, "bilinear": 1}[interp]
    channels = [
        ndimage.map_coordinates(
            img[..., c].astype(np.float64), [src_y, src_x], order=order, mode="nearest"
        )
        for c in range(3)
    ]
    return np.clip(np.round(np.stack(channels, axis=-1)), 0, 255).astype(np.uint8)


def corrupt(
    img: np.ndarray,
    scale: float | tuple[float, float] = 3.3,
    phase: tuple[float, float] = (0.0, 0.0),
    interp: str = "bilinear",
    blur: float = 0.5,
    noise: float = 3.0,
    seed: int | None = None,
) -> np.ndarray:
    """Upscale + mild blur + gaussian noise: a cheap stand-in corruption."""
    rng = np.random.default_rng(seed)
    big = upscale(img, scale=scale, phase=phase, interp=interp).astype(np.float64)
    if blur > 0:
        big = ndimage.gaussian_filter(big, sigma=(blur, blur, 0))
    if noise > 0:
        big = big + rng.normal(0.0, noise, size=big.shape)
    return np.clip(np.round(big), 0, 255).astype(np.uint8)
