"""End-to-end demo on a synthetic sprite with known ground truth.

The sprite deliberately contains 1-px isolated details (white eye pixels) so
the detail-retention metric has something meaningful to measure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ai2pixelart.corrupt import corrupt
from ai2pixelart.metrics import evaluate
from ai2pixelart.pipeline import clean

BG = (30, 34, 52)
BODY = (94, 167, 64)
SHADE = (60, 120, 45)
OUTLINE = (20, 24, 30)
WHITE = (240, 240, 240)
MOUTH = (170, 60, 60)


def make_sprite() -> np.ndarray:
    """A 24x24 blob character with a 6-color palette and 1-px eye details."""
    img = np.zeros((24, 24, 3), dtype=np.uint8)
    img[:, :] = BG

    yy, xx = np.mgrid[0:24, 0:24]
    body = ((yy - 13.0) / 8.0) ** 2 + ((xx - 11.5) / 7.5) ** 2 <= 1.0
    img[body] = BODY

    # simple bottom-right shading inside the body
    shade = body & (((yy - 10.0) / 10.0) ** 2 + ((xx - 9.5) / 9.0) ** 2 > 1.0)
    img[shade] = SHADE

    # 1-px outline around the body
    interior = body.copy()
    interior[1:-1, 1:-1] = (
        body[1:-1, 1:-1] & body[:-2, 1:-1] & body[2:, 1:-1] & body[1:-1, :-2] & body[1:-1, 2:]
    )
    img[body & ~interior] = OUTLINE

    img[10, 9] = WHITE  # left eye  (isolated 1-px detail)
    img[10, 14] = WHITE  # right eye (isolated 1-px detail)
    img[13, 11:14] = MOUTH
    return img


def run_demo(outdir: str | Path, scale: float = 3.3, seed: int = 0) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gt = make_sprite()
    # blur 0.3 is the working point of the classical baseline; see README
    # "Known baseline limitations" for what breaks beyond it (and why that
    # is the learned model's job)
    corrupted = corrupt(gt, scale=scale, phase=(0.7, 1.3), blur=0.3, seed=seed)
    result = clean(corrupted)
    report = evaluate(result.image, gt)
    report["grid"] = (
        None
        if result.grid is None
        else {"pitch_y": result.grid.y.pitch, "pitch_x": result.grid.x.pitch, "true_pitch": scale}
    )

    _save(gt, outdir / "ground_truth.png", preview_scale=8)
    _save(corrupted, outdir / "corrupted.png", preview_scale=2)
    _save(result.image, outdir / "cleaned.png", preview_scale=8)
    (outdir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def _save(img: np.ndarray, path: Path, preview_scale: int = 1) -> None:
    Image.fromarray(img).save(path)
    if preview_scale > 1:
        big = Image.fromarray(img).resize(
            (img.shape[1] * preview_scale, img.shape[0] * preview_scale), Image.NEAREST
        )
        big.save(path.with_name(path.stem + "_preview.png"))
