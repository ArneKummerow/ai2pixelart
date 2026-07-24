import numpy as np
import pytest

from ai2pixelart.corrupt import corrupt, upscale
from ai2pixelart.grid import estimate_grid


@pytest.mark.parametrize("pitch", [2.0, 3.0, 3.5, 4.2, 6.0])
def test_pitch_recovery_nearest(random_art, pitch):
    big = upscale(random_art, scale=pitch, interp="nearest")
    grid = estimate_grid(big)
    assert grid is not None
    assert abs(grid.y.pitch - pitch) / pitch < 0.02
    assert abs(grid.x.pitch - pitch) / pitch < 0.02


def test_pitch_recovery_noisy_bilinear(random_art):
    big = corrupt(random_art, scale=3.3, phase=(0.7, 1.3), seed=1)
    grid = estimate_grid(big)
    assert grid is not None
    assert abs(grid.y.pitch - 3.3) / 3.3 < 0.03
    assert abs(grid.x.pitch - 3.3) / 3.3 < 0.03


def test_anisotropic_pitch(random_art):
    big = upscale(random_art, scale=(3.0, 4.0), interp="nearest")
    grid = estimate_grid(big)
    assert grid is not None
    assert abs(grid.y.pitch - 3.0) < 0.1
    assert abs(grid.x.pitch - 4.0) < 0.15


def test_cell_count_matches_source(random_art):
    big = upscale(random_art, scale=3.5, interp="nearest")
    grid = estimate_grid(big)
    assert grid is not None
    assert grid.shape == random_art.shape[:2]


def test_no_grid_on_flat_image():
    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    assert estimate_grid(flat) is None


def test_regrain_edge_cases():
    from ai2pixelart.grid import Grid, regrain, uniform_axis_grid

    g = Grid(y=uniform_axis_grid(21, 3.0), x=uniform_axis_grid(21, 3.0))
    fine = regrain(g, 3)
    assert fine.shape == (21, 21)
    assert np.allclose(np.diff(fine.y.edges), 1.0)
    coarse = regrain(g, 1 / 2)  # 7 cells -> 4, trailing odd cell kept
    assert coarse.shape == (4, 4)
    assert coarse.y.edges[-1] == 21.0
    assert regrain(g, 1.2) is g  # rounds to factor 1: no-op
    assert regrain(g, 4).shape == (21, 21)  # capped at 1 px cells


def test_blurry_small_pitch_keeps_fine_grid(sprite):
    """Blur once let a spurious coarse scale (4.32) beat the higher-scoring
    true 3.0 on one axis via the largest-near-best rule — output grid too
    coarse, detail eaten. The joint square-pair selection lets the confident
    axis veto it."""
    for blur in (0.6, 0.8):
        g = estimate_grid(corrupt(sprite, scale=3.0, blur=blur, seed=0))
        assert g is not None
        assert abs(g.y.pitch - 3.0) < 0.1
        assert abs(g.x.pitch - 3.0) < 0.1


def test_decorative_periodicity_does_not_own_the_grid():
    """04.png: horizontal scanlines create a strong y-comb at 3.58 that used
    to drag the chosen pair to 3.58x3.59 — halving output resolution and
    mushing every sprite. The champion veto (art grid at x 2.0 out-scores
    the decorative scale) restores the fine square pair."""
    from pathlib import Path

    from PIL import Image

    path = Path(__file__).resolve().parents[1] / "examples" / "gemini" / "04.png"
    g = estimate_grid(np.array(Image.open(path).convert("RGB")))
    assert g is not None
    assert g.y.pitch < 2.6
    assert g.x.pitch < 2.6
