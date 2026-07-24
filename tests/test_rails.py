import numpy as np

from ai2pixelart.corrupt import upscale
from ai2pixelart.rails import cell_colors, validity_mask


def _meta(scale: float, phase=(0.0, 0.0)) -> dict:
    return {"scale_y": scale, "scale_x": scale, "phase_y": phase[0], "phase_x": phase[1]}


def test_identity_corruption_is_fully_valid(sprite):
    big = upscale(sprite, scale=3.0, interp="nearest")
    assert validity_mask(sprite, big, _meta(3.0)).all()
    assert np.array_equal(cell_colors(big, _meta(3.0), *sprite.shape[:2]), sprite)


def test_noise_within_tolerance_stays_valid(sprite):
    big = upscale(sprite, scale=3.0, interp="nearest")
    noisy = np.clip(big.astype(np.int16) + 5, 0, 255).astype(np.uint8)
    assert validity_mask(sprite, noisy, _meta(3.0)).all()


def test_moved_content_is_masked_out(sprite):
    """Cells the 'editor' repainted must be poison; untouched cells valid."""
    big = upscale(sprite, scale=3.0, interp="nearest")
    vandal = big.copy()
    vandal[9:30, 9:30] = [255, 0, 255]
    m = validity_mask(sprite, vandal, _meta(3.0))
    assert not m[4:9, 4:9].any()  # cells fully inside the repainted block
    assert m[:3, :].all() and m[:, :3].all() and m[11:, 11:].all()
