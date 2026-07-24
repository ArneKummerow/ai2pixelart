import numpy as np

from ai2pixelart.autoqa import assess, detail_survival, speckle_rate
from ai2pixelart.corrupt import corrupt, upscale
from ai2pixelart.pipeline import clean


def test_qa_sanity_on_clean_sprite(sprite):
    big = upscale(sprite, scale=3.0, interp="nearest")
    result = clean(big)
    report = assess(big, result)
    assert report["boundary_snr"] > 2.0  # grid sits on real edges
    assert report["cell_fit_mean"] < 1.0  # palette represents the cells
    assert report["speckle_rate"] < 0.01
    assert report["detail_survival"] > 0.9
    assert report["n_details"] > 0  # the sprite's eyes are counted


def test_smoothing_removes_speckle_and_keeps_details(sprite):
    big = corrupt(sprite, scale=3.3, phase=(0.7, 1.3), blur=0.3, seed=0)
    noisy = clean(big, smooth=False)
    smoothed = clean(big)
    # the metric fires on the unsmoothed result and the default pipeline
    # removes (almost) everything it fires on
    assert speckle_rate(noisy) > 0.005
    assert speckle_rate(smoothed) < speckle_rate(noisy) / 2
    # the detail guard: smoothing must not eat outlier cells beyond what
    # quantization already costs (measured: 0.60 both ways here — the loss
    # is quantization's, at a corruption level where exact detail colors
    # are documented as classically unrecoverable)
    s_noisy, n = detail_survival(noisy)
    s_smooth, _ = detail_survival(smoothed)
    assert n > 0
    assert s_smooth >= s_noisy - 1e-9
