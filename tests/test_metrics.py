import pytest

from ai2pixelart.metrics import (
    cell_accuracy,
    detail_retention,
    evaluate,
    isolated_details,
    palette_fidelity,
)


def test_cell_accuracy_perfect(sprite):
    acc = cell_accuracy(sprite, sprite)
    assert acc["exact"] == 1.0
    assert acc["tolerant"] == 1.0


def test_cell_accuracy_detects_error(sprite):
    pred = sprite.copy()
    pred[0, 0] = (255, 0, 0)
    acc = cell_accuracy(pred, sprite)
    assert acc["exact"] < 1.0


def test_isolated_details_finds_eyes(sprite):
    mask = isolated_details(sprite)
    assert mask[10, 9] and mask[10, 14]  # the two 1-px white eyes
    assert not mask[0, 0]  # background is not isolated


def test_detail_retention_drops_when_eye_lost(sprite):
    pred = sprite.copy()
    pred[10, 9] = pred[10, 8]  # paint left eye over with body color
    ret = detail_retention(pred, sprite)
    assert ret["n_details"] >= 2
    assert ret["retained"] == ret["n_details"] - 1


def test_palette_fidelity_identity(sprite):
    fid = palette_fidelity(sprite, sprite)
    assert fid["matched_mean_de"] == 0.0
    assert fid["unmatched"] == 0


def test_evaluate_handles_size_mismatch(sprite):
    report = evaluate(sprite[:-1], sprite)
    assert report["size_match"] is False
    assert "cells" not in report
    assert "palette" in report


def test_shape_mismatch_raises(sprite):
    with pytest.raises(ValueError):
        cell_accuracy(sprite[:-1], sprite)
