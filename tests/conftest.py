import numpy as np
import pytest

PALETTE = np.array(
    [
        [30, 34, 52],
        [94, 167, 64],
        [60, 120, 45],
        [20, 24, 30],
        [240, 240, 240],
        [170, 60, 60],
    ],
    dtype=np.uint8,
)


@pytest.fixture
def palette():
    return PALETTE


@pytest.fixture
def random_art():
    """32x32 true-resolution art with random cells from a 6-color palette."""
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(PALETTE), size=(32, 32))
    return PALETTE[idx]


@pytest.fixture
def sprite():
    from ai2pixelart.demo import make_sprite

    return make_sprite()
