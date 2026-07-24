from pathlib import Path

import numpy as np

from ai2pixelart.palette import delta_e, image_palette, rgb_to_lab
from ai2pixelart.spritegen import (
    build_palette_pool,
    load_palette_pool,
    random_palette,
    random_sprite,
)


def test_palette_pool_roundtrip_and_sprites(tmp_path):
    """Real-image palettes flow into sprites: ~half the draws use a pool
    palette verbatim (so training sees inference-time color statistics)."""
    img_path = Path(__file__).resolve().parents[1] / "examples" / "gemini" / "01.png"
    pool_json = tmp_path / "pool.json"
    assert build_palette_pool([img_path], pool_json) == 1
    pool = load_palette_pool(pool_json)
    assert len(pool) == 1 and len(pool[0]) >= 5

    pool_colors = {tuple(c) for c in pool[0]}
    rng = np.random.default_rng(3)
    used_pool = 0
    for _ in range(12):
        sprite = random_sprite(rng, size=40, pool=pool)
        pal = image_palette(sprite)
        assert len(pal) <= 16
        if {tuple(c) for c in pal} <= pool_colors:
            used_pool += 1
    assert used_pool >= 3


def test_procedural_palettes_contain_near_hue_pairs():
    """Some palettes must contain area-color pairs <12 ΔE apart — the
    skin-vs-hair discrimination real images demand (15.png hallucination)."""
    rng = np.random.default_rng(0)
    found = 0
    for _ in range(40):
        pal = random_palette(rng)
        lab = rgb_to_lab(pal)
        d = delta_e(lab[:, None, :], lab[None, :, :])
        d[np.diag_indices(len(pal))] = np.inf
        if d.min() < 12.0:
            found += 1
    assert found >= 5
