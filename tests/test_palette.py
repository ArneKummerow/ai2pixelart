import numpy as np

from ai2pixelart.palette import (
    delta_e,
    extract_palette,
    image_palette,
    parse_hex_palette,
    quantize,
    rgb_to_lab,
)


def test_extract_recovers_palette(palette, random_art):
    rng = np.random.default_rng(0)
    noisy = np.clip(
        random_art.astype(np.int16) + rng.integers(-3, 4, random_art.shape), 0, 255
    ).astype(np.uint8)
    extracted = extract_palette(noisy, merge_de=6.0)
    assert len(extracted) == len(palette)
    # every true color has a close match in the extracted palette
    de = delta_e(
        rgb_to_lab(palette).reshape(-1, 1, 3), rgb_to_lab(extracted).reshape(1, -1, 3)
    )
    assert de.min(axis=1).max() < 3.0


def test_extract_single_color():
    img = np.full((8, 8, 3), (10, 200, 30), dtype=np.uint8)
    assert len(extract_palette(img)) == 1


def test_max_colors_cap(random_art):
    extracted = extract_palette(random_art, merge_de=1.0, max_colors=4)
    assert len(extracted) <= 4


def test_quantize_is_idempotent_on_palette_image(palette, random_art):
    quantized, idx = quantize(random_art, palette)
    assert np.array_equal(quantized, random_art)
    assert np.array_equal(palette[idx], random_art)


def test_quantize_forces_palette(palette):
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(16, 16, 3)).astype(np.uint8)
    quantized, _ = quantize(img, palette)
    assert len(image_palette(quantized)) <= len(palette)
    assert all(tuple(c) in {tuple(p) for p in palette} for c in image_palette(quantized))


def test_parse_hex_palette():
    pal = parse_hex_palette("#1e2234, #5ea740,#f0f0f0")
    assert pal.shape == (3, 3)
    assert tuple(pal[0]) == (0x1E, 0x22, 0x34)


def test_max_colors_is_target_and_cap():
    """An explicit color count must also STOP merging, not only cap it:
    asking for more colors than the merge_de cutoff yields splits clusters
    back up (user report: >natural had no effect, <natural worked)."""
    rng = np.random.default_rng(0)
    base = np.array([[10, 10, 10], [230, 20, 20], [20, 230, 20], [20, 20, 230]], np.int16)
    variants = np.clip(np.repeat(base, 10, axis=0) + rng.integers(-1, 2, (40, 3)), 0, 255)
    colors = np.repeat(variants, 25, axis=0).astype(np.uint8)  # (1000, 3)

    natural = len(extract_palette(colors))
    assert natural == 4  # jitter merges away at the default merge_de

    assert len(extract_palette(colors, max_colors=10)) == 10  # target above natural
    assert len(extract_palette(colors, max_colors=2)) == 2  # cap below natural
    # more colors than distinct inputs exist: bounded by the input
    n_uniq = len(np.unique(colors.reshape(-1, 3), axis=0))
    assert len(extract_palette(colors, max_colors=100)) == n_uniq


def test_reseed_replaces_shadowed_entries():
    """A palette entry no color maps to must be reseeded onto the worst-
    represented observed color, so quantize() uses every entry."""
    from ai2pixelart.palette import _reseed_unused

    colors = np.array([[0, 0, 0]] * 10 + [[255, 255, 255]] * 10 + [[255, 0, 0]] * 3, np.uint8)
    pal = np.array([[0, 0, 0], [2, 2, 2], [255, 255, 255]], np.uint8)  # (2,2,2) is shadowed
    out = _reseed_unused(pal.copy(), colors)
    assert len(out) == 3
    assert [255, 0, 0] in out.tolist()  # reseeded to the unrepresented color
    _, idx = quantize(colors, out)
    assert len(np.unique(idx)) == 3


def test_reseed_drops_unwinnable_entries():
    """Fewer distinct colors than palette entries: the dead entry is dropped
    rather than promising a count the output cannot show."""
    from ai2pixelart.palette import _reseed_unused

    colors = np.array([[0, 0, 0]] * 5 + [[255, 255, 255]] * 5, np.uint8)
    pal = np.array([[0, 0, 0], [2, 2, 2], [255, 255, 255]], np.uint8)
    out = _reseed_unused(pal.copy(), colors)
    assert sorted(out.tolist()) == [[0, 0, 0], [255, 255, 255]]


def test_dominant_color_survives_many_unique_colors():
    """>MAX_UNIQUE distinct colors force linkage onto representatives; the
    dominant flat-area color must come through (near-)exactly. Grid-bin
    centers used to tint it by several ΔE (user report: 'more reddish',
    visibly wrong large areas)."""
    rng = np.random.default_rng(1)
    dom = np.array([137, 72, 155], np.uint8)
    noise = rng.integers(0, 256, (3000, 3)).astype(np.uint8)
    colors = np.concatenate([np.repeat(dom[None], 20000, axis=0), noise])
    pal = extract_palette(colors)
    d = float(delta_e(rgb_to_lab(dom.reshape(1, 3)), rgb_to_lab(pal)).min())
    assert d < 0.75
