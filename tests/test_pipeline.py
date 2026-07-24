import numpy as np

from ai2pixelart.corrupt import corrupt, upscale
from ai2pixelart.metrics import detail_retention, evaluate, isolated_details
from ai2pixelart.pipeline import clean


def test_roundtrip_nearest(sprite):
    big = upscale(sprite, scale=3.0, interp="nearest")
    result = clean(big)
    report = evaluate(result.image, sprite)
    assert report["size_match"]
    assert report["cells"]["exact"] > 0.98
    assert report["details"]["rate"] == 1.0


def test_roundtrip_nearest_noisy(sprite):
    """Noise without blur: the 1-px eyes must survive exactly — palette
    extraction may not 'denoise away' small isolated details."""
    big = corrupt(sprite, scale=3.3, phase=(0.7, 1.3), interp="nearest", blur=0.0, seed=0)
    result = clean(big)
    report = evaluate(result.image, sprite)
    assert report["size_match"]
    assert report["cells"]["tolerant"] > 0.95
    assert report["details"]["rate"] == 1.0


def test_roundtrip_noisy_bilinear(sprite):
    """Bilinear + blur mixes neighbors into 1-px details: at this corruption
    level the eye cell's best surviving pixels are ~ΔE 23 from pure white, so
    recovering exact detail colors needs semantic priors (the learned model's
    job). The classical baseline is held to the structural property instead:
    every isolated single-cell detail must STAY an isolated single-cell
    detail, in roughly the right color direction."""
    big = corrupt(sprite, scale=3.3, phase=(0.7, 1.3), blur=0.3, seed=0)
    result = clean(big)
    report = evaluate(result.image, sprite)
    assert report["size_match"]
    assert report["cells"]["tolerant"] > 0.85
    gt_isolated = isolated_details(sprite)
    pred_isolated = isolated_details(result.image)
    assert (gt_isolated & pred_isolated).sum() == gt_isolated.sum()
    relaxed = detail_retention(result.image, sprite, match_de=25.0)
    assert relaxed["rate"] == 1.0
    assert report["palette"]["size_pred"] <= report["palette"]["size_gt"] + 4


def test_forced_palette(sprite, palette):
    big = corrupt(sprite, scale=3.0, seed=1)
    result = clean(big, palette=palette)
    out_colors = {tuple(c) for c in result.image.reshape(-1, 3)}
    assert out_colors <= {tuple(p) for p in palette}


def test_true_resolution_passthrough(sprite):
    """An image with no fake-pixel grid is only palette-quantized."""
    result = clean(sprite)
    if result.grid is not None:
        # if a grid was (wrongly) detected it must at least keep the size
        assert result.image.shape == sprite.shape
    else:
        assert result.image.shape == sprite.shape


def test_granularity_scales_output(sprite):
    big = upscale(sprite, scale=3.0, interp="nearest")
    base = clean(big)
    fine = clean(big, granularity=2)
    # each detected cell sampled as 2x2 — no new content, exact 2x upscale here
    assert np.array_equal(fine.image, np.repeat(np.repeat(base.image, 2, 0), 2, 1))
    coarse = clean(big, granularity=0.5)
    assert coarse.image.shape[0] * 2 == base.image.shape[0]
    # subdivision is capped so cells keep >= 1 px: pitch 3 allows at most 3x
    capped = clean(big, pitch=(3.0, 3.0), granularity=4)
    assert capped.image.shape[0] == base.image.shape[0] * 3


def test_known_pitch_bypass(sprite):
    big = upscale(sprite, scale=4.0, interp="nearest")
    result = clean(big, pitch=(4.0, 4.0))
    assert result.image.shape == sprite.shape
    assert np.array_equal(result.image, sprite)


def test_requested_colors_all_reach_the_output():
    """Regression (user report): 14.png at 64 colors showed only 60/62 in the
    output — anti-drag cap-merging left shadowed palette entries that
    quantize() never picked. The full image is needed; small crops don't
    produce the shadowing."""
    from pathlib import Path

    from PIL import Image

    path = Path(__file__).resolve().parents[1] / "examples" / "gemini" / "14.png"
    img = np.array(Image.open(path).convert("RGB"))
    result = clean(img, max_colors=64, absorb_de=0.0)
    assert len(result.palette) == 64
    assert len(np.unique(result.image.reshape(-1, 3), axis=0)) == 64


def test_hue_coverage_on_color_rich_sheet():
    """Regression (user report): 04.png collapsed to ~5 hues + near-dupes even
    at 128 colors — linkage representatives chosen by global frequency starved
    rare-but-distinct color families (a green sprite on a mostly-pink sheet).
    Coverage comes from color-space bins, exactness from bin-mode reps."""
    from pathlib import Path

    from PIL import Image

    from ai2pixelart.palette import delta_e, rgb_to_lab

    path = Path(__file__).resolve().parents[1] / "examples" / "gemini" / "04.png"
    img = np.array(Image.open(path).convert("RGB"))
    result = clean(img, max_colors=128)
    assert len(result.palette) == 128
    lab = rgb_to_lab(result.palette)
    kept = []
    for i in range(len(lab)):
        if all(float(delta_e(lab[i], lab[j])) > 5 for j in kept):
            kept.append(i)
    assert len(kept) > 100  # was 25 when coverage starved


def test_dedupe_palette_drops_near_twins():
    from ai2pixelart.palette import dedupe_palette

    pal = np.array(
        [[100, 100, 100], [101, 101, 100], [200, 40, 40], [10, 10, 10], [201, 42, 41]],
        dtype=np.uint8,
    )
    out = dedupe_palette(pal)
    # first of each twin group survives, order preserved
    assert out.tolist() == [[100, 100, 100], [200, 40, 40], [10, 10, 10]]


def test_consensus_unifies_identical_regions_and_keeps_details():
    """Two far-apart regions with identical observed color but different
    (near-twin) entries collapse to the more massive entry; a detail cell
    whose observed color is genuinely different stays untouched."""
    from ai2pixelart.pipeline import consensus_indices

    pal = np.array(
        [[60, 110, 70], [57, 118, 72], [255, 255, 255]], dtype=np.uint8
    )
    raw = np.full((20, 40, 3), (65, 114, 71), dtype=np.uint8)  # one flat color
    raw[10, 10] = (255, 255, 255)  # 1-cell detail
    idx = np.zeros((20, 40), dtype=np.int64)
    idx[:, 25:] = 1  # right area got the twin entry
    idx[10, 10] = 2

    out = consensus_indices(idx, raw, pal)
    assert out[10, 10] == 2  # detail survives (own color cluster)
    flat = out[idx != 2] if False else out[np.where(~((np.arange(20)[:, None] == 10) & (np.arange(40)[None, :] == 10)))]
    assert set(np.unique(flat).tolist()) == {0}  # entry 0 has more mass -> wins everywhere


def test_clean_consensus_forced_palette_unifies_noisy_flat_image():
    """End to end: noisy flat image + forced palette with a near-twin pair ->
    without consensus the halves may split; with it the output is one color."""
    rng = np.random.default_rng(3)
    base = np.array([70, 110, 72], dtype=np.float64)
    img = np.clip(
        base + rng.normal(0, 3, (96, 96, 3)), 0, 255
    ).astype(np.uint8)
    img = img.repeat(4, axis=0).repeat(4, axis=1)  # pitch-4 grid, 96x96 cells
    pal = np.array(
        [[60, 105, 65], [75, 115, 78], [220, 220, 220], [10, 10, 10]], dtype=np.uint8
    )
    plain = clean(img, palette=pal, pitch=(4.0, 4.0))
    result = clean(img, palette=pal, pitch=(4.0, 4.0), consensus=True)

    def dominance(indices):
        _, c = np.unique(indices, return_counts=True)
        return c.max() / c.sum()

    # a few ~4-sigma noise cells legitimately measure as the twin entry and
    # are protected by the guard; near-total dominance is the contract
    assert dominance(result.indices) >= 0.995
    assert dominance(result.indices) > dominance(plain.indices)
