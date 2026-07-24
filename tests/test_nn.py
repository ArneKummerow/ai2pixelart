import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image

from ai2pixelart.metrics import isolated_details
from ai2pixelart.nndata import (
    K_MAX,
    PairDataset,
    majority_vote_cells,
    source_cell_maps,
)
from ai2pixelart.nnmodel import PixelCleanNet, pad_to_multiple
from ai2pixelart.pairgen import generate_pairs
from ai2pixelart.palette import image_palette
from ai2pixelart.spritegen import random_sprite

from test_vae import MockVAE  # identity autoencoder


def test_sprite_properties():
    rng = np.random.default_rng(0)
    for _ in range(5):
        sprite = random_sprite(rng)
        pal = image_palette(sprite)
        assert 4 <= len(pal) <= K_MAX
        assert isolated_details(sprite).sum() >= 1  # guaranteed 1-px detail


def test_model_forward_shapes_and_mask():
    model = PixelCleanNet(base=16, feat_dim=32)
    img = torch.randn(2, 3, 96, 96)
    palette = torch.randn(2, K_MAX, 3)
    mask = torch.zeros(2, K_MAX, dtype=torch.bool)
    mask[:, :5] = True
    logits = model(img, palette, mask)
    assert logits.shape == (2, K_MAX, 96, 96)
    assert (logits[:, 5:] < -1e3).all()  # padded palette slots masked out
    assert logits.argmax(1).max() < 5


def test_pad_to_multiple():
    img = torch.randn(1, 3, 23, 37)
    padded, (h, w) = pad_to_multiple(img)
    assert (h, w) == (23, 37)
    assert padded.shape[-2] % 8 == 0 and padded.shape[-1] % 8 == 0


@pytest.fixture
def pairs_dir(tmp_path):
    rng = np.random.default_rng(1)
    src = tmp_path / "sprites"
    src.mkdir()
    for i in range(3):
        Image.fromarray(random_sprite(rng, size=40)).save(src / f"s{i}.png")
    outdir = tmp_path / "pairs"
    generate_pairs(
        sorted(src.iterdir()), outdir, MockVAE(), vae_name="mock",
        n_per_image=2, scale_range=(3.0, 5.0), seed=5,
    )
    return outdir


def test_target_alignment_exact(pairs_dir):
    """With the identity VAE + nearest upscale, painting the target indices
    with the palette must reproduce the corrupted image EXACTLY — this pins
    the entire supervision path (scale/phase metadata -> cell maps)."""
    ds = PairDataset(pairs_dir, split="train", val_frac=0.2)
    assert len(ds) >= 4
    for meta in ds.items:
        corrupt, target, palette = ds.load_pair(meta)
        assert np.array_equal(palette[target], corrupt)


def test_majority_vote_roundtrip(pairs_dir):
    ds = PairDataset(pairs_dir, split="val", val_frac=0.2)
    meta = ds.items[0]
    corrupt, target, palette = ds.load_pair(meta)
    clean = np.array(Image.open(pairs_dir / meta["clean"]).convert("RGB"))
    rows, cols = source_cell_maps(
        meta, corrupt.shape[0], corrupt.shape[1], clean.shape[0], clean.shape[1]
    )
    cell_idx = majority_vote_cells(
        target, rows, cols, clean.shape[0], clean.shape[1], len(palette)
    )
    assert np.array_equal(palette[cell_idx], clean)


def test_dataset_batch_shapes(pairs_dir):
    ds = PairDataset(pairs_dir, crop=96, split="train", val_frac=0.2)
    item = ds[0]
    assert item["img"].shape == (3, 96, 96)
    assert item["target"].shape == (96, 96)
    assert item["palette"].shape == (K_MAX, 3)
    assert item["pal_mask"].sum() >= 4
    assert item["img"].min() >= -1.0 and item["img"].max() <= 1.0


def test_one_training_step(pairs_dir):
    """Loss decreases over a few steps of overfitting a single batch (CPU)."""
    ds = PairDataset(pairs_dir, crop=96, split="train", val_frac=0.2)
    batch = ds[0]
    img = batch["img"][None]
    target = batch["target"][None]
    palette = batch["palette"][None]
    mask = batch["pal_mask"][None]

    model = PixelCleanNet(base=8, feat_dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(8):
        loss = torch.nn.functional.cross_entropy(model(img, palette, mask), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]

def test_nn_clean_image_returns_clean_result(sprite):
    from ai2pixelart.corrupt import upscale
    from ai2pixelart.nninfer import nn_clean_image

    model = PixelCleanNet(base=8, feat_dim=16)  # untrained; shapes only
    big = upscale(sprite, 3.0, interp="nearest")
    result = nn_clean_image(big, model, device="cpu")
    assert result.image.dtype == np.uint8
    assert len(result.palette) <= K_MAX
    if result.grid is not None:
        assert result.image.shape[:2] == result.grid.shape
    # output uses only palette colors (structural guarantee)
    out_colors = {tuple(c) for c in result.image.reshape(-1, 3)}
    assert out_colors <= {tuple(c) for c in result.palette}


def test_majority_vote_fills_empty_cells():
    """Sub-pixel cells that contain no pixel center must inherit the nearest
    voted cell's index, not silently become palette entry 0 (grid-shaped
    dark lines at granularity 2 on ~1px grids)."""
    from ai2pixelart.nndata import majority_vote_cells

    pred = np.full((4, 4), 3, dtype=np.int64)
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 0, 0, 0])  # every pixel votes into cell column 0
    out = majority_vote_cells(pred, rows, cols, 2, 2, k=5)
    assert (out == 3).all()  # empty column 1 fills from its voted neighbor


def test_validity_mask_zeroes_loss_weights(pairs_dir):
    """A pair carrying a rails mask must contribute zero loss on masked cells."""
    from ai2pixelart.nndata import PairDataset, source_cell_maps

    ds = PairDataset(pairs_dir, split="train", seed=0)
    meta = dict(ds.items[0])
    clean = np.array(Image.open(ds.dir / meta["clean"]).convert("RGB"))
    mask = np.full(clean.shape[:2], 255, np.uint8)
    mask[0, 0] = 0
    Image.fromarray(mask).save(ds.dir / "m0.png")
    meta["mask"] = "m0.png"
    corrupt, _, _, weight = ds._load_full(meta)
    rows, cols = source_cell_maps(meta, corrupt.shape[0], corrupt.shape[1], *clean.shape[:2])
    poisoned = (rows[:, None] == 0) & (cols[None, :] == 0)
    assert (weight[poisoned] == 0).all()
    assert (weight[~poisoned] > 0).all()


def test_edit_pairs_and_multi_dir_dataset(tmp_path):
    """Rail-guarded img2img pairs carry masks; PairDataset mixes corpora."""
    from ai2pixelart.pairgen import generate_edit_pairs, generate_pairs
    from ai2pixelart.spritegen import random_sprite

    rng = np.random.default_rng(0)
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        Image.fromarray(random_sprite(rng, size=40)).save(src / f"s{i}.png")
    paths = sorted(src.glob("*.png"))

    vae_dir = tmp_path / "vae"
    generate_pairs(paths, vae_dir, MockVAE(), "mock", n_per_image=1, scale_range=(3.0, 4.0))

    edit_dir = tmp_path / "edit"
    identity = lambda img, pipe, strength, seed: img.copy()
    metas = generate_edit_pairs(
        paths, edit_dir, pipe=None, editor_name="identity", n_per_image=1,
        scale_range=(3.0, 4.0), corrupt_fn=identity,
    )
    assert len(metas) == 3
    assert all(m["valid_frac"] == 1.0 and m["mask"] for m in metas)

    # a vandalizing editor gets its pairs rejected by the rails
    vandal = lambda img, pipe, strength, seed: np.full_like(img, 128)
    rej = generate_edit_pairs(paths, tmp_path / "rej", pipe=None, editor_name="vandal",
                              n_per_image=1, corrupt_fn=vandal)
    assert len(rej) == 0

    ds = PairDataset([vae_dir, edit_dir], crop=64, split="train", val_frac=0.2, seed=0)
    roots = {m["_root"].split("/")[-1] for m in ds.items}
    assert roots == {"vae", "edit"}
    item = ds[0]
    assert item["img"].shape[0] == 3 and item["weight"].shape == item["target"].shape


def test_downscaled_pairs_keep_exact_alignment(tmp_path):
    """Corrupt-then-halve pairs: the halved metadata (scale/2, phase/2) must
    reproduce source cells exactly under a 2x box filter — this recipe is
    how the fine-pitch regime (inaccessible to f8 latents) gets trained."""
    from ai2pixelart.nndata import PairDataset
    from ai2pixelart.pairgen import generate_edit_pairs
    from ai2pixelart.rails import cell_colors
    from ai2pixelart.spritegen import random_sprite

    rng = np.random.default_rng(1)
    src = tmp_path / "src"
    src.mkdir()
    for i in range(4):
        Image.fromarray(random_sprite(rng, size=44)).save(src / f"s{i}.png")

    identity = lambda img, pipe, strength, seed: img.copy()
    metas = generate_edit_pairs(
        sorted(src.glob("*.png")), tmp_path / "out", pipe=None, editor_name="identity",
        n_per_image=1, downscale_frac=1.0, corrupt_fn=identity, seed=3,
    )
    assert len(metas) == 4
    assert all(m["downscaled"] and 2.0 <= m["scale_y"] <= 3.3 for m in metas)
    # identity corruption + box halving: cells reproduce the clean sprite.
    # ~1% of cells (thin outlines at high-contrast boundaries) are genuinely
    # ambiguous at pitch ~2.4 — box-averaged pixels there are mostly
    # mixtures — and the rails CORRECTLY mask them rather than supervise on
    # colors the image does not contain.
    assert all(m["valid_frac"] >= 0.97 for m in metas)
    for m in metas:
        clean = np.array(Image.open(tmp_path / "out" / m["clean"]).convert("RGB"))
        corrupt = np.array(Image.open(tmp_path / "out" / m["corrupt"]).convert("RGB"))
        cc = cell_colors(corrupt, m, *clean.shape[:2])
        from ai2pixelart.palette import delta_e, rgb_to_lab
        d = delta_e(rgb_to_lab(cc), rgb_to_lab(clean))
        assert float(np.median(d)) < 2.0  # interior cells essentially exact
    # and the dataset trains from them (per-pixel targets resolve)
    ds = PairDataset(tmp_path / "out", crop=64, split="train", val_frac=0.25, seed=0)
    assert len(ds) >= 1
    item = ds[0]
    assert item["target"].shape == (64, 64)


def test_shaded_pairs_keep_flat_targets(tmp_path):
    """Shading augmentation: the corruption carries a gradient but the target
    stays the flat sprite, and rails still accept the pair (the gradient
    amplitude sits below keep_de) — teaches vignette collapse."""
    from ai2pixelart.pairgen import generate_edit_pairs
    from ai2pixelart.spritegen import random_sprite

    rng = np.random.default_rng(2)
    src = tmp_path / "src"
    src.mkdir()
    for i in range(4):
        Image.fromarray(random_sprite(rng, size=40)).save(src / f"s{i}.png")
    identity = lambda img, pipe, strength, seed: img.copy()
    metas = generate_edit_pairs(
        sorted(src.glob("*.png")), tmp_path / "out", pipe=None, editor_name="identity",
        n_per_image=1, shade_prob=1.0, corrupt_fn=identity, seed=5,
    )
    assert len(metas) == 4 and all(m["shaded"] for m in metas)
    assert all(m["valid_frac"] >= 0.9 for m in metas)
    m = metas[0]
    corrupt = np.array(Image.open(tmp_path / "out" / m["corrupt"]).convert("RGB")).astype(float)
    # the corruption actually carries a low-frequency field: corner means differ
    q = corrupt.shape[0] // 3, corrupt.shape[1] // 3
    corners = [corrupt[: q[0], : q[1]].mean(), corrupt[-q[0]:, -q[1]:].mean()]
    assert abs(corners[0] - corners[1]) > 1.0


def test_corruptor_domain_harvest(tmp_path):
    """Harvest picks only rail-accepted img2img outputs + oversampled real."""
    from ai2pixelart.corruptor import DomainCrops, harvest_domain_images
    from ai2pixelart.pairgen import generate_edit_pairs, generate_pairs
    from ai2pixelart.spritegen import random_sprite

    rng = np.random.default_rng(4)
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        Image.fromarray(random_sprite(rng, size=40)).save(src / f"s{i}.png")
    identity = lambda img, pipe, strength, seed: img.copy()
    generate_edit_pairs(sorted(src.glob("*.png")), tmp_path / "edit", pipe=None,
                        editor_name="id", n_per_image=1, corrupt_fn=identity)
    generate_pairs(sorted(src.glob("*.png")), tmp_path / "vae", MockVAE(), "mock",
                   n_per_image=1, scale_range=(3.0, 4.0))
    real = tmp_path / "real"
    real.mkdir()
    Image.fromarray(random_sprite(rng, size=40)).save(real / "r.png")

    paths = harvest_domain_images([tmp_path / "edit", tmp_path / "vae"], [real], real_oversample=5)
    assert len(paths) == 3 + 5  # vae corpus (no img2img/mask) excluded; real x5
    ds = DomainCrops(paths, size=96, seed=0)
    crop = ds[0]
    assert tuple(crop.shape) == (3, 96, 96)
    assert -1.0 <= float(crop.min()) and float(crop.max()) <= 1.0


def test_leash_mask_bounds_choices():
    """The leash allows only near entries plus always the nearest one."""
    from ai2pixelart.nninfer import leash_mask

    img = np.zeros((2, 2, 3), np.uint8)
    img[0, 0] = [200, 60, 20]  # orange pixel
    palette = np.array([[205, 65, 25], [40, 160, 60], [10, 10, 10]], np.uint8)  # near-orange, green, black
    m = leash_mask(img, palette, leash=20.0)
    assert m[0, 0, 0] and not m[0, 0, 1] and not m[0, 0, 2]  # orange: only near-orange
    # black pixels: black entry allowed; green/orange excluded
    assert m[1, 1, 2] and not m[1, 1, 1]
    # a pixel with NO entry within leash still gets its nearest
    img2 = np.full((1, 1, 3), 128, np.uint8)
    far = np.array([[255, 255, 255], [0, 0, 0]], np.uint8)
    m2 = leash_mask(img2, far, leash=5.0)
    assert m2[0, 0].sum() == 1


def test_decoy_palette_training_samples(pairs_dir):
    """Training samples pad palettes with decoy entries (dense-palette
    discrimination); decoys never collide with real colors and targets
    still index only real entries."""
    from ai2pixelart.nndata import K_MAX, PairDataset, make_decoys
    from ai2pixelart.palette import delta_e, rgb_to_lab

    ds = PairDataset(pairs_dir, split="train", seed=0, decoy_max=48)
    item = ds[0]
    assert item["palette"].shape == (K_MAX, 3)
    corrupt, target, real_palette = ds.load_pair(ds.items[0])
    assert int(item["target"].max()) < len(real_palette) + 48

    rng = np.random.default_rng(0)
    pal = np.array([[10, 10, 10], [200, 50, 40], [240, 240, 240]], np.uint8)
    dec = make_decoys(pal, 20, rng)
    assert len(dec) > 5
    d = delta_e(rgb_to_lab(dec)[:, None, :], rgb_to_lab(pal)[None, :, :])
    assert float(d.min()) >= 5.0


def test_sheet_sprites():
    """2x2 tile sheets: bigger canvas, shared palette <= 16 colors, gutters."""
    from ai2pixelart.palette import image_palette
    from ai2pixelart.spritegen import sheet_sprite

    rng = np.random.default_rng(6)
    for _ in range(4):
        s = sheet_sprite(rng)
        assert s.shape[0] >= 59  # 2 tiles + gutters
        assert len(image_palette(s)) <= 16


def test_noise_augmented_pairs(tmp_path):
    from ai2pixelart.pairgen import generate_edit_pairs
    from ai2pixelart.spritegen import random_sprite

    rng = np.random.default_rng(7)
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        Image.fromarray(random_sprite(rng, size=40)).save(src / f"s{i}.png")
    identity = lambda img, pipe, strength, seed: img.copy()
    metas = generate_edit_pairs(sorted(src.glob("*.png")), tmp_path / "out", pipe=None,
                                editor_name="id", n_per_image=1, noise_prob=1.0,
                                corrupt_fn=identity, seed=8)
    assert len(metas) == 3
    assert all(m["noise_sigma"] >= 3.0 for m in metas)
    m = metas[0]
    clean = np.array(Image.open(tmp_path / "out" / m["clean"]).convert("RGB"))
    corrupt = np.array(Image.open(tmp_path / "out" / m["corrupt"]).convert("RGB")).astype(float)
    # the noise is really in the corruption
    assert corrupt.std() > 0 and float(np.abs(np.diff(corrupt[0, :, 0])).mean()) > 1.0
