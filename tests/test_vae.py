import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ai2pixelart.pairgen import generate_pairs
from ai2pixelart.vae import vae_roundtrip


class MockVAE(torch.nn.Module):
    """Identity autoencoder with the diffusers AutoencoderKL surface."""

    class _Dist:
        def __init__(self, x):
            self.x = x

        def sample(self, generator=None):
            return self.x

        def mode(self):
            return self.x

    class _Out:
        def __init__(self, x, attr):
            setattr(self, attr, x)

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def encode(self, x):
        return self._Out(self._Dist(x), "latent_dist")

    def decode(self, z):
        return self._Out(z, "sample")


@pytest.fixture
def mock_vae():
    return MockVAE()


def test_roundtrip_shape_and_dtype_non_multiple_of_8(mock_vae, sprite):
    img = sprite[:23, :21]  # deliberately not divisible by 8
    out = vae_roundtrip(img, mock_vae)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_roundtrip_identity_through_mock(mock_vae, sprite):
    out = vae_roundtrip(sprite, mock_vae, sample=False)
    assert np.array_equal(out, sprite)


def test_generate_pairs_manifest(tmp_path, mock_vae, sprite):
    from PIL import Image

    src = tmp_path / "src.png"
    Image.fromarray(sprite).save(src)
    outdir = tmp_path / "out"
    metas = generate_pairs([src], outdir, mock_vae, vae_name="mock", n_per_image=3, seed=7)

    assert len(metas) == 3
    lines = (outdir / "pairs.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    for line, meta in zip(lines, metas):
        rec = json.loads(line)
        assert (outdir / rec["clean"]).exists()
        assert (outdir / rec["corrupt"]).exists()
        assert 2.5 <= rec["scale_y"] <= 8.5
        assert rec["vae"] == "mock"
        # corrupted size must match the cropped nearest-upscale exactly
        corrupt = np.array(Image.open(outdir / rec["corrupt"]))
        expect_h = int(np.floor(sprite.shape[0] * rec["scale_y"] - 0.5 - rec["phase_y"] - 1e-9)) + 1
        expect_w = int(np.floor(sprite.shape[1] * rec["scale_x"] - 0.5 - rec["phase_x"] - 1e-9)) + 1
        assert corrupt.shape[:2] == (expect_h, expect_w)


def test_generate_pairs_deterministic(tmp_path, mock_vae, sprite):
    from PIL import Image

    src = tmp_path / "src.png"
    Image.fromarray(sprite).save(src)
    m1 = generate_pairs([src], tmp_path / "a", mock_vae, vae_name="mock", n_per_image=2, seed=3)
    m2 = generate_pairs([src], tmp_path / "b", mock_vae, vae_name="mock", n_per_image=2, seed=3)
    assert [(m.scale_y, m.phase_x, m.seed) for m in m1] == [
        (m.scale_y, m.phase_x, m.seed) for m in m2
    ]