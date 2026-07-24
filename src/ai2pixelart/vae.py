"""VAE roundtrips: genuine latent-diffusion artifacts with zero content drift.

Encoding nearest-upscaled pixel art through a latent diffusion VAE and
decoding it back reproduces the artifact family we must learn to invert —
color bleed, grid wobble, extra shades — without moving any content, so
(clean, roundtripped) pairs are perfectly aligned by construction.

Torch/diffusers are imported lazily so the classical toolkit stays usable
in environments without them.
"""

from __future__ import annotations

import numpy as np

DEFAULT_VAE = "madebyollin/sdxl-vae-fp16-fix"


def load_vae(model_id: str = DEFAULT_VAE, device: str = "cuda", dtype=None):
    """Load a diffusers AutoencoderKL in eval mode on `device`."""
    import torch
    from diffusers import AutoencoderKL

    if dtype is None:
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
    vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=dtype)
    return vae.eval().to(device)


def vae_roundtrip(
    img: np.ndarray, vae, sample: bool = True, seed: int | None = None
) -> np.ndarray:
    """Encode + decode one (H, W, 3) uint8 image through `vae`.

    sample: draw from the posterior (adds artifact diversity across seeds)
        instead of taking its mode. Content is unaffected either way.
    Sizes need not be multiples of 8; the image is replicate-padded for the
    encoder and cropped back afterwards.
    """
    import torch

    param = next(vae.parameters())
    h, w = img.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1)[None]
    x = x / 127.5 - 1.0
    x = torch.nn.functional.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode="replicate")
    x = x.to(param.device, param.dtype)

    with torch.no_grad():
        posterior = vae.encode(x).latent_dist
        if sample:
            gen = None
            if seed is not None:
                gen = torch.Generator(device=param.device).manual_seed(seed)
            z = posterior.sample(generator=gen)
        else:
            z = posterior.mode()
        y = vae.decode(z).sample

    y = ((y.float().clamp(-1.0, 1.0) + 1.0) * 127.5).round()
    y = y[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return y[:h, :w]
