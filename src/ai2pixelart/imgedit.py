"""SDXL img2img corruption backend.

Re-renders nearest-upscaled clean pixel art through a latent diffusion
img2img pass at low strength: unlike the pure VAE roundtrip, the denoising
steps apply the model's generative prior — soft shading, antialiasing
choices, subtle reinterpretation — which is exactly the corruption class
real AI pseudo-pixel-art exhibits and the VAE cannot produce. Alignment is
NOT guaranteed (the editor may move content); pairs from this backend must
carry rails.validity_mask.
"""

from __future__ import annotations

import numpy as np

DEFAULT_EDITOR = "stabilityai/stable-diffusion-xl-base-1.0"
# steer the re-render toward the pseudo-pixel-art look, not photo realism
DEFAULT_PROMPT = "pixel art, retro game sprite, crisp pixel grid, flat colors"
NEGATIVE_PROMPT = "photo, realistic, blurry, smooth painting"


def load_editor(model_id: str = DEFAULT_EDITOR, device: str = "cuda", lora: str | None = None):
    import torch

    from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.float16, variant="fp16"
    ).to(device)
    if lora:
        # corruptor LoRA: keeps high-strength re-renders on the fake-pixel
        # grid instead of recomposing
        pipe.load_lora_weights(lora)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def edit_corrupt(
    img: np.ndarray,
    pipe,
    strength: float = 0.25,
    prompt: str = DEFAULT_PROMPT,
    seed: int | None = None,
    steps: int = 30,
    guidance: float = 5.0,
) -> np.ndarray:
    """img2img re-render of an (H, W, 3) uint8 image, same size out.

    The image is edge-padded to a multiple of 8 before the pipeline (its own
    preprocessor would RESIZE to a multiple of 8, silently breaking the
    pixel alignment the pair metadata promises) and cropped back after.
    """
    import torch
    from PIL import Image

    h, w = img.shape[:2]
    ph, pw = (-h) % 8, (-w) % 8
    padded = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge")
    gen = torch.Generator("cpu").manual_seed(seed if seed is not None else 0)
    out = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=Image.fromarray(padded),
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=gen,
    ).images[0]
    return np.asarray(out.convert("RGB"))[:h, :w]
