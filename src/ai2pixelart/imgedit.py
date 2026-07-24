"""SDXL img2img corruption backend (milestone 3, first editor).

Re-renders nearest-upscaled clean pixel art through a latent diffusion
img2img pass at low strength: unlike the pure VAE roundtrip, the denoising
steps apply the model's generative prior — soft shading, antialiasing
choices, subtle reinterpretation — which is exactly the corruption class
real AI pseudo-pixel-art exhibits and the VAE cannot produce. Alignment is
NOT guaranteed (the editor may move content); pairs from this backend must
carry rails.validity_mask.

SDXL was chosen as the first backend because it is ungated and small enough
for the local cards; FLUX.1 Kontext dev (instruction editing, gated) and
Qwen-Image-Edit are the planned upgrades.
"""

from __future__ import annotations

import numpy as np

DEFAULT_EDITOR = "stabilityai/stable-diffusion-xl-base-1.0"
KONTEXT_EDITOR = "black-forest-labs/FLUX.1-Kontext-dev"
# steer the re-render toward the pseudo-pixel-art look, not photo realism
DEFAULT_PROMPT = "pixel art, retro game sprite, crisp pixel grid, flat colors"
NEGATIVE_PROMPT = "photo, realistic, blurry, smooth painting"
# Kontext is an instruction editor with no strength knob: ask it to re-render
# faithfully and let its generative pass supply the corruption; the rails
# mask whatever it moves anyway
KONTEXT_PROMPT = "redraw this pixel art exactly as it is, same colors, same details"


def load_editor(model_id: str = DEFAULT_EDITOR, device: str = "cuda", lora: str | None = None):
    import torch

    if "kontext" in model_id.lower():
        from diffusers import FluxKontextPipeline

        # bf16 even on Turing (no native support -> emulated, slower):
        # FLUX in fp16 is overflow-prone and correctness beats speed here
        pipe = FluxKontextPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload(device=device)  # 12B + T5: offload to fit comfortably
        pipe.set_progress_bar_config(disable=True)
        return pipe

    from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.float16, variant="fp16"
    ).to(device)
    if lora:
        # corruptor LoRA (milestone 4): keeps high-strength re-renders on
        # the fake-pixel grid instead of recomposing
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
    kontext = pipe.__class__.__name__.startswith("FluxKontext")
    mult = 16 if kontext else 8
    ph, pw = (-h) % mult, (-w) % mult
    padded = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge")
    gen = torch.Generator("cpu").manual_seed(seed if seed is not None else 0)
    if kontext:
        out = pipe(
            prompt=KONTEXT_PROMPT,
            image=Image.fromarray(padded),
            width=padded.shape[1],
            height=padded.shape[0],
            num_inference_steps=max(12, int(steps * 2 * strength)),  # strength scales effort
            guidance_scale=2.5,
            generator=gen,
        ).images[0]
    else:
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
