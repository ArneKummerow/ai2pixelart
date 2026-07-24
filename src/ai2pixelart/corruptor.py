"""Milestone 4: corruptor LoRA — a faithful high-strength img2img editor.

Base SDXL drifts at img2img strengths above ~0.45 (its prior pulls toward
generic imagery, so content moves and the rails reject the pair). The fix
is a domain-adaptation LoRA: finetune the UNet's denoising prior on the
corruption domain itself — the rail-ACCEPTED corrupted outputs plus the
real AI images (the gold-standard look, oversampled). With the prior
anchored in pseudo-pixel-art, high-strength denoising stays on the fake
grid instead of recomposing, so the strength knobs can go up and the pairs
get harsher while staying aligned (rejection-sampling distillation, as
planned).

Training details that matter:
- fixed prompt = the exact prompt `edit_corrupt` uses at generation time,
  so the LoRA's behavior is activated by the same conditioning;
- timesteps biased to t <= 0.75*T: img2img at strength s only ever runs
  the last s of the noise schedule — training effort goes where inference
  happens;
- crops of 512 px: the editor's working scale for our pair sizes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LORA_RANK = 16
TRAIN_SIZE = 512
TIMESTEP_FRAC = 0.75


def harvest_domain_images(
    pair_dirs: list[Path], real_dirs: list[Path], real_oversample: int = 32
) -> list[Path]:
    """Corruption-domain image paths: rail-accepted corrupted outputs from
    img2img corpora plus real AI images (oversampled — they are few but are
    the true target distribution)."""
    paths: list[Path] = []
    for d in pair_dirs:
        d = Path(d)
        manifest = d / "pairs.jsonl"
        if not manifest.exists():
            continue
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            meta = json.loads(line)
            if meta.get("corruption") == "img2img" and meta.get("mask"):
                paths.append(d / meta["corrupt"])
    from ai2pixelart.webapp import collect_sources

    for d in real_dirs:
        real = collect_sources(Path(d))
        paths.extend(real * real_oversample)
    return paths


class DomainCrops:
    """Random square crops of domain images as [-1, 1] float tensors."""

    def __init__(self, paths: list[Path], size: int = TRAIN_SIZE, seed: int = 0):
        self.paths = paths
        self.size = size
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        import torch
        from PIL import Image

        img = np.array(Image.open(self.paths[i]).convert("RGB"))
        h, w = img.shape[:2]
        s = self.size
        if h < s or w < s:  # small pairs: pad by reflection to crop size
            img = np.pad(img, ((0, max(0, s - h)), (0, max(0, s - w)), (0, 0)), mode="reflect")
            h, w = img.shape[:2]
        y0 = int(self._rng.integers(0, h - s + 1))
        x0 = int(self._rng.integers(0, w - s + 1))
        crop = img[y0 : y0 + s, x0 : x0 + s]
        return torch.from_numpy(crop.transpose(2, 0, 1)).float() / 127.5 - 1.0


def train_corruptor_lora(
    pair_dirs: list[Path],
    real_dirs: list[Path],
    outdir: str | Path,
    base_model: str | None = None,
    steps: int = 2000,
    batch_size: int = 4,
    rank: int = LORA_RANK,
    lr: float = 1e-4,
    device: str = "cuda",
    seed: int = 0,
    log=print,
) -> Path:
    """LoRA-finetune the SDXL UNet on the corruption domain; returns the
    LoRA directory (loadable via `gen-edit-pairs --lora`)."""
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from torch.utils.data import DataLoader

    from ai2pixelart.imgedit import DEFAULT_EDITOR, DEFAULT_PROMPT

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    paths = harvest_domain_images(pair_dirs, real_dirs)
    if not paths:
        raise ValueError("no domain images harvested")
    log(f"domain images: {len(paths)} (incl. oversampled real)")

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    ).to(device)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model or DEFAULT_EDITOR, vae=vae, torch_dtype=torch.float16, variant="fp16"
    ).to(device)
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    with torch.no_grad():
        embeds, _, pooled, _ = pipe.encode_prompt(prompt=DEFAULT_PROMPT, device=device)
    size_cond = torch.tensor(
        [[TRAIN_SIZE, TRAIN_SIZE, 0, 0, TRAIN_SIZE, TRAIN_SIZE]], device=device
    )
    del pipe.text_encoder, pipe.text_encoder_2

    unet = pipe.unet.to(device=device, dtype=torch.float32)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    ))
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    log(f"LoRA params: {sum(p.numel() for p in lora_params) / 1e6:.2f}M")
    opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))

    ds = DomainCrops(paths, seed=seed)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=2,
        persistent_workers=True, drop_last=True,
    )
    t_max = int(scheduler.config.num_train_timesteps * TIMESTEP_FRAC)

    step, losses = 0, []
    unet.train()
    while step < steps:
        for imgs in loader:
            if step >= steps:
                break
            imgs = imgs.to(device)
            with torch.no_grad():
                lat = vae.encode(imgs.half()).latent_dist.sample().float()
                lat = lat * vae.config.scaling_factor
            noise = torch.randn_like(lat)
            t = torch.randint(0, t_max, (lat.shape[0],), device=device)
            noisy = scheduler.add_noise(lat, noise, t)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                pred = unet(
                    noisy, t,
                    encoder_hidden_states=embeds.float().expand(lat.shape[0], -1, -1),
                    added_cond_kwargs={
                        "text_embeds": pooled.float().expand(lat.shape[0], -1),
                        "time_ids": size_cond.expand(lat.shape[0], -1),
                    },
                ).sample
                loss = F.mse_loss(pred.float(), noise)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            losses.append(float(loss.detach()))
            if step % 100 == 0:
                log(f"step {step}/{steps} loss {np.mean(losses):.4f}")
                losses = []

    lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(outdir, unet_lora_layers=lora_state)
    log(f"LoRA saved to {outdir}")
    return outdir
