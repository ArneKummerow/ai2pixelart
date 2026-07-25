"""Command-line interface.

    ai2pixelart clean input.png -o out.png                    # classical
    ai2pixelart clean input.png -o out.png --approach robust  # neural
    ai2pixelart viewer my_images/                             # web workspace
    ai2pixelart inspect my_images/                            # quality check
    ai2pixelart data ...   |   ai2pixelart train ...          # data + training
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
from PIL import Image


def _load(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _save_with_preview(out_path: str, image: np.ndarray, preview_scale: int) -> Path:
    """Write image to out_path (creating parents); optionally a NEAREST preview."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out)
    if preview_scale > 1:
        h, w = image.shape[:2]
        Image.fromarray(image).resize(
            (w * preview_scale, h * preview_scale), Image.NEAREST
        ).save(out.with_name(out.stem + "_preview.png"))
    return out




@click.group()
def main() -> None:
    """Turn AI-generated pseudo pixel art into pixel-perfect pixel art."""


# --------------------------------------------------------------------------
# user-facing commands
# --------------------------------------------------------------------------

@main.command("clean")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", required=True, type=click.Path(),
              help="Output file for a single image, or output folder when INPUT is a folder.")
@click.option("--approach", default="simple", show_default=True,
              help="'simple' (classical, no GPU) or a neural model: a run name "
                   "(uses runs/<name>/best.safetensors or best.ckpt) or a path "
                   "to a .safetensors/.ckpt file.")
@click.option("--colors", "max_colors", default=None, type=int,
              help="Palette size: exact target when the image has that many "
                   "distinguishable colors, cap otherwise (default: natural size).")
@click.option("--palette", default=None, help="Force palette: '#rrggbb,#rrggbb,...'.")
@click.option("--pitch", default=None, type=float,
              help="Known art-pixel size in image pixels (skips grid detection).")
@click.option("--detail", "granularity", default=1.0, show_default=True,
              help="Output resolution relative to the detected grid "
                   "(2 = 2x finer, 0.5 = 2x coarser).")
@click.option("--merge-de", default=3.0, show_default=True,
              help="Palette merge threshold (CIE76 ΔE); higher merges more shades.")
@click.option("--leash", default=None, type=float,
              help="Neural only: max ΔE a cell may move from its observed color "
                   "(lower = stricter; try 2-4 on dense palettes, 8 otherwise).")
@click.option("--denoise/--no-denoise", default=True, show_default=True,
              help="Median-filter each cell before reading its color.")
@click.option("--smooth/--no-smooth", default=True, show_default=True,
              help="De-speckle flat regions (detail-guarded).")
@click.option("--consensus", is_flag=True,
              help="Unify near-identical flat areas to one palette entry image-wide.")
@click.option("--device", default="auto", show_default=True,
              help="Neural only: 'auto' (GPU if available, else CPU), 'cuda', or 'cpu'.")
@click.option("--preview-scale", default=0, type=int,
              help="Also write an Nx nearest-neighbor preview.")
def clean_cmd(input_path, output_path, approach, max_colors, palette, pitch, granularity,
              merge_de, leash, denoise, smooth, consensus, device, preview_scale):
    """Clean pseudo-pixel-art into true-resolution pixel art.

    INPUT is one image, or a folder to batch-clean (then OUTPUT is a folder
    and each image is written as <name>.png). Uses the classical pipeline by
    default (--approach simple); pass --approach <model> for a trained net."""
    from ai2pixelart.palette import parse_hex_palette
    from ai2pixelart.webapp import collect_sources

    sources = collect_sources(Path(input_path))
    if not sources:
        raise click.ClickException(f"no images found in {input_path}")
    batch = Path(input_path).is_dir()
    pal = parse_hex_palette(palette) if palette else None
    pitch_pair = (pitch, pitch) if pitch else None
    shared = dict(
        merge_de=merge_de, max_colors=max_colors, palette=pal, pitch=pitch_pair,
        granularity=granularity, denoise=denoise, smooth=smooth, consensus=consensus,
    )

    # neural: resolve device + load the model ONCE, reused across all images
    model = device_used = None
    method = "classical"
    if approach != "simple":
        from ai2pixelart.models import (
            ModelNotAvailable,
            resolve_checkpoint,
            resolve_device,
        )
        from ai2pixelart.nninfer import load_checkpoint

        device = resolve_device(device)
        device_used = device
        try:
            ckpt = resolve_checkpoint(approach)
        except ModelNotAvailable as e:
            raise click.ClickException(str(e))
        model = load_checkpoint(str(ckpt), device=device)
        # label by the approach the user asked for (a name), not the cache
        # directory (which is a commit sha for hub-downloaded checkpoints)
        is_path = "/" in approach or approach.endswith((".ckpt", ".safetensors"))
        method = f"neural ({Path(approach).stem if is_path else approach})"

    if batch:
        Path(output_path).mkdir(parents=True, exist_ok=True)

    def clean_one(img):
        if model is None:
            from ai2pixelart.pipeline import clean

            return clean(img, **shared)
        from ai2pixelart.nninfer import nn_clean_image

        return nn_clean_image(
            img, model, device=device, max_colors=max_colors, palette=pal,
            leash=leash, merge_de=merge_de, pitch=pitch_pair, granularity=granularity,
            denoise=denoise, smooth=smooth, consensus=consensus,
        )

    infos = []
    for src in sources:
        result = clean_one(_load(str(src)))
        dest = (Path(output_path) / f"{src.stem}.png") if batch else output_path
        out = _save_with_preview(str(dest), result.image, preview_scale)
        infos.append({
            "input": src.name,
            "method": method,
            "device": device_used,
            "output": str(out),
            "output_shape": list(result.image.shape[:2]),
            "palette_size": int(len(result.palette)),
            "grid": None if result.grid is None else {
                "pitch_y": round(result.grid.y.pitch, 3),
                "pitch_x": round(result.grid.x.pitch, 3),
                "score_y": round(result.grid.y.score, 2),
                "score_x": round(result.grid.x.score, 2),
            },
        })
        if batch:
            h, w = result.image.shape[:2]
            click.echo(f"{src.name} -> {out}  ({w}x{h}, {len(result.palette)} colors)")

    if batch:
        click.echo(f"cleaned {len(infos)} images -> {output_path}")
    else:
        click.echo(json.dumps(infos[0], indent=2))


@main.command("viewer")
@click.argument("images_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--port", default=8412, show_default=True)
@click.option("--model", "ckpts", multiple=True,
              help="Neural checkpoint as NAME=PATH (repeatable) or bare PATH "
                   "(named after its run directory). Each becomes a "
                   "'Neural NAME' approach. Default: every runs/*/best.* ckpt.")
@click.option("--device", default="auto", show_default=True,
              help="Neural inference device: 'auto' (GPU if available, else CPU), 'cuda', or 'cpu'.")
def viewer_cmd(images_dir, port, ckpts, device):
    """Start the interactive web workspace over a folder of images.

    Opens a local web app (default http://127.0.0.1:8412) where you tune
    parameters live and compare approaches side by side. Everything is
    computed on demand and cached in memory only — nothing is precomputed
    or written to disk (except images you drop into the app)."""
    from ai2pixelart.webapp import serve

    named: dict[str, Path] = {}
    for spec in ckpts:
        name, _, path = spec.rpartition("=")
        path = Path(path)
        if not path.is_file():
            raise click.ClickException(f"checkpoint not found: {path}")
        named[name or path.parent.name] = path
    scan = None
    if not named:
        # no explicit pins: discover now AND keep scanning while serving,
        # so a training run that finishes appears without a restart
        from ai2pixelart.models import REGISTRY, discover_runs

        scan = Path("runs")
        named.update(discover_runs(scan))
        if named:
            click.echo(f"local neural models: {sorted(named)}")
        hub = sorted(n for n, s in REGISTRY.items() if s.repo_id and n not in named)
        if hub:
            click.echo(f"downloadable neural models: {hub} (fetched on first use)")
    serve(Path(images_dir), port=port, ckpts=named, ckpt_scan=scan, device=device)


@main.command("inspect")
@click.argument("src", type=click.Path(exists=True))
@click.option("--detail", "granularity", default=1.0, show_default=True,
              help="Output resolution relative to the detected grid.")
@click.option("--smooth/--no-smooth", default=True, show_default=True,
              help="De-speckle flat regions before assessing.")
def inspect_cmd(src, granularity, smooth):
    """Quality check without ground truth over SRC (an image or folder).

    Reports, per image, how well the classical pipeline handled it: grid
    boundary SNR, tile-wise pitch consistency, palette cell-fit, speckle
    rate, shade flicker, and detail survival. Marginal boundary SNR (~1)
    flags the fine-pitch images the method struggles with."""
    from ai2pixelart.autoqa import assess
    from ai2pixelart.pipeline import clean
    from ai2pixelart.webapp import collect_sources

    sources = collect_sources(Path(src))
    if not sources:
        raise click.ClickException(f"no images found in {src}")
    cols = ["boundary_snr", "pitch_consistency", "cell_fit_mean", "cell_fit_p95",
            "speckle_rate", "shade_flicker", "detail_survival", "n_details"]
    click.echo(f"{'image':<12}" + "".join(f"{c:>18}" for c in cols))
    for path in sources:
        img = _load(path)
        result = clean(img, smooth=smooth, granularity=granularity)
        if result.grid is None:
            click.echo(f"{path.stem:<12}{'no grid detected':>18}")
            continue
        report = assess(img, result)
        click.echo(f"{path.stem:<12}" + "".join(f"{str(report[c]):>18}" for c in cols))


@main.command("eval")
@click.argument("pred_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("gt_path", type=click.Path(exists=True, dir_okay=False))
def eval_cmd(pred_path, gt_path):
    """Compare a predicted true-resolution image against ground truth."""
    from ai2pixelart.metrics import evaluate

    click.echo(json.dumps(evaluate(_load(pred_path), _load(gt_path)), indent=2))


@main.command("demo")
@click.option("-o", "--outdir", default="examples/output/demo", show_default=True)
@click.option("--scale", default=3.3, show_default=True, help="Fake-pixel scale of the corruption.")
def demo_cmd(outdir, scale):
    """Synthetic sprite -> corruption -> cleanup -> metrics."""
    from ai2pixelart.demo import run_demo

    report = run_demo(outdir, scale=scale)
    click.echo(json.dumps(report, indent=2))
    click.echo(f"images written to {outdir}")


# --------------------------------------------------------------------------
# data engine
# --------------------------------------------------------------------------

@main.group("data")
def data() -> None:
    """Generate the training corpus (sprites, palettes, corruption pairs)."""


@data.command("sprites")
@click.option("-o", "--outdir", required=True, type=click.Path(file_okay=False))
@click.option("-n", "--count", default=1500, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--palette-pool", default=None, type=click.Path(exists=True, dir_okay=False),
              help="JSON palette pool (see 'data palettes'); ~half the sprites draw real palettes from it.")
def data_sprites_cmd(outdir, count, seed, palette_pool):
    """Generate procedural clean pixel-art sprites (training ground truth)."""
    from ai2pixelart.spritegen import generate_sprites

    paths = generate_sprites(outdir, count, seed=seed, progress=click.echo, pool_path=palette_pool)
    click.echo(f"wrote {len(paths)} sprites to {outdir}")


@data.command("palettes")
@click.argument("src", type=click.Path(exists=True))
@click.option("-o", "--out", "out_json", required=True, type=click.Path(dir_okay=False))
def data_palettes_cmd(src, out_json):
    """Extract real-image palettes (classical proposals) into a JSON pool
    for 'data sprites --palette-pool'."""
    from ai2pixelart.spritegen import build_palette_pool
    from ai2pixelart.webapp import collect_sources

    paths = collect_sources(Path(src))
    if not paths:
        raise click.ClickException(f"no images found in {src}")
    n = build_palette_pool(paths, out_json, progress=click.echo)
    click.echo(f"wrote {n} palettes to {out_json}")


@data.command("pairs")
@click.argument("sources", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-o", "--outdir", required=True, type=click.Path(file_okay=False))
@click.option("-n", "--n-per-image", default=4, show_default=True)
@click.option("--scale-min", default=3.0, show_default=True, help="Floor 3.0: smaller cells dissolve in the f8 VAE.")
@click.option("--scale-max", default=8.0, show_default=True)
@click.option("--vae", "vae_id", default=None, help="Diffusers VAE model id.")
@click.option("--seed", default=0, show_default=True)
@click.option("--device", default="cuda", show_default=True)
def data_pairs_cmd(sources, outdir, n_per_image, scale_min, scale_max, vae_id, seed, device):
    """Generate aligned (clean, VAE-corrupted) training pairs + manifest.

    SOURCES are image files and/or directories of images."""
    from ai2pixelart.pairgen import generate_pairs
    from ai2pixelart.vae import DEFAULT_VAE, load_vae
    from ai2pixelart.webapp import collect_sources

    src_paths = [p for s in sources for p in collect_sources(Path(s))]
    if not src_paths:
        raise click.ClickException("no images found")
    vae_id = vae_id or DEFAULT_VAE
    vae = load_vae(vae_id, device=device)
    metas = generate_pairs(
        src_paths, Path(outdir), vae, vae_name=vae_id,
        n_per_image=n_per_image, scale_range=(scale_min, scale_max), seed=seed,
    )
    click.echo(f"wrote {len(metas)} pairs to {outdir} (manifest: pairs.jsonl)")


@data.command("edit-pairs")
@click.argument("sources", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-o", "--outdir", required=True, type=click.Path(file_okay=False))
@click.option("-n", "--n-per-image", default=1, show_default=True)
@click.option("--scale-min", default=3.0, show_default=True)
@click.option("--scale-max", default=12.0, show_default=True)
@click.option("--strength-min", default=0.15, show_default=True)
@click.option("--strength-max", default=0.45, show_default=True)
@click.option("--editor", "editor_id", default=None, help="Diffusers img2img model id.")
@click.option("--backend", type=click.Choice(["sdxl", "vae"]), default="sdxl", show_default=True,
              help="Corruptor: sdxl img2img (generative prior) or plain VAE roundtrip.")
@click.option("--downscale-frac", default=0.0, show_default=True,
              help="Fraction of pairs corrupted at 4.2-6.4 then box-halved to reach the fine-pitch regime (2.1-3.2 px cells).")
@click.option("--shade-prob", default=0.0, show_default=True,
              help="Fraction of pairs given a random background gradient/vignette before corruption (targets stay flat — teaches gradient collapse).")
@click.option("--noise-prob", default=0.0, show_default=True,
              help="Fraction of pairs given heavy per-pixel noise after corruption (targets stay clean — matches noisy real assets).")
@click.option("--lora", "lora_path", default=None, type=click.Path(exists=True),
              help="Corruptor LoRA directory (see 'train corruptor') — enables higher strengths.")
@click.option("--seed", default=0, show_default=True)
@click.option("--device", default="cuda", show_default=True)
def data_edit_pairs_cmd(sources, outdir, n_per_image, scale_min, scale_max,
                        strength_min, strength_max, editor_id, backend,
                        downscale_frac, shade_prob, noise_prob, lora_path, seed, device):
    """Generate rail-guarded img2img training pairs (masks included).

    SOURCES are image files and/or directories of images."""
    from ai2pixelart.pairgen import generate_edit_pairs
    from ai2pixelart.webapp import collect_sources

    src_paths = [p for s in sources for p in collect_sources(Path(s))]
    if not src_paths:
        raise click.ClickException("no images found")
    if backend == "vae":
        from ai2pixelart.vae import DEFAULT_VAE, load_vae, vae_roundtrip

        vae = load_vae(editor_id or DEFAULT_VAE, device=device)
        pipe = None
        editor_name = editor_id or DEFAULT_VAE

        def corrupt_fn(img, pipe, strength, seed):
            return vae_roundtrip(img, vae, sample=True, seed=seed)
    else:
        from ai2pixelart.imgedit import DEFAULT_EDITOR, load_editor

        editor_name = editor_id or DEFAULT_EDITOR
        pipe = load_editor(editor_name, device=device, lora=lora_path)
        if lora_path:
            editor_name += f"+lora:{Path(lora_path).name}"
        corrupt_fn = None
    metas = generate_edit_pairs(
        src_paths, Path(outdir), pipe, editor_name=editor_name,
        n_per_image=n_per_image, scale_range=(scale_min, scale_max),
        strength_range=(strength_min, strength_max), seed=seed,
        downscale_frac=downscale_frac, shade_prob=shade_prob, noise_prob=noise_prob,
        corrupt_fn=corrupt_fn, progress=click.echo,
    )
    click.echo(f"wrote {len(metas)} rail-guarded pairs to {outdir}")


@data.command("vae-roundtrip")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", "output_path", required=True, type=click.Path(dir_okay=False))
@click.option("--scale", default=None, type=float, help="Nearest-upscale factor applied first (also writes the upscaled reference next to the output).")
@click.option("--phase", default=0.0, type=float, help="Sub-cell phase offset for --scale.")
@click.option("--vae", "vae_id", default=None, help="Diffusers VAE model id.")
@click.option("--seed", default=None, type=int)
@click.option("--device", default="cuda", show_default=True)
def data_vae_roundtrip_cmd(input_path, output_path, scale, phase, vae_id, seed, device):
    """Encode+decode an image through a diffusion VAE (artifact inspection)."""
    from ai2pixelart.corrupt import upscale
    from ai2pixelart.vae import DEFAULT_VAE, load_vae, vae_roundtrip

    img = _load(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if scale is not None:
        img = upscale(img, scale=scale, phase=(phase, phase), interp="nearest")
        Image.fromarray(img).save(out.with_name(out.stem + "_reference.png"))
    vae = load_vae(vae_id or DEFAULT_VAE, device=device)
    Image.fromarray(vae_roundtrip(img, vae, seed=seed)).save(out)
    click.echo(f"wrote {out}")


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

@main.group("train")
def train_grp() -> None:
    """Train the restoration net or the corruptor LoRA."""


@train_grp.command("model")
@click.option("--pairs", "pairs_dir", required=True, multiple=True, type=click.Path(exists=True, file_okay=False),
              help="Pair directories from 'data pairs' / 'data edit-pairs' (repeatable).")
@click.option("-o", "--outdir", required=True, type=click.Path(file_okay=False))
@click.option("--steps", default=4000, show_default=True)
@click.option("--batch-size", default=16, show_default=True)
@click.option("--lr", default=3e-4, show_default=True)
@click.option("--val-every", default=500, show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--decoy-max", default=48, show_default=True,
              help="Max decoy palette entries appended per training sample (0 disables).")
@click.option("--detail-weight", default=8.0, show_default=True,
              help="Loss upweight for pixels of isolated 1-px details.")
def train_model_cmd(pairs_dir, outdir, steps, batch_size, lr, val_every, device, seed,
                    detail_weight, decoy_max):
    """Train the palette-classification restoration net on corruption pairs."""
    from ai2pixelart.train import train

    final = train(
        pairs_dir, outdir, steps=steps, batch_size=batch_size, lr=lr,
        val_every=val_every, device=device, seed=seed, detail_weight=detail_weight,
        decoy_max=decoy_max, log=click.echo,
    )
    click.echo(json.dumps(final, indent=2))


@train_grp.command("corruptor")
@click.option("--pairs", "pair_dirs", required=True, multiple=True, type=click.Path(exists=True, file_okay=False),
              help="img2img pair dirs whose rail-accepted outputs form the corruption domain.")
@click.option("--real", "real_dirs", multiple=True, type=click.Path(exists=True),
              help="Real AI image folders (the target look; oversampled).")
@click.option("-o", "--outdir", required=True, type=click.Path(file_okay=False))
@click.option("--steps", default=2000, show_default=True)
@click.option("--batch-size", default=4, show_default=True)
@click.option("--rank", default=16, show_default=True)
@click.option("--lr", default=1e-4, show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--seed", default=0, show_default=True)
def train_corruptor_cmd(pair_dirs, real_dirs, outdir, steps, batch_size, rank, lr, device, seed):
    """LoRA-finetune the SDXL corruptor on its own rail-accepted outputs
    so higher img2img strengths stay content-faithful."""
    from ai2pixelart.corruptor import train_corruptor_lora

    out = train_corruptor_lora(
        [Path(p) for p in pair_dirs], [Path(p) for p in real_dirs], outdir,
        steps=steps, batch_size=batch_size, rank=rank, lr=lr,
        device=device, seed=seed, log=click.echo,
    )
    click.echo(f"corruptor LoRA at {out} — use 'data edit-pairs --lora {out}'")


# --------------------------------------------------------------------------
# models (registry + checkpoint format)
# --------------------------------------------------------------------------

@main.group("models")
def models_grp() -> None:
    """Inspect the model registry and convert checkpoints to safetensors."""


@models_grp.command("list")
def models_list_cmd():
    """Show registered models and which checkpoints are available locally."""
    from ai2pixelart.models import REGISTRY, discover_runs

    local = discover_runs("runs")
    names = sorted(set(REGISTRY) | set(local))
    if not names:
        click.echo("no models registered or found under runs/")
        return
    for name in names:
        path = local.get(name)
        where = str(path) if path else "not available locally"
        spec = REGISTRY.get(name)
        desc = f" — {spec.description}" if spec and spec.description else ""
        click.echo(f"{name:<10} {where}{desc}")


@models_grp.command("download")
@click.argument("names", nargs=-1)
def models_download_cmd(names):
    """Pre-fetch registered model weights from the Hub into the local cache.

    With no NAMES, downloads every registered model. Useful to warm the
    cache before going offline; normally weights fetch automatically on
    first use."""
    from ai2pixelart.models import REGISTRY, ModelNotAvailable, download_model

    targets = list(names) or [n for n, s in REGISTRY.items() if s.repo_id]
    if not targets:
        raise click.ClickException("no downloadable models in the registry")
    for name in targets:
        try:
            path = download_model(name)
        except ModelNotAvailable as e:
            raise click.ClickException(str(e))
        click.echo(f"{name}: {path}")


@models_grp.command("convert")
@click.argument("src", required=False, type=click.Path(exists=True))
@click.option("-o", "--output", "output", default=None, type=click.Path(dir_okay=False),
              help="Destination .safetensors (default: source with .safetensors suffix).")
@click.option("--all", "all_runs", is_flag=True,
              help="Convert every runs/*/best.ckpt to a sibling best.safetensors.")
def models_convert_cmd(src, output, all_runs):
    """Convert a legacy torch .ckpt to .safetensors (config -> metadata).

    Pass a .ckpt file, or --all to convert every runs/*/best.ckpt."""
    from ai2pixelart.models import convert_checkpoint

    if all_runs:
        if src or output:
            raise click.ClickException("--all takes no SRC/--output")
        ckpts = sorted(Path("runs").glob("*/best.ckpt"))
        if not ckpts:
            raise click.ClickException("no runs/*/best.ckpt found")
        for c in ckpts:
            out = convert_checkpoint(c)
            click.echo(f"{c} -> {out}")
        return
    if not src:
        raise click.ClickException("give a .ckpt SRC or use --all")
    out = convert_checkpoint(src, output)
    click.echo(f"{src} -> {out}")
