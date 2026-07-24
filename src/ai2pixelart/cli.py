"""Command-line interface.

    ai2pixelart clean input.png -o out.png [--preview-scale 8]
    ai2pixelart eval pred.png gt.png
    ai2pixelart demo -o examples/output/demo
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
from PIL import Image


def _load(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


@click.group()
def main() -> None:
    """Turn AI-generated pseudo pixel art into pixel-perfect pixel art."""


@main.command("clean")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", "output_path", required=True, type=click.Path(dir_okay=False))
@click.option("--merge-de", default=3.0, show_default=True, help="Palette merge threshold (CIE76 ΔE).")
@click.option("--max-colors", default=None, type=int, help="Palette size: exact target when the image has that many distinguishable colors, cap otherwise.")
@click.option("--palette", default=None, help="Force palette: '#rrggbb,#rrggbb,...'.")
@click.option("--pitch", default=None, type=float, help="Known cell pitch (skips estimation).")
@click.option("--granularity", default=1.0, show_default=True, help="Output resolution relative to the detected grid (2 = 2x finer, 0.5 = 2x coarser; integer factors).")
@click.option("--preview-scale", default=0, type=int, help="Also write an Nx nearest-neighbor preview.")
def clean_cmd(input_path, output_path, merge_de, max_colors, palette, pitch, granularity, preview_scale):
    """Clean one pseudo-pixel-art image to true-resolution pixel art."""
    from ai2pixelart.palette import parse_hex_palette
    from ai2pixelart.pipeline import clean

    img = _load(input_path)
    result = clean(
        img,
        merge_de=merge_de,
        max_colors=max_colors,
        palette=parse_hex_palette(palette) if palette else None,
        pitch=(pitch, pitch) if pitch else None,
        granularity=granularity,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.image).save(out)
    if preview_scale > 1:
        h, w = result.image.shape[:2]
        Image.fromarray(result.image).resize(
            (w * preview_scale, h * preview_scale), Image.NEAREST
        ).save(out.with_name(out.stem + "_preview.png"))

    info = {
        "output_shape": list(result.image.shape[:2]),
        "palette_size": int(len(result.palette)),
        "grid": None
        if result.grid is None
        else {
            "pitch_y": round(result.grid.y.pitch, 3),
            "pitch_x": round(result.grid.x.pitch, 3),
            "score_y": round(result.grid.y.score, 2),
            "score_x": round(result.grid.x.score, 2),
        },
    }
    click.echo(json.dumps(info, indent=2))


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


@main.command("qa")
@click.argument("src", type=click.Path(exists=True))
@click.option("--smooth/--no-smooth", default=True, show_default=True)
@click.option("--granularity", default=1.0, show_default=True)
def qa_cmd(src, smooth, granularity):
    """No-ground-truth quality assessment of the auto pipeline over SRC
    (an image or folder): grid fit, palette representation, flatness,
    detail survival. See autoqa module docstring for metric meanings."""
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


@main.command("view")
@click.argument("images_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--port", default=8412, show_default=True)
@click.option("--ckpt", "ckpts", multiple=True,
              help="Neural checkpoint as NAME=PATH (repeatable) or bare PATH "
                   "(named after its run directory). Each becomes a "
                   "'Neural (NAME)' preset. Default: every runs/*/best.ckpt.")
def view_cmd(images_dir, port, ckpts):
    """Serve the interactive workspace viewer over a folder of images.

    Everything is computed live and cached in memory only — nothing is
    precomputed or written to disk (except images you drop into the app)."""
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
        scan = Path("runs")
        for path in sorted(scan.glob("*/best.ckpt")):
            named[path.parent.name] = path
        if named:
            click.echo(f"using neural checkpoints {sorted(named)} (pass --ckpt to override)")
    serve(Path(images_dir), port=port, ckpts=named, ckpt_scan=scan)


@main.command("vae-roundtrip")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", "output_path", required=True, type=click.Path(dir_okay=False))
@click.option("--scale", default=None, type=float, help="Nearest-upscale factor applied first (also writes the upscaled reference next to the output).")
@click.option("--phase", default=0.0, type=float, help="Sub-cell phase offset for --scale.")
@click.option("--vae", "vae_id", default=None, help="Diffusers VAE model id.")
@click.option("--seed", default=None, type=int)
@click.option("--device", default="cuda", show_default=True)
def vae_roundtrip_cmd(input_path, output_path, scale, phase, vae_id, seed, device):
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


@main.command("gen-sprites")
@click.option("-o", "--outdir", required=True, type=click.Path(file_okay=False))
@click.option("-n", "--count", default=1500, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--palette-pool", default=None, type=click.Path(exists=True, dir_okay=False),
              help="JSON palette pool (see extract-palettes); ~half the sprites draw real palettes from it.")
def gen_sprites_cmd(outdir, count, seed, palette_pool):
    """Generate procedural clean pixel-art sprites (training ground truth)."""
    from ai2pixelart.spritegen import generate_sprites

    paths = generate_sprites(outdir, count, seed=seed, progress=click.echo, pool_path=palette_pool)
    click.echo(f"wrote {len(paths)} sprites to {outdir}")


@main.command("extract-palettes")
@click.argument("src", type=click.Path(exists=True))
@click.option("-o", "--out", "out_json", required=True, type=click.Path(dir_okay=False))
def extract_palettes_cmd(src, out_json):
    """Extract real-image palettes (classical proposals) into a JSON pool
    for gen-sprites --palette-pool."""
    from ai2pixelart.spritegen import build_palette_pool
    from ai2pixelart.webapp import collect_sources

    paths = collect_sources(Path(src))
    if not paths:
        raise click.ClickException(f"no images found in {src}")
    n = build_palette_pool(paths, out_json, progress=click.echo)
    click.echo(f"wrote {n} palettes to {out_json}")


@main.command("gen-edit-pairs")
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
              help="Corruptor LoRA directory (see train-corruptor) — enables higher strengths.")
@click.option("--seed", default=0, show_default=True)
@click.option("--device", default="cuda", show_default=True)
def gen_edit_pairs_cmd(sources, outdir, n_per_image, scale_min, scale_max,
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


@main.command("train-corruptor")
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
    (milestone 4) so higher img2img strengths stay content-faithful."""
    from ai2pixelart.corruptor import train_corruptor_lora

    out = train_corruptor_lora(
        [Path(p) for p in pair_dirs], [Path(p) for p in real_dirs], outdir,
        steps=steps, batch_size=batch_size, rank=rank, lr=lr,
        device=device, seed=seed, log=click.echo,
    )
    click.echo(f"corruptor LoRA at {out} — use gen-edit-pairs --lora {out}")


@main.command("train")
@click.option("--pairs", "pairs_dir", required=True, multiple=True, type=click.Path(exists=True, file_okay=False))
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
def train_cmd(pairs_dir, outdir, steps, batch_size, lr, val_every, device, seed, detail_weight,
              decoy_max):
    """Train the palette-classification restoration net on gen-pairs data."""
    from ai2pixelart.train import train

    final = train(
        pairs_dir, outdir, steps=steps, batch_size=batch_size, lr=lr,
        val_every=val_every, device=device, seed=seed, detail_weight=detail_weight,
        decoy_max=decoy_max,
        log=click.echo,
    )
    click.echo(json.dumps(final, indent=2))


@main.command("nn-clean")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", "output_path", required=True, type=click.Path(dir_okay=False))
@click.option("--ckpt", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--device", default="cuda", show_default=True)
@click.option("--max-colors", default=None, type=int, help="Palette proposal size (target and cap); default: natural size, soft-capped at 16.")
@click.option("--palette", default=None, help="Force palette: '#rrggbb,#rrggbb,...'.")
@click.option("--preview-scale", default=0, type=int)
def nn_clean_cmd(input_path, output_path, ckpt, device, max_colors, palette, preview_scale):
    """Clean an image with the trained restoration net."""
    from ai2pixelart.nninfer import load_checkpoint, nn_clean_image
    from ai2pixelart.palette import parse_hex_palette

    model = load_checkpoint(ckpt, device=device)
    result = nn_clean_image(
        _load(input_path), model, device=device, max_colors=max_colors,
        palette=parse_hex_palette(palette) if palette else None,
    )
    out_img = result.image
    info = {
        "grid": None
        if result.grid is None
        else {"pitch_y": result.grid.y.pitch, "pitch_x": result.grid.x.pitch},
        "palette_size": int(len(result.palette)),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out_img).save(out)
    if preview_scale > 1:
        h, w = out_img.shape[:2]
        Image.fromarray(out_img).resize((w * preview_scale, h * preview_scale), Image.NEAREST).save(
            out.with_name(out.stem + "_preview.png")
        )
    click.echo(json.dumps({"output_shape": list(out_img.shape[:2]), **info}, indent=2))


@main.command("gen-pairs")
@click.argument("sources", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-o", "--outdir", required=True, type=click.Path(file_okay=False))
@click.option("-n", "--n-per-image", default=4, show_default=True)
@click.option("--scale-min", default=3.0, show_default=True, help="Floor 3.0: smaller cells dissolve in the f8 VAE.")
@click.option("--scale-max", default=8.0, show_default=True)
@click.option("--vae", "vae_id", default=None, help="Diffusers VAE model id.")
@click.option("--seed", default=0, show_default=True)
@click.option("--device", default="cuda", show_default=True)
def gen_pairs_cmd(sources, outdir, n_per_image, scale_min, scale_max, vae_id, seed, device):
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
        src_paths,
        Path(outdir),
        vae,
        vae_name=vae_id,
        n_per_image=n_per_image,
        scale_range=(scale_min, scale_max),
        seed=seed,
    )
    click.echo(f"wrote {len(metas)} pairs to {outdir} (manifest: pairs.jsonl)")
