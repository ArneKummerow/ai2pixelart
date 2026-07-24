"""Training loop for PixelCleanNet on gen-pairs data."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ai2pixelart.metrics import evaluate
from ai2pixelart.nndata import PairDataset, majority_vote_cells, source_cell_maps
from ai2pixelart.nninfer import predict_indices


def evaluate_model(model, dataset: PairDataset, device: str, limit: int = 24) -> dict:
    """Image-level eval with the KNOWN grid from pair metadata: full forward,
    per-cell majority vote, then the standard metric suite vs clean GT."""
    model.eval()
    rows_out = []
    for meta in dataset.items[:limit]:
        corrupt, _, palette = dataset.load_pair(meta)
        clean_img = np.array(Image.open(dataset.path_of(meta, "clean")).convert("RGB"))
        pred_idx = predict_indices(model, corrupt, palette, device)
        rows, cols = source_cell_maps(
            meta, corrupt.shape[0], corrupt.shape[1], clean_img.shape[0], clean_img.shape[1]
        )
        cell_idx = majority_vote_cells(
            pred_idx, rows, cols, clean_img.shape[0], clean_img.shape[1], len(palette)
        )
        rep = evaluate(palette[cell_idx], clean_img)
        rows_out.append(
            (rep["cells"]["exact"], rep["cells"]["tolerant"], rep["details"]["rate"])
        )
    model.train()
    arr = np.asarray(rows_out, dtype=float)
    return {
        "exact": round(float(arr[:, 0].mean()), 4),
        "tolerant": round(float(arr[:, 1].mean()), 4),
        "detail_rate": round(float(np.nanmean(arr[:, 2])), 4),
        "n_images": len(rows_out),
    }


def train(
    pairs_dir: str | Path,
    outdir: str | Path,
    steps: int = 4000,
    batch_size: int = 16,
    lr: float = 3e-4,
    device: str = "cuda",
    val_every: int = 500,
    workers: int = 2,
    seed: int = 0,
    base: int = 48,
    detail_weight: float = 8.0,
    decoy_max: int = 48,
    log=print,
) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from ai2pixelart.nnmodel import PixelCleanNet

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    ds_train = PairDataset(
        pairs_dir, split="train", seed=seed, detail_weight=detail_weight, decoy_max=decoy_max
    )
    ds_val = PairDataset(pairs_dir, split="val", seed=seed)
    log(
        f"train pairs: {len(ds_train)} ({ds_train.n_dropped_small} dropped < crop), "
        f"val pairs: {len(ds_val)}"
    )
    loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=workers, persistent_workers=workers > 0,
    )

    config = {"base": base}
    model = PixelCleanNet(**config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model params: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))

    metrics_path = outdir / "metrics.jsonl"
    best_tolerant = -1.0
    step, t0 = 0, time.perf_counter()
    running_loss, running_acc = [], []

    model.train()
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            img = batch["img"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            weight = batch["weight"].to(device, non_blocking=True)
            palette = batch["palette"].to(device, non_blocking=True)
            pal_mask = batch["pal_mask"].to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                logits = model(img, palette, pal_mask)
                ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")
                loss = (ce * weight).sum() / weight.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1

            running_loss.append(float(loss.detach()))
            running_acc.append(float((logits.detach().argmax(1) == target).float().mean()))

            if step % 100 == 0:
                log(
                    f"step {step}/{steps} loss {np.mean(running_loss):.4f} "
                    f"px-acc {np.mean(running_acc):.4f} ({time.perf_counter()-t0:.0f}s)"
                )
                running_loss, running_acc = [], []

            if step % val_every == 0 or step == steps:
                val = evaluate_model(model, ds_val, device)
                log(f"  val @{step}: {val}")
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({"step": step, **val}) + "\n")
                ckpt = {"model": model.state_dict(), "config": config, "step": step, "val": val}
                torch.save(ckpt, outdir / "last.ckpt")
                if val["tolerant"] > best_tolerant:
                    best_tolerant = val["tolerant"]
                    torch.save(ckpt, outdir / "best.ckpt")

    final = {"best_tolerant": best_tolerant, "steps": steps, "params_m": round(n_params / 1e6, 2)}
    (outdir / "summary.json").write_text(json.dumps(final, indent=2))
    return final
