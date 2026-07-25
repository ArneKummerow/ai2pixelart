"""Model registry, checkpoint format, and resolution.

Trained nets ship as **safetensors** (safe — no pickle code execution;
mmap-fast; the Hugging Face Hub standard). The model config lives in the
safetensors metadata header, so a checkpoint is one self-describing file.
Legacy torch `.ckpt` pickles ({model, config, step, val}) still load.

Resolution order for an approach name (see `resolve_checkpoint`):
1. a direct path to a `.safetensors` / `.ckpt` file,
2. a local run: `runs/<name>/best.safetensors` (preferred) or `best.ckpt`,
3. a registered model downloaded from a hub (see `ModelSpec` — wired in a
   later step; today the registry has no repo so this raises).

This module's top level is dependency-light (dataclasses/pathlib/json) so
the classical path and the CLI can import it without torch; the format
helpers import torch/safetensors lazily.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FORMAT_TAG = "ai2pixelart/1"
# preference order when a run directory holds more than one checkpoint file
CKPT_NAMES = ("best.safetensors", "best.ckpt")


@dataclass(frozen=True)
class ModelSpec:
    """A shippable model. Hub fields stay None until a hub is configured;
    while None, the model is only usable from a local `runs/<name>/`."""

    name: str
    description: str = ""
    repo_id: str | None = None      # e.g. "you/ai2pixelart-models"
    filename: str | None = None     # e.g. "robust-v11.safetensors"
    revision: str | None = None     # tag/commit for reproducible pins
    sha256: str | None = None       # optional integrity check


# The models we maintain. Names match the run directories and the viewer
# approach labels ("Neural Robust" / "Neural Detail").
_HUB_REPO = "arnekummerow/ai2pixelart"
# pinned to a specific commit so a later re-upload can't silently change what
# existing installs download (bump alongside filename when shipping a new model)
_HUB_REV = "dbb31488de8c0cf6804ebcc69a65be908560aab1"

REGISTRY: dict[str, ModelSpec] = {
    "robust": ModelSpec(
        "robust",
        description="Recommended default; best on real AI renders, dense "
        "palettes, noisy backgrounds, and tile sheets.",
        repo_id=_HUB_REPO,
        filename="robust-v11.safetensors",
        revision=_HUB_REV,
        sha256="2217a79085e8276bfabb4653338e9dd6d614a3875cac06c65f0eec72304e63be",
    ),
    "detail": ModelSpec(
        "detail",
        description="Best 1-px detail retention on fine 2-3 px grids; the "
        "fallback when robust over-corrects.",
        repo_id=_HUB_REPO,
        filename="detail-v9.safetensors",
        revision=_HUB_REV,
        sha256="fbced2f1cefc37fbb674419f392adbb397027bc0a5112a6e5b910fa645d268f0",
    ),
}


class ModelNotAvailable(Exception):
    """An approach could not be resolved to a checkpoint file."""


# --------------------------------------------------------------------------
# discovery / resolution (torch-free)
# --------------------------------------------------------------------------

def local_checkpoint(name: str, runs_dir: str | Path = "runs") -> Path | None:
    """The best local checkpoint file for a run name, or None. Prefers
    safetensors over a legacy .ckpt when both are present."""
    d = Path(runs_dir) / name
    for fn in CKPT_NAMES:
        p = d / fn
        if p.is_file():
            return p
    return None


def discover_runs(runs_dir: str | Path) -> dict[str, Path]:
    """Map every run directory under `runs_dir` that holds a checkpoint to
    its best checkpoint path (name -> path)."""
    runs_dir = Path(runs_dir)
    out: dict[str, Path] = {}
    if runs_dir.is_dir():
        for d in sorted(runs_dir.glob("*")):
            if d.is_dir():
                p = local_checkpoint(d.name, runs_dir)
                if p is not None:
                    out[d.name] = p
    return out


def available_local(runs_dir: str | Path = "runs") -> list[str]:
    return sorted(discover_runs(runs_dir))


def resolve_checkpoint(approach: str, runs_dir: str | Path = "runs") -> Path:
    """Resolve a neural approach name to a checkpoint file.

    Raises ModelNotAvailable with a helpful message if nothing matches.
    """
    direct = Path(approach)
    if direct.is_file():
        return direct

    local = local_checkpoint(approach, runs_dir)
    if local is not None:
        return local

    spec = REGISTRY.get(approach)
    if spec is not None and spec.repo_id:
        return _download_from_hub(spec)  # configured in a later step

    avail = available_local(runs_dir)
    known = sorted(REGISTRY)
    msg = (
        f"unknown approach '{approach}': expected 'simple', a run name, or a "
        f"path to a .safetensors/.ckpt file."
    )
    if avail:
        msg += f" Available locally: {', '.join(avail)}."
    ungettable = [n for n in known if n not in avail]
    if ungettable:
        msg += (
            f" Registered but not downloadable yet (no hub configured): "
            f"{', '.join(ungettable)}."
        )
    raise ModelNotAvailable(msg)


def download_model(name: str) -> Path:
    """Force-fetch a registered model from its hub into the local cache
    (bypassing any local runs/ copy). Returns the cached file path."""
    spec = REGISTRY.get(name)
    if spec is None or not spec.repo_id:
        raise ModelNotAvailable(f"'{name}' is not a downloadable registered model")
    return _download_from_hub(spec)


def _download_from_hub(spec: ModelSpec) -> Path:
    """Fetch a registered model into the local Hugging Face cache and verify
    its integrity against the pinned sha256."""
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as e:
        raise ModelNotAvailable(
            f"model '{spec.name}' downloads from the Hugging Face Hub, but "
            f"huggingface_hub is not installed — run "
            f'`pip install "ai2pixelart[neural]"`.'
        ) from e

    path = Path(hf_hub_download(
        repo_id=spec.repo_id, filename=spec.filename, revision=spec.revision,
    ))
    if spec.sha256:
        _verify_sha256(path, spec.sha256)
    return path


def _verify_sha256(path: Path, expected: str) -> None:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != expected:
        raise ModelNotAvailable(
            f"integrity check failed for {path}: expected sha256 {expected}, "
            f"got {got}"
        )


# --------------------------------------------------------------------------
# format: save / load / convert (torch + safetensors)
# --------------------------------------------------------------------------

def save_safetensors(state_dict, config: dict, path: str | Path,
                     step: int | None = None, val: dict | None = None) -> Path:
    """Write a state dict + config to a self-describing safetensors file
    (config/step/val go into the metadata header)."""
    from safetensors.torch import save_file

    meta = {"format": FORMAT_TAG, "config": json.dumps(config)}
    if step is not None:
        meta["step"] = str(step)
    if val is not None:
        meta["val"] = json.dumps(val)
    tensors = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=meta)
    return path


def load_checkpoint(path: str | Path, device: str = "cuda"):
    """Load a checkpoint (either format) into an eval-mode PixelCleanNet."""
    import torch

    from ai2pixelart.nnmodel import PixelCleanNet

    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
        config = json.loads(meta.get("config", "{}"))
        state = load_file(str(path), device=device)
        step = int(meta["step"]) if meta.get("step") else None
    else:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        config, state, step = ckpt["config"], ckpt["model"], ckpt.get("step")

    model = PixelCleanNet(**config).to(device)
    model.load_state_dict(state)
    model.eval()
    model.ckpt_step = step  # surfaced in run info (in-training ckpts)
    return model


def resolve_device(device: str | None) -> str:
    """Resolve an inference device request. 'auto' (or None) picks 'cuda'
    when a GPU is available, else 'cpu'; an explicit 'cuda'/'cpu' is honored
    as given (so a user can force CPU even with a GPU present)."""
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"


def convert_checkpoint(src: str | Path, dst: str | Path | None = None) -> Path:
    """Convert a legacy torch `.ckpt` to `.safetensors` (config -> metadata).
    Default destination is the source with a `.safetensors` suffix."""
    import torch

    src = Path(src)
    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    dst = Path(dst) if dst else src.with_suffix(".safetensors")
    return save_safetensors(
        ckpt["model"], ckpt["config"], dst,
        step=ckpt.get("step"), val=ckpt.get("val"),
    )
