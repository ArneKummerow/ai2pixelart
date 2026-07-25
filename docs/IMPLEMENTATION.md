# Implementation

## Module layout

| Module | Role |
|---|---|
| `grid.py` | Grid pitch/phase estimation (FFT candidates + Rayleigh scoring, joint square-pair selection, champion veto), boundary refinement, `regrain` (granularity) |
| `palette.py` | Palette extraction (Lab agglomerative, coverage bins, mode centroids, absorption, dead-entry reseeding), quantization, `dedupe_palette`, hex parsing |
| `pipeline.py` | Classical cleaner: grid → cell sampling → palette → assignment → `smooth_indices` → optional `consensus_indices`; `CleanResult` |
| `metrics.py` | Ground-truth metrics: cell accuracy (exact/tolerant), palette fidelity, detail retention |
| `autoqa.py` | No-ground-truth QA (`inspect` CLI): boundary SNR, tile-wise pitch consistency, cell fit mean/p95, speckle rate, shade flicker, detail survival |
| `models.py` | Model registry (`ModelSpec`/`REGISTRY`), approach → checkpoint resolution (`resolve_checkpoint`, `discover_runs`), and checkpoint format (safetensors save/load/convert; legacy `.ckpt` still loads) |
| `corrupt.py` | Smoke-test corruptions for tests/demo — NOT the training degradation |
| `vae.py`, `pairgen.py` | VAE-roundtrip corruption; pair generation for both backends (manifest `pairs.jsonl` with exact scale/phase per pair, validity masks, fine-pitch downscaling, shading/noise augmentation) |
| `imgedit.py` | SDXL img2img corruption backend (fp16-fix VAE, edge-pad to %8 — the pipeline's own preprocessor would silently resize and break alignment; optional corruptor LoRA) |
| `rails.py` | Per-cell validity masks from exact metadata (mean cell-color drift ≤ 14 ΔE → valid); pairs < 70% valid are rejected |
| `corruptor.py` | Corruptor LoRA training: domain adaptation of the SDXL UNet prior on rail-accepted outputs + real AI images (oversampled), timesteps biased to the img2img regime, rank 16 |
| `spritegen.py` | Procedural clean sprites: palette pools from real images, near-hue ramps, 2×2 tile sheets, frames/stripes backgrounds |
| `nnmodel.py` | `PixelCleanNet`: U-Net encoder + palette-pointer head (features · embedded Lab palette keys, softmax over ≤ K_MAX=96 entries, padding masked) |
| `nndata.py` | `PairDataset` (multiple pair dirs, per-pair validity masks as loss weights, crop 88, drops smaller pairs), `make_decoys` (train-only near-miss entries), `majority_vote_cells` (zero-vote cells filled from nearest voted cell) |
| `nninfer.py` | Neural inference split into `nn_propose` (classical proposal + net logits — the expensive, cacheable half) and `nn_finalize` (leash + per-cell vote + smooth/consensus — cheap, param-sensitive); `nn_clean_image` composes both. Leash runs on the compute device (`_apply_leash`), not host numpy. Re-exports `load_checkpoint`/`resolve_device` from `models.py` |
| `train.py` | Training loop: detail-weighted per-pixel cross-entropy (isolated-detail pixels ×20), mask-weighted, periodic val, `best.ckpt`/`last.ckpt` + `summary.json`/`metrics.jsonl` |
| `webapp.py`, `viewer.html` | Workspace viewer (see [VIEWER.md](VIEWER.md)) |
| `demo.py`, `cli.py` | Synthetic end-to-end demo; click CLI |

ΔE thresholds throughout are CIE76 (Euclidean distance in CIELAB).

## Classical flow

`pipeline.clean(img, ...)`:

1. `grid.estimate_grid` → `Grid` (per-axis pitch/phase/score + refined
   edges). `granularity` applies `grid.regrain` (integer subdivide/merge
   of refined edges, cells stay ≥ 1 px).
2. Cell colors sampled at cell interiors (optional denoise).
3. `palette.extract_palette`: `max_colors` is target *and* cap — when the
   ΔE-cut yields fewer clusters, the merge tree is re-cut with
   `criterion=maxclust`; absorption radius auto-adapts to pitch (15 at
   pitch ≥ 3, 10 below); unused ("shadowed") entries are reseeded onto
   the worst-represented colors. Forced palettes (`palette=`) are deduped
   at ΔE 2 (order-preserving).
4. Assignment → `smooth_indices` (≥5/8 neighborhood majority AND the
   candidate fits the raw cell within 3 ΔE — the guard protects 1-px
   details) → optional `consensus_indices` (global flat-color consensus:
   connected components of equal index clustered by observed color;
   a component flips iff within `merge_de` of the cluster centroid or
   the fit delta is ≤ `guard_de`).

## Neural flow

`nninfer.nn_clean_image`: classical proposal (palette defaults to its
natural size — the train-time K cap is deliberately NOT enforced at
inference) → per-pixel pointer logits → optional leash (a *relative*
ΔE margin over the nearest entry, applied per-pixel via logit masking
AND per-cell from denoised cell colors) → per-cell majority vote →
smooth/consensus as above.

## Data engine & retraining

```bash
ai2pixelart data palettes my_images/ -o data/palettes.json
ai2pixelart data sprites -o data/sprites -n 5000 --palette-pool data/palettes.json
ai2pixelart data pairs data/sprites -o data/pairs_vae -n 2             # VAE roundtrip
ai2pixelart data edit-pairs data/sprites -o data/pairs_edit -n 1 \
    --downscale-frac 0.6 --shade-prob 0.5 --noise-prob 0.55 \
    --lora runs_lora/corruptor_v1                                      # SDXL img2img
ai2pixelart train model --pairs data/pairs_vae --pairs data/pairs_edit -o runs/my-run
ai2pixelart train corruptor --pairs data/pairs_edit --real my_images/ -o runs_lora/my-lora
```

Constraints that matter:

- `data pairs`/`data edit-pairs` scale floor is 3.0 (sub-3 px cells
  dissolve in the f8 VAE); the fine-pitch regime is reached via
  `--downscale-frac` (corrupt at 4.2–6.4, box-halve; metadata transform
  is exact for a box filter).
- `PairDataset` drops pairs smaller than the 88-px crop.
- `train model --decoy-max` (default 48) appends near-miss palette decoys
  to training samples only; the val split never sees them.
- Rails: `data edit-pairs` writes per-pair masks + `valid_frac` and
  rejects pairs < 70% valid.

## Models & packaging

Install tiers (`pyproject.toml` extras): base install is classical only
(numpy/scipy/scikit-image/pillow/click, no torch — every torch import is
lazy); `pip install ai2pixelart[neural]` adds torch + safetensors +
huggingface_hub to run the nets; `ai2pixelart[gen]` adds the full
data/training stack (diffusers, transformers, accelerate, peft) and
implies neural.

Checkpoints ship as **safetensors** — safe (no pickle code execution),
mmap-fast, and the Hub standard. The model config (`{"base": 48}`) and
`step`/`val` live in the safetensors metadata header, so a checkpoint is
one self-describing file. `training` still writes torch `.ckpt` (it also
holds resume state); `ai2pixelart models convert --all` rewrites every
`runs/*/best.ckpt` to a sibling `best.safetensors`, and `load_checkpoint`
reads either format.

`models.resolve_checkpoint(name)` maps an approach to a file, in order:
a direct path → `runs/<name>/best.safetensors` (preferred) or `best.ckpt`
→ a registered model downloaded from the Hub. `REGISTRY` pins `robust`
and `detail` to [arnekummerow/ai2pixelart](https://huggingface.co/arnekummerow/ai2pixelart)
at a fixed commit, each with a `sha256` verified after download. A fresh
`pip install ai2pixelart[neural]` with no local `runs/` fetches on first
use and caches. `ai2pixelart models list` shows the registry and what is
local; `ai2pixelart models download [NAMES]` pre-fetches for offline use.

To ship a new model: upload `name-vN.safetensors` to the repo, then bump
that model's `filename`/`revision`/`sha256` in `REGISTRY` (a code
release). The pinned `revision` means an accidental re-upload can't change
what existing installs download.

## Viewer server

Stdlib `ThreadingHTTPServer` (port 8412), no third-party web deps.

- `GET /` — the single-file UI (`viewer.html`)
- `GET /api/images` — gallery + presets + served neural models
  (re-scans `runs/*/best.ckpt` on every call: a finishing training run
  appears without restart)
- `GET /img/<name>` — original image bytes
- `POST /api/clean` `{image, params}` — validated against
  `webapp.PARAM_SPEC`; results cached in memory keyed by (file sha1,
  params), cache dies with the process
- `POST /api/upload` `{name, data}` — adds an image to the workspace
  (sanitized name, never overwrites; the app's only disk write)

A small **proposal cache** (`ViewerApp._proposals`, LRU of 2) holds each
neural clean's grid/palette/logits keyed by (image, model, proposal
params), so tweaking leash/smooth/consensus re-runs only `nn_finalize`.
The whole-result cache still short-circuits identical (image, params).
`▦ batch` in the UI is client-side: it POSTs `/api/clean` per workspace
image with the current params (reusing both caches) and zips the results
in-browser — no batch endpoint.

Method selection is a param (`method: classical|nn`, `model: <variant>`);
neural variants are served from `--model NAME=PATH` or auto-discovery.
Checkpoint identity (path + mtime) is part of the cache key, so an
overwritten `best.ckpt` never serves stale results.
