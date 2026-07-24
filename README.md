# ai2pixelart

Turn AI-generated *pseudo* pixel art (wobbly grids, mixed-color cells, way too
many shades) into pixel-perfect, palette-clean pixel art — without losing the
single-pixel details that carry the design.

## Setup

```bash
conda env create -f environment.yml   # creates env "ai2pixelart"
conda activate ai2pixelart
```

Or into an existing env: `pip install -e ".[dev]"`.

## Usage

```bash
ai2pixelart clean input.png -o out.png --preview-scale 8
ai2pixelart clean input.png -o out.png --palette "#1e2234,#5ea740,#f0f0f0"
ai2pixelart eval pred.png gt.png          # metrics JSON (needs ground truth)
ai2pixelart qa examples/gemini            # no-GT quality table for real images
ai2pixelart demo                          # synthetic end-to-end demo
pytest                                    # run the test suite

# data engine (GPU):
ai2pixelart vae-roundtrip art.png -o rt.png --scale 4.3   # artifact inspection
ai2pixelart gen-pairs clean1.png clean2.png -o data/pairs -n 4

# workspace viewer: serve a folder of images, everything computed live
ai2pixelart view examples/gemini                           # http://127.0.0.1:8412
```

The viewer is a three-column workspace over a folder of images: a left
sidebar with the image gallery (click to select, drag-and-drop to add), a
right sidebar with the live parameters on top (preset incl. neural models,
granularity, sliders, auto-checkboxes, an explicit palette field for up to
256 colors) and the run stats below — both sidebars collapsible. The main
view shows the output, switchable to a side-by-side original|output split;
Space always flips original/output (except while typing in a text field —
Enter commits and returns Space to the toggle). Wheel zooms anchored at
the cursor, dragging pans, double-click refits; recomputes dim the stale
image under a sliding progress bar. The top bar holds only the ⬇ download
button (source.preset.WxH.png), the 🎨 palette popover (exact distinct
colors as swatches, click-to-copy), and the status line (red on errors). Nothing is precomputed — every cleaned image is
computed on demand as you drag the parameter sliders (debounced live runs)
and cached in memory keyed by (input file content, params), so the cache
dies with the process. The colors control is a target as well as a
cap: values above the natural palette size split the merge tree back up
(exact count whenever the image has that many distinguishable colors). A
granularity slider (⅓×–4×) controls output
resolution *relative to* the detected grid — integer subdivide/merge of the
refined cell boundaries (`grid.regrain`), capped so cells keep ≥ 1 px —
for when the estimated grid is coarser than the detail you want to keep. Space toggles every Cleaned panel between the
output and the original for flicker comparison; the Workspace panel selects
the image to work on and accepts drag-and-dropped files (the only thing the
app ever writes to disk). Info shown per run: estimated pitch, detection z,
palette size, runtime, and the reconstruction residual — cleaned cells
painted back onto the input grid and compared against the input in ΔE, an
honest fidelity signal when there is no ground truth. Named parameter
presets live in `webapp.APPROACHES` and appear in the panel preset dropdown
automatically.

## Approach

The learned pipeline is trained on (clean pixel art → gen-AI-corrupted
rendering) pairs, inverted. Detail triage — deciding which of the AI's excess
details survive — is learned from the pairing itself; the clean original is
the answer key. Corruption alignment is enforced with "rails" (structure-
preserving generation + per-cell validity masks), not hoped for.

Milestones:

1. **Classical toolkit + metrics** (done): grid pitch/phase estimation, cell
   downsampling, palette extraction/quantization, and the metric suite (cell
   accuracy, palette fidelity, **detail retention**). This is both the
   baseline and the rails infrastructure — the alignment scorer and validity
   masks for training pairs are built from exactly these pieces. Validated on
   real Gemini samples (`examples/gemini`): e.g. a 1024px piece cleans to
   crisp 200x200 at pitch 5.12 with all content intact.
2. **VAE-roundtrip pair generation** (done): encode/decode pixel art through
   a latent diffusion VAE (SDXL) at randomized non-integer scales and
   offsets. Genuine diffusion-pipeline artifacts, zero content drift by
   construction. Measured on real data: 14-color art explodes to ~20-30k
   colors (a real Gemini sample has ~9.5k), grid pitch survives exactly
   (estimator recovers 3.30/4.30/6.00), per-color shifts of 0.7-4.4 ΔE with
   per-cell wobble std 2-5 (dark colors worst) — the corruption the NN must
   invert, and beyond what palette quantization can (classical tolerant
   accuracy on VAE output: ~0.4-0.55). Two hard constraints found: cells
   below ~3 px dissolve in the f8 VAE (scale floor 3.0), and phases must
   leave every border cell >=1.5 px of presence for cell-complete pairs.
3. **Rail-guarded img2img corruption** (first backend live): low-strength
   img2img re-rendering adds the generative prior — soft shading,
   antialiasing, reinterpretation — that the VAE roundtrip cannot produce
   and real AI pseudo-pixel-art exhibits. Per-cell validity rails
   (`rails.validity_mask`, exact scale/phase metadata, no estimation)
   separate "model added detail" (gold signal, keep) from "model moved
   content" (poison, masked out of the loss); pairs below 70% valid are
   rejected. First backend: SDXL img2img (`imgedit.py`, ungated) — measured
   on pilots: strengths 0.15–0.45 keep 97–100% of cells valid and the grid
   detectable. Planned upgrades: FLUX.1 Kontext dev (gated; needs an HF
   token) and Qwen-Image-Edit, plus VLM-captioned prompts.

   ```bash
   ai2pixelart gen-edit-pairs data/sprites_v3 -o data/pairs_edit -n 1
   ai2pixelart train --pairs data/pairs_v3 --pairs data/pairs_edit -o runs/v4
   ```
4. **Corruptor fine-tuning loop**: LoRA-finetune an editing model on filtered
   pairs to be a faithful corruptor; yield goes up, raise the strength knobs,
   repeat.
5. **Restoration model** (v2 trained): U-Net with a palette-pointer
   classification head — per-pixel features dot-producted against embedded
   palette colors, so the output is structurally incapable of off-palette
   colors and one model serves any palette size ≤ 16. Output lives at input
   resolution (a "regularized" indexed image); true resolution comes from
   per-cell majority vote. Trained with **detail-weighted** per-pixel
   cross-entropy (pixels of isolated 1-px details upweighted 8x — uniform CE
   trades them away) on VAE pairs from procedural sprites
   (`gen-sprites` -> `gen-pairs` -> `train`), where targets are exact by
   construction from the pair metadata. Val results (both methods given the
   same oracles — known grid + GT palette — isolating per-cell color
   assignment under VAE corruption): classical exact 0.963 / detail
   retention 0.879; v1 uniform CE 0.990 / 0.853; **v2 weighted 0.997 /
   0.905** — an 11x cell-error reduction over classical and better detail
   survival. Inference on arbitrary images (`nn-clean`, or the viewer's
   "Neural" preset, enabled via `view --ckpt` / auto-discovered newest
   `runs/*/best.ckpt`): classical proposes palette+grid, the net assigns
   colors. Note the viewer's residual ΔE is fidelity to the *corrupted*
   input — a net that corrects more scores worse there by design; judge the
   NN on GT metrics or visually.

   Neural runs are parameterized too: `max_colors` sets the palette-proposal
   size (the pointer head accepts any palette size; >16 is extrapolation
   beyond training; unset = natural size soft-capped at 16), `palette`
   forces an explicit output palette ('#rrggbb,...' —
   also works for the classical method), and pitch/granularity pass through
   to the grid proposal. Checkpoints are served as named variants
   (`view --ckpt vae=runs/vae/best.ckpt --ckpt img2img=...`; default = every
   `runs/*/best.ckpt`, named by run directory) and each appears as a
   "Neural (name)" preset — the current corruption tracks stay selectable
   side by side as new ones (img2img, ...) arrive. The `vae` variant is
   trained on enriched procedural scenes (two hue ramps, several close darks,
   frames/stripes, min palette ΔE 4) after the v2 model proved unable to
   discriminate close dark palette entries on real images.

```bash
# v3 recipe: real-image palettes + near-hue pairs + corruption scales 3-20
ai2pixelart extract-palettes examples/gemini -o data/palettes_real.json
ai2pixelart gen-sprites -o data/sprites_v3 -n 2500 --palette-pool data/palettes_real.json
ai2pixelart gen-pairs data/sprites_v3 -o data/pairs_v3 -n 2 --scale-max 20
ai2pixelart train --pairs data/pairs_v3 -o runs/v3 --steps 12000
ai2pixelart nn-clean input.png -o out.png --ckpt runs/v3/best.ckpt
```

The v3 data recipe exists because v1/v2/vae nets hallucinated colors on real
images (recolored a yellow-green skin with the hair yellow; painted white
eyeballs cream): they had only ever seen procedural HSV palettes under VAE
corruption at scales ≤ 8. v3 draws ~half its sprite palettes verbatim from
real-image classical proposals (`extract-palettes`), generates near-hue AREA
color pairs (adjacent blobs 4-12 ΔE apart — the skin-vs-hair case), and
covers corruption scales up to 20 (the large-cell regime of 2048px art).

## Layout

- `src/ai2pixelart/grid.py` — grid pitch/phase estimation + boundary refinement
- `src/ai2pixelart/palette.py` — palette extraction (Lab agglomerative), quantization
- `src/ai2pixelart/pipeline.py` — classical cleaner: grid → cell median → quantize
- `src/ai2pixelart/metrics.py` — cell accuracy, palette fidelity, detail retention (vs ground truth)
- `src/ai2pixelart/autoqa.py` — no-ground-truth QA for real images (`ai2pixelart qa <folder>`): grid boundary SNR, tile-wise pitch consistency, palette cell-fit, speckle rate, shade flicker, detail survival
- `src/ai2pixelart/corrupt.py` — smoke-test corruptions (NOT the training degradation)
- `src/ai2pixelart/vae.py` / `pairgen.py` — VAE-roundtrip corruption + aligned pair generation (manifest: pairs.jsonl with exact scale/phase per pair)
- `src/ai2pixelart/webapp.py` / `viewer.html` — interactive workspace viewer (live compute, in-memory cache)
- `src/ai2pixelart/demo.py` / `cli.py` — demo and CLI

Note: cleaned Gemini outputs can bootstrap pair sources for machinery tests
(`examples/output/gemini/*_clean.png`), but real training data should come
from genuine pixel-art collections (Lospec, OpenGameArt, Kenney, itch.io
packs) so classical-baseline biases don't leak into ground truth.

ΔE thresholds throughout are CIE76 (Euclidean in CIELAB).

## Known baseline limitations (measured, intentional)

These are the failure modes that motivate the learned model; the metric
suite exists to quantify them:

- **Exact colors of 1-cell details under resampling mix.** At pitch ~3.3 with
  mild blur, the best surviving pixel of a 1-px white eye is only ~70% eye
  color (ΔE ≈ 23 from truth) — no local method can recover pure white from
  pixels that never contain it. The baseline preserves the detail
  *structurally* (it stays an isolated, much-lighter cell); recovering "eyes
  are pure white" needs learned priors.
- **Noise-splitting of dominant colors.** Beyond ~blur 0.35 + noise 3, the
  background's color cloud splits into clusters ~3.5 ΔE apart while genuinely
  distinct palette neighbors (outline vs bg) sit at ~5.5 ΔE — the two cases
  overlap in every local statistic (separation, spread, size). Tolerant cell
  accuracy drops off a cliff (0.89 -> 0.60). Distinguishing them needs
  density bimodality or semantics.
- **Single global grid.** Locally varying fake-pixel pitch (the "AI crams
  2x2 detail into a 3x3 world" pathology) is out of scope for the
  FFT+Rayleigh estimator by construction.
