<!--
This is the Hugging Face MODEL CARD. Upload it as the README.md of your HF
model repo. Replace GITHUB_URL below with your repo's URL before uploading.
-->
---
license: cc-by-nc-4.0
library_name: ai2pixelart
pipeline_tag: image-to-image
tags:
  - pixel-art
  - image-restoration
  - image-to-image
---

# ai2pixelart — model weights

Trained restoration networks for **[ai2pixelart](https://github.com/ArneKummerow/ai2pixelart)** — turning
AI-generated *pseudo* pixel art (wobbly grids, mixed-color cells, too many
shades) into pixel-perfect, palette-clean pixel art.

The networks assign each cell a palette entry from the classical
grid/palette proposal; they are structurally incapable of off-palette
colors and carry learned priors (recovering pure 1-px details, collapsing
shaded backgrounds) that no local method can.

## Models

| File | Approach | Use |
|------|----------|-----|
| `robust-v11.safetensors` | **Neural Robust** | Recommended default; best on real AI renders, dense palettes, noisy backgrounds, tile sheets. |
| `detail-v9.safetensors` | **Neural Detail** | Best 1-px detail retention on fine 2–3 px grids; fallback when Robust over-corrects. |

Each file is a self-describing safetensors checkpoint (model config in the
metadata header).

## Usage

```bash
pip install "ai2pixelart[neural]"
ai2pixelart clean input.png -o out.png --approach robust   # or: detail
```

The weights download automatically on first use and are cached locally.

## License

Licensed under **CC BY-NC 4.0**, with an important carve-out: **the pixel
art you make with these models is yours** — you may use it commercially.
The noncommercial term restricts the *models themselves* (you may not sell
them or offer their functionality as a paid product/service/API), not your
output. See `LICENSE-WEIGHTS` in this repo.

The source code is licensed separately under the PolyForm Noncommercial
License 1.0.0 (see the code repository).
