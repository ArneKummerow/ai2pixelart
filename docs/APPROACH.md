# Approach

The problem: AI image models produce *pseudo* pixel art — the pixel grid
wobbles and is non-integer, single "pixels" are mixtures of several
colors, and a 14-color design arrives with thousands of shades. The goal
is to recover pixel-perfect, palette-clean art at true resolution without
losing the 1-px details that carry the design.

## Two methods, one pipeline

**Classical ("Simple")** — fully deterministic, no GPU:

1. *Grid estimation*: gradient profiles are FFT-scanned for pitch
   candidates, each locally refined and scored with a Rayleigh test.
   The (y, x) pitch pair is chosen *jointly* (square-consistent pairs
   compete; harmonic families arbitrate; a strictly stronger non-harmonic
   axis champion can veto a decorative periodicity such as scanlines).
2. *Cell sampling*: each grid cell's color is a robust median of its
   interior pixels, optionally denoised.
3. *Palette extraction*: Lab-space agglomerative clustering whose
   representatives guarantee color-space *coverage* while staying exact
   image colors; cluster centroids snap to the cluster's dominant exact
   color where one exists, making flat-area colors byte-exact. Low-mass
   satellite shades are absorbed with a pitch-adaptive radius.
4. *Assignment + cleanup*: cells map to nearest palette entries; a
   detail-guarded mode filter kills flat-region speckle without touching
   isolated 1-px details.

**Neural ("Neural Robust" / "Neural Detail")** — the classical pipeline
still proposes the grid and the palette; a small U-Net (3.5M params) then
assigns each pixel a palette *entry* instead of a color. The head is a
palette pointer: per-pixel features are dot-producted against embedded
Lab palette keys, so off-palette output is structurally impossible and
one model serves any palette size. True resolution comes from a per-cell
majority vote. The net carries learned priors the classical path cannot:
it recovers "eyes are pure white" from cells that never contain pure
white, and collapses vignetted backgrounds to the one color a pixel
artist would have used.

The restoration mapping is deliberately *not* generative — no diffusion
on the output side. Cleanup should be deterministic; sampling would add
hallucination risk for nothing.

## Training data: corruption pairs with rails

The net trains on (clean pixel art → gen-AI-corrupted rendering) pairs,
inverted. Corruption comes from the diffusion stack itself, because
procedural degradations (blur/noise/resampling) cannot teach *detail
triage* — deciding which of the AI's excess details are real:

- **VAE roundtrip**: clean art encoded/decoded through the SDXL VAE at
  random non-integer scales/phases. Genuine latent artifacts, zero
  content drift by construction. Constraint: cells below ~3 px dissolve.
- **SDXL img2img**: a low-strength re-render adds the generative prior
  (soft shading, antialiasing, reinterpretation) the VAE can't produce.
  Alignment is not guaranteed, so **rails** compute per-cell validity
  from the exact pair metadata: drifted cells are masked out of the loss,
  while added interior detail is kept — that is precisely the corruption
  class the net must learn to judge.
- **Fine-pitch downscaling**: the 2–3 px cell regime of real AI art is
  unreachable by latent corruption (see above), so those pairs are
  corrupted at 4.2–6.4 px and box-downscaled 2× with an exact metadata
  transform.
- **Corruptor LoRA**: the editor is domain-adapted on its own
  rail-accepted outputs plus real AI images, which keeps high-strength
  re-renders on the fake-pixel grid — harsher, more realistic corruption
  at high acceptance rates.
- **Robustness augmentations**: near-miss palette decoys (dense-palette
  discrimination), 2×2 tile-sheet sprites, background gradients that the
  target keeps flat (gradient collapse), and post-corruption noise.

## The two variants

- **Neural Robust** — recommended default; trained with all of the above.
  Best on real AI renders, dense forced palettes, noisy backgrounds,
  tile sheets.
- **Neural Detail** — an earlier consolidation with the best 1-px detail
  retention on fine 2–3 px grids; the A/B fallback when Robust
  over-corrects.

Useful knobs on either: **leash** restricts each cell to palette entries
within a ΔE margin of the observed color (2–4 on dense forced palettes,
8 on auto palettes); **consensus** forces near-identical flat regions
across the image to the same entry (local decisions otherwise flip on
noise). A high `cell_fit` vs classical means the net is recoloring —
trust classical for color fidelity there.

## Limitations

- True pitch below ~2 px is Nyquist-blind for the grid estimator; use the
  pitch override or granularity.
- One global grid: locally varying fake-pixel pitch ("2×2 detail crammed
  into a 3×3 world") is out of scope by construction, for both methods.
- Colors genuinely missing from a forced palette cannot be assigned well
  by any method — the nearest entry may be 15+ ΔE away.
- Classical cannot recover exact colors of 1-cell details under
  resampling mix, and heavy noise makes it split flat colors; these two
  are exactly what the neural path fixes.

The measured history behind all of this — including what was tried and
rejected — is in [DEVLOG.md](DEVLOG.md).
