# Development log

The project went through eleven restoration-model iterations and several
data-engine generations in July 2026. This file records the path: what was
tried, what each step taught, and which ideas were measured and rejected —
so nothing gets re-tried out of ignorance. The shipped result is two
checkpoints (**Neural Robust** = iteration 11, **Neural Detail** =
iteration 9) plus the classical "Simple" pipeline they build on.

## The approach, in short

Train a restoration net on (clean pixel art → gen-AI-corrupted rendering)
pairs, inverted. Detail triage — deciding which of the AI's excess detail
survives — is learned from the pairing; the clean original is the answer
key. Alignment between clean and corrupted is *enforced* with per-cell
validity rails, not hoped for. The classical pipeline serves as baseline,
as the palette/grid proposer at inference, and as the rails infrastructure.

Design decisions made early and never reversed:

- **Gen-AI corruption over procedural degradations.** Blur/noise/resampling
  can't teach detail triage; the diffusion stack's own artifacts can.
  (Procedural corruption survives only as cheap smoke-test fixtures in
  `corrupt.py`.)
- **No diffusion for the restoration itself.** The cleanup mapping should be
  deterministic; generative sampling on the output side adds hallucination
  risk for nothing.
- **Classical stays.** It needs no GPU, wins on some image classes, and its
  `cell_fit` gap to the net is the honest "is the net recoloring?" signal.

## Classical baseline — where it peaked

The classical path (FFT+Rayleigh grid estimation, Lab agglomerative palette
extraction, per-cell voting) went through several correctness rounds worth
remembering:

- **Grid**: per-axis pitch selection was replaced by *joint* square-pair
  selection (near-best (y,x) pairs compete, harmonic families arbitrate),
  plus a "champion veto" for decorative periodicity (scanlines out-scoring
  the true art grid). Tile-based arbitration was measured and rejected —
  square-tile votes vanish at 3×3 tiling and the z margins were too thin.
  A later aspect-distortion fix: when the two axes' near-best sets share no
  scale at all (one axis's fundamental crowded just under the rel_tol cut
  by its own p/2 sub-harmonic), the independent fallback could pick
  harmonically related pitches and distort the aspect 2:1; square
  completion now rescores each candidate on the other axis first and lets
  detectable completions compete as square pairs.
- **Palette**: representatives must come from color-space *bins* (coverage),
  but each rep must be that bin's most frequent *exact* color — bin centers
  tint dominants 2–6 ΔE; global-frequency reps starve rare-but-distinct
  color families. Cluster centroids snap to the cluster's exact mode color
  when it carries ≥5% of cluster mass (flat-area dominants do; noise clouds
  don't), making dominant colors byte-exact. Dead ("shadowed") entries get
  reseeded onto the worst-represented colors.
- **Adaptive absorb radius**: fixed absorption ate detail colors on
  fine-pitch sheets; the radius is now pitch-dependent (15 at pitch ≥ 3
  where draining recovers detail, 10 below where bleed mass is too large).
- **Measured ceiling** (why effort moved to the NN): true pitch < 2 px is
  Nyquist-blind by construction; at pitch ~3.3 a 1-px white eye's best
  surviving pixel is only ~70% eye color — recovering "eyes are pure white"
  needs learned priors; noisy backgrounds split into clusters that overlap
  genuinely-distinct palette neighbors in every local statistic; and a
  single global grid can't represent per-sprite mixed pitches. These are
  structural, not bugs.

## Corruption engines (the data side)

1. **VAE roundtrip** (SDXL f8 VAE, randomized non-integer scale/phase).
   Genuine diffusion artifacts, zero content drift by construction.
   Measured: 14-color art explodes to 20–30k colors, per-color shifts
   0.7–4.4 ΔE with per-cell wobble std 2–5 (dark colors worst). Two hard
   constraints: **cells < 3 px dissolve in the f8 VAE** (scale floor 3.0)
   and border cells need ≥ 1.5 px presence. Still used for cheap volume.
2. **SDXL img2img + validity rails.** Low-strength re-rendering adds the
   generative prior (shading, AA, reinterpretation) the VAE can't produce.
   Rails compute per-cell validity from exact metadata: drift → masked out
   of the loss, added interior detail → kept as gold signal; pairs < 70%
   valid rejected. Strengths 0.15–0.45 keep 97–100% of cells valid.
   Gotcha: the pipeline's preprocessor silently resizes to %8 — edge-pad
   first or alignment dies. Text encoders stay on CPU unless the pipeline
   is `.to(device)` *before* `encode_prompt`.
3. **The fine-pitch problem.** The ~2 px-cell regime of real AI art is
   *inaccessible* to latent-space corruption (scale floor 3.0), so no
   amount of legal-scale pairs teaches pitch-2 statistics. Fix: corrupt at
   4.2–6.4 px, then 2× box-downscale to 2.1–3.2 px; the metadata transform
   (scale/2, phase/2) is exact for a box filter. This unlocked the single
   biggest quality jump (iteration 5).
4. **Corruptor LoRA** (domain adaptation, not paired distillation):
   finetune the editor's denoising prior on rail-accepted outputs + real
   AI images oversampled, timesteps biased to the img2img regime. Validity
   at strength 0.5 rose 70→82%, at 0.6 56→77%; corruption at 0.5–0.6 looks
   like genuinely harsher AI renders. Validated at scale: 91.5% acceptance
   generating at strengths 0.35–0.6.
5. **FLUX.1 Kontext dev — tried, shelved.** On Turing cards (no native
   bf16) it ran 444–696 s *per image* and recomposed the input
   (zoom/crop, 38% cell validity) despite a faithful-re-render
   instruction. The code branch was removed in the public cleanup (it
   lives in git history); see Future work.

## Restoration model iterations

The model itself (3.5M-param U-Net + palette-pointer head: per-pixel
features dot-producted against embedded Lab palette keys, so off-palette
colors are structurally impossible; true resolution via per-cell majority
vote) barely changed after iteration 2 — **almost all progress came from
the training data**.

| # | Change (single lever where possible) | Result / lesson |
|---|---|---|
| 1 | Uniform cross-entropy, VAE pairs | High cell accuracy, but rare 1-px details traded away (.853 detail) — uniform CE optimizes the wrong thing. |
| 2 | Detail-weighted CE (isolated details ×8) | .997 exact / .905 detail — beats classical on both. Weighting rare pixels is the cheapest detail lever. |
| "vae" | Enriched procedural sprites (close darks, hue ramps) | Fixed dark-entry confusion; proved palette *distribution* drives discrimination. |
| 3 | Real-image palette pools + near-hue area pairs + scales→20 | Fixed color hallucination on real images (cream eyeballs, wrong skin). v2 collapsed on the new val (.874) — out-of-distribution, not capacity. A K=12 probe confirmed K was never the lever. |
| 4 | First SDXL img2img pairs | Rails validated; real-image fits improve on coarse art, but the fine-pitch gap did **not** close → led to the downscale insight. |
| 5 | Downscaled fine-pitch pairs (2.1–3.2 px) | Best across the board; fine-pitch regime finally trainable. |
| 6 | Detail weight 8→20 (single variable) | Fine detail +8 pp *and* fine exact up; first classical-parity fit on a real probe image. |
| 7 | 2× data volume | Fine detail +6.4 pp, no tradeoff — the recipe scales. Also: built a *fresh* held-out benchmark after realizing fixed eval subsets leak into differently-split training runs. |
| 8 | Shading-field augmentation (gradients/vignettes, targets stay flat) | Real AI backgrounds are not flat (~14-RGB vignettes); the net used to faithfully band them. Fixed the background-collapse behavior. Synthetic benchmarks did *not* discriminate this — only a real-image probe did. Small synthetic dip = dilution (more data, same steps). |
| 9 | Consolidation: same corpus, 30k steps | Best of its era; shipped as **Neural Detail**. Still the best synthetic fine-pitch detail score. |
| 10 | Hard-pair round (LoRA corruption, strengths .35–.6, one shot) | **Regression.** Harsh pairs at 13% of data taught aggressive repainting (a white-eye uniformity probe dropped 136/136 → 89/136). Lesson: corruption difficulty needs a curriculum, not a cliff; also dilution again. |
| 11 | Dense-palette decoys (K_MAX 16→96, train-time only), 2×2 tile-sheet sprites, post-corruption noise (σ 3–14) | Current **Neural Robust**. Targeted a real user asset profile (256-color forced palettes, noise-std-16 backgrounds, tile sheets). Halved the far-cell hallucination class *without* leashing; best-ever real-image fits; probes stay perfect. Only synthetic fine detail dipped (dilution signature — consolidation is future work). |

## Inference-side lessons

- The pointer head is palette-size-agnostic: the train-time K cap must not
  be enforced at inference. Uncapping it *improved* color-rich sheets
  (a capped 16-entry palette scored fit 10.1 vs 7.8 uncapped) — richer
  palettes let imperfect discrimination land closer.
- **Leash** (restrict entries to within a ΔE margin of the observed color):
  a *relative* margin over the nearest entry works; an absolute radius is
  useless on dense palettes. Cell-level leashing (from denoised cell
  colors) is what fixes background mottle; per-pixel alone doesn't.
  Guidance: leash 2–4 on dense forced palettes, 8 on auto palettes.
- **Palette dedup + global consensus**: user palettes repeat whole ramps
  1–2 ΔE apart, making entry choice a noise-decided coin flip, and every
  net/vote decision is local — far-apart same-color areas decide
  independently. Deduping forced palettes (ΔE 2) plus a global flat-color
  consensus pass (cluster equal-index components by observed color, force
  each cluster to its mass-weighted winner) fixed 49%-dominance mottle to
  96–100% without touching real artwork variation.
- Majority vote must handle zero-vote cells (sub-pixel cells at high
  granularity silently became palette entry 0; nearest-voted-cell fill).
- The viewer's residual ΔE measures fidelity to the *corrupted* input — a
  net that corrects more scores worse there by design. `cell_fit` vs
  classical is the honest recoloring alarm.

## Dead ends (measured, rejected — don't re-try without new evidence)

- Procedural degradations as training corruption (motivating decision).
- Tile-based grid arbitration (votes vanish at 3×3, thin margins).
- Palette reps from bin centers (tints) or global frequency (starves rare
  families).
- Shrinking K to fix color hallucination (distribution, not K).
- Absolute-radius leash on dense palettes.
- One-shot high-strength corruption corpora (iteration 10).
- Fixed eval subsets reused across runs (leakage; use the seed-held-out
  benchmark).
- FLUX.1 Kontext on pre-Ampere hardware.

## Future work

- **FLUX.1 Kontext / Qwen-Image-Edit corruption backends** on Ampere+
  hardware (Kontext: gated weights, needs an accepted license; Qwen:
  ~40 GB fp16). The removed Kontext integration (bf16 pipeline, %16
  padding, instruction prompt) is in git history for reference.
- **Fine-pitch detail consolidation** for Neural Robust: longer training,
  upweighted fine-pitch corpora, larger crops.
- **Corruption curriculum**: ramp LoRA strengths 0.35→0.7 across training
  instead of iteration 10's one-shot hard corpus; LoRA round 2 retrained
  on the accepted hard pairs.
- **VLM-captioned prompts** for the img2img corruptor.
- **Real pixel-art collections** (Lospec, OpenGameArt, Kenney, itch.io) as
  clean sources, so classical-baseline biases don't leak into ground truth
  via bootstrapped cleans.
- **Palette extension**: when a forced palette simply lacks a region's
  color (nearest entry 15+ ΔE), no assignment fixes it — propose adding
  entries instead.
- **Mixed per-sprite grids**: the single-global-grid assumption is the
  remaining structural limit; tile-wise pitch consistency (`inspect`)
  already measures it.
