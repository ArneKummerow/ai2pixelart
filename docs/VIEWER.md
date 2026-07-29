# Workspace viewer

```bash
ai2pixelart viewer my_images/                # http://127.0.0.1:8412
ai2pixelart viewer my_images/ --port 9000
ai2pixelart viewer my_images/ --model myrun=path/to/best.ckpt   # explicit models
ai2pixelart viewer my_images/ --device cpu   # force CPU (default: GPU if available)
```

Serves a folder of images as an interactive workspace. Nothing is
precomputed and nothing is written to disk (except explicit uploads):
every cleaned image is computed on demand as you move the parameters,
debounced while you drag, and cached in memory keyed by (input file
content, params) — the cache dies with the process. Stale responses from
superseded runs are dropped. UI state (selected image, params, layout)
persists in the browser's localStorage.

Without `--model`, every `runs/*/best.ckpt` is auto-discovered and
re-scanned while the server lives — a finishing training run appears as a
new preset without restart or reload (the client re-syncs on window
focus).

## Layout

Three columns, both sidebars collapsible (fold buttons) and resizable
(drag the divider strips):

- **Left — gallery.** Click to select the working image. Drag-and-drop
  image files to add them to the workspace (the app's only disk write;
  names are sanitized and never overwrite).
- **Center — main view.** Single output view, or side-by-side
  original | output (toggle button; a corner badge names each pane).
- **Right — parameters** (top) **and run stats** (bottom).

## Parameters

Parameter names are user-facing; the technical name (as used in the API
and code) is in parentheses and shown in each label's hover tooltip. The
**? help** button in the top bar opens an in-app explanation of the
approaches and every parameter.

- **Approach**: *Simple* (classical), one *Neural <Name>* entry per
  served checkpoint, and *Custom* — touching any parameter switches to
  Custom automatically; approaches are starting points, the params are
  the real interface.
- **Output detail** (`granularity`, ⅓×–4×): output resolution *relative
  to* the detected grid — integer subdivide/merge of the refined cell
  edges, for when the estimated grid is coarser than the detail you want
  to keep.
- **Color merging** (`merge_de`): ΔE below which shades merge into one
  palette color.
- **Cell sampling** (`keep_frac`): fraction of each cell's interior read
  for its color.
- **Absorb amount / absorb range** (`absorb_frac` / `absorb_de`): rare
  shades below the *amount* share may be swallowed by a stronger color
  within *range* ΔE; range is auto by default (pitch-adaptive).
- **Pixel size** (`pitch`): auto-checkbox; uncheck to override the grid
  pitch (the fix for sub-2 px grids).
- **Colors** (`max_colors`): target as well as cap — values above the
  natural palette size split the merge tree back up to the exact count
  when the image has that many distinguishable colors. *Auto* uses the
  natural size.
- **Fixed palette** (`palette`): explicit output palette as
  `#rrggbb,#rrggbb,...` (up to 256 entries, applies to both methods;
  deduped at ΔE 2 server-side). The **⬆** button next to the field
  imports one from a file — see [Palettes](#palettes) below.
- **Denoise / De-speckle / Unify areas** (`denoise` / `smooth` /
  `consensus`): cell-color denoising, detail-guarded speckle removal,
  and global flat-color consensus (forces near-identical flat regions to
  the same palette entry; default off — toggle to A/B).
- **Limit recolor** (`leash`, neural only): restrict each cell to palette
  entries within a ΔE margin of the observed color. Guidance: 2–4 with
  dense forced palettes, 8 (default) with auto palettes.

## Interactions

- **Space** — flip output ↔ original, always — except while a text/number
  input is focused; **Enter** in any input commits and blurs, so Space
  works immediately after any confirmation. Buttons/selects blur on use.
- **Wheel** — zoom anchored at the cursor. **Drag** — pan (panes stay in
  sync in split view). **Double-click / 0 / Fit** — refit.
  **1×/2×/4×/8×** — zoom presets; **100%** is pixel-perfect (1 image px =
  1 screen px).
- **⬇ download** — saves the current result as
  `<source>.<preset>.<WxH>.png` (client-side, no server round-trip).
- **🎨 palette popover** — exact distinct-color tally of the displayed
  image (client-side), total + top swatches with hex/share,
  click-to-copy, and **⬇ export** of the whole palette (see below).
- **? help** — in-app reference: what each approach does, every
  parameter's effect, and how to read the quality signals.
- **▦ batch** — clean every image in the workspace with the current
  parameters, showing per-image progress and thumbnails, then offers a
  **download all (zip)** of the results. Reuses the result cache, so
  images already computed at those params are instant.

Neural param tweaks are fast to re-run: the grid, palette, and net logits
are cached per (image, model, proposal params), so changing leash /
de-speckle / unify-areas re-runs only the cheap final step rather than the
whole net (the stats' time reflects this).
- Recomputes show *in* the image: the stale output dims under a sliding
  progress bar until the new result lands. Errors turn the status line
  red.

## Palettes

Import and export are both client-side; no format ever reaches the server,
which only ever sees the `#rrggbb,...` string of the fixed-palette
parameter.

**Export** — in the 🎨 popover: pick a format, hit **⬇ export**. It writes
every distinct color of the output (the swatch grid shows at most 256 of
them, the file has all), most-used first, as
`<source>.<preset>.palette.<ext>`:

| Format | What it is |
| --- | --- |
| PNG Image (1×/8×/32×) | one square swatch per color, that many pixels a side; wraps into rows past 256 colors |
| PAL File (JASC) | `JASC-PAL` text, `r g b` per line (Paint Shop Pro, Aseprite, GrafX2) |
| Photoshop ASE | Adobe Swatch Exchange, one RGB color block per entry |
| Paint.net TXT | `;` comments plus one `AARRGGBB` per line |
| GIMP GPL | `GIMP Palette` header, `r g b<TAB>hex` per line (GIMP, Krita, Aseprite) |
| HEX File | bare `RRGGBB` per line (Lospec) |

**Import** — the **⬆** button next to *fixed palette* reads all of those
back (`.gpl`, `.pal` as JASC *or* Microsoft RIFF, `.ase`, `.txt`, `.hex`),
plus any image: an image contributes its distinct colors in scan order,
which reads back an exported PNG strip at any scale and also lifts a
palette off a reference picture. Detection is by content (`ASEF`/`RIFF`
magic, `JASC-PAL`/`GIMP Palette` header), so a mislabeled extension still
works; unknown text is scanned for `#rrggbb` / `aarrggbb` tokens, then for
`r g b` triplets. ASE entries in CMYK, Gray or Lab are converted to RGB.
Duplicates are dropped, and anything past the first 256 colors is cut with
a notice (the server's cap).

The CLI speaks the same formats — `ai2pixelart palette extract`/`convert`,
and `clean --palette <file>` / `--palette-out <file>` — so a palette moves
between the viewer, the CLI and an external editor unchanged.

## Run stats

Estimated pitch, detection z, palette size, runtime, plus two honesty
signals: the **reconstruction residual** (cleaned cells painted back onto
the input grid, ΔE vs the input — note a neural model that *corrects*
more scores worse here by design) and **cell_fit** (mean ΔE from each raw
cell to its assigned entry): when a neural preset's cell_fit is far above
Simple's, the net is recoloring — trust Simple for color fidelity there.

## API

See the server section of [IMPLEMENTATION.md](IMPLEMENTATION.md) for the
four endpoints (`/api/images`, `/img/<name>`, `/api/clean`,
`/api/upload`) and the caching contract.
