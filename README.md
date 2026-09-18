# nap (nimslo alignment pipeline)

a python pipeline for aligning 4-lens nimslo camera photos into smooth stereoscopic boomerang gifs. takes 4 slightly offset images and aligns them so the subject stays in place while the background shifts, creating that classic nimslo parallax effect.

## what it does

1. **preprocessing** — reduces film grain and normalizes exposure across frames
2. **segmentation** — detects the main subject using u²-net (or depth-based fallback)
3. **centering** — translates each frame so the mask centroid aligns to the reference
4. **alignment** — sift matching + affine-ransac inlier rejection + translation-only warp
5. **render** — applies transforms to the *original scans* (preserves film grain)
6. **export** — boomerang gif or mp4 with crop + brightness normalization

## architecture

```
┌─────────────┐   ┌──────────────┐   ┌─────────────────┐
│ 4 scans     │──▶│ preprocess   │──▶│ u²-net segment  │
│ (jpg)       │   │ denoise/exp  │   │ + mask centroid │
└─────────────┘   └──────────────┘   └────────┬────────┘
                                               │
                    ┌──────────────────────────▼──────────────────────────┐
                    │              per-frame pair (→ ref frame 1)         │
                    │  sift (masked) → flann + lowe 0.75                   │
                    │  → affine_partial ransac (inliers only)               │
                    │  → centroid translation fit (0° rotation)           │
                    └──────────────────────────┬──────────────────────────┘
                                               │
                    ┌──────────────────────────▼──────────────────────────┐
                    │  warp originals → boomerang → gif/mp4                   │
                    └───────────────────────────────────────────────────────┘
```

### alignment design

the final warp is **always translation-only** — nimslo lenses are fixed horizontally and rotation would break the stereo effect. the trick is using a **looser ransac model for inlier voting** (affine partial: translation + rotation + uniform scale) while **discarding** the rotation/scale from the estimated transform. this handles depth parallax during correspondence filtering without rotating the output.

confidence score: `0.5 × inlier_ratio + 0.5 × mask_iou`

see `notes.md` for the aug 2026 benchmark that validated this approach.

## structure

```
nap/
├── nimslo_cli.py              # main cli
├── nimslo_visualize.py        # pipeline + matplotlib debug output
├── notes.md                   # experiment log (tracked)
├── benchmark_output/          # gitignored — csv/gif benchmark artifacts
├── profile_output/            # gitignored — profiler csv output
├── nimslo_core/
│   ├── preprocessing.py       # film grain reduction, exposure balancing
│   ├── segmentation.py        # u²-net subject detection
│   ├── alignment.py           # sift matching, affine-ransac, translation warp
│   ├── gif_generator.py       # boomerang frame order + gif/mp4 encode
│   └── rectification.py       # stereo rectification utilities
├── notebooks/
│   └── nimslo_alignment_dashboard.py   # molab walkthrough
└── notes.md                   # development notes & experiment log
```

## quick start

### browser notebook

run the browser-oriented alignment walkthrough in molab:

[open `nimslo_alignment_dashboard.py` in molab](https://molab.marimo.io/github.com/yuckyman/nap/blob/main/notebooks/nimslo_alignment_dashboard.py)

the molab notebook uses the committed lightweight sample scans in `notebooks/web-scans/`.
keep full-resolution/private scans out of the public repo; use the local cli for those.

### single batch

```bash
python nimslo_cli.py ./nimslo_raw/01/ -o output.gif
```

### batch processing

```bash
python nimslo_cli.py ./nimslo_raw/ --batch -o ./outputs/
```

### Nimslo range shortcut (`nap START END`)

The `nap` shell alias accepts an inclusive batch-number range. It always uses the
local Nimslo source directory and the `best` preset, then writes both formats
from the same alignment run:

```bash
nap 20 137
```

Outputs are written as:

- GIF: `~/path/to/wigglegrams/20.gif`
- MP4: `~/path/to/wigglegrams/output_mp4/20.mp4`

Replace `20` with each batch number in the requested range. Existing output
files are overwritten. Use `--longer N` if the MP4 should repeat its boomerang
sequence more than once.

### with visualizations

```bash
python nimslo_visualize.py ./nimslo_raw/01/ -o output.gif --viz-dir ./viz/
```

## usage

### cli options

```bash
python nimslo_cli.py INPUT [END] [-o OUTPUT] [OPTIONS]

positional:
  INPUT                 path to one batch, parent dir (with --batch), or first batch number
  END                   inclusive final batch number; enables `nap START END` range mode

options:
  -o, --output PATH     output path (file for single, directory for batch)
  --batch               process all subdirectories as batches

range mode:
  `nap START END` always uses best quality and writes both a GIF and MP4 for each
  existing numbered batch. GIFs are written to
  `~/path/to/wigglegrams/`; MP4s are written to
  `~/path/to/wigglegrams/output_mp4/`.
  -q, --quality         quality preset: fast, balanced (default), best
  --format              output format: gif or mp4 (otherwise inferred from -o)
  --show-masks          save segmentation mask visualization
  --preview             open result after processing (single mode only)
  -v, --verbose         enable verbose output
  --loops / --longer N  mp4 only: repeat boomerang sequence N times
```

### boomerang frame order

| format | sequence | why |
|---|---|---|
| **gif** | `1→2→3→4→3→2` | loops `2→1` cleanly, no duplicate hold |
| **mp4** | `1→2→3→4→3→2→1` | ends on frame 1 for seamless concatenation |

3-frame fallback (mechanical failure): `1→2→3→2`

### mp4 defaults (high fidelity / film grain friendly)

mp4 export uses `ffmpeg` + `libx264` and is tuned to preserve grain:

- **fps**: 10
- **loops**: 1 (default). use `--longer N` (or `--loops N`) to concatenate more loops.
- **tune**: grain
- **crf**: 18
- **even dimensions**: enforced via 1px crop when needed (no resampling blur)
- **edge crop**: removes warp borders by intersecting valid regions across frames

### quality presets

| preset | sift features | max dimension | denoise |
|---|---|---|---|
| fast | 500 | 400px | no |
| balanced | 1000 | 600px | yes |
| best | 2000 | 800px | yes |

## core modules

### `preprocessing.py`

- `preprocess_image()` — denoise + exposure balance
- `normalize_sizes()` — match dimensions across frames

### `segmentation.py`

subject detection with fallback chain:

1. **u²-net** (primary) — rembg/onnxruntime
2. **depth-based** — intel dpt
3. **grabcut** — opencv refinement

exports: `get_segmentation_mask()` → `(mask, confidence, method)`

### `alignment.py`

key functions:

| function | role |
|---|---|
| `extract_features()` | sift inside subject mask |
| `match_features()` | flann + lowe ratio test (0.75) |
| `ransac_inlier_mask()` | ransac inlier rejection (production: `affine_partial`) |
| `fit_translation_from_points()` | centroid translation on inliers |
| `estimate_translation_ransac()` | ransac + translation fit (single entry point) |
| `align_pair()` / `align_images()` | full per-pair / multi-frame alignment |
| `center_images_on_subject()` | mask-centroid pre-alignment |

### `gif_generator.py`

- `make_boomerang_frames()` — forward + reverse frame sequence
- `encode_gif()` / `encode_mp4()` — export with grain-friendly settings
- `_crop_to_valid_region()` — removes black warp borders
- `_normalize_brightness()` — prevents exposure flashing between frames

## benchmarking

dev scripts (`benchmark_alignment.py`, `benchmark_optimizations.py`, `profile_pipeline.py`, `smoke_test_framing.py`) are gitignored — keep them locally for sweeps. outputs go to `benchmark_output/` and `profile_output/` (also gitignored).

compare alignment variants on real rolls:

```bash
python benchmark_alignment.py \
  --input ~/path/to/nimslo \
  --output benchmark_output \
  --write-gifs --limit 12 --stride 7
```

writes `benchmark_output/alignment_benchmark.csv` and optional gifs per variant.

### pipeline profiling

quantify where wall time goes (segmentation substeps, alignment, export):

```bash
python profile_pipeline.py ./nimslo_raw/61/
python profile_pipeline.py --input ~/path/to/nimslo --limit 5
python profile_pipeline.py ./batch/ --segmentation-only --runs 3
```

writes `profile_output/pipeline_profile.csv`.

## dependencies

see `requirements.txt`. key deps:

- `opencv-python` — image processing, sift, ransac
- `numpy<2.0` — onnxruntime compatibility
- `rembg` + `onnxruntime` — u²-net segmentation
- `pillow` — gif encoding
- `matplotlib` — visualizations (optional)
- `ffmpeg` — mp4 export

## known issues

### jupyter kernel crashes

rembg/onnxruntime causes jupyter kernel crashes on macos (openmp conflicts). use the cli instead:

```bash
python nimslo_cli.py ./nimslo_raw/01/ -o output.gif
```

### numpy 2.x incompatibility

onnxruntime isn't fully compatible with numpy 2.x. requirements constrain to `numpy<2.0`.

## development notes

see [`notes.md`](notes.md) for experiment logs, benchmark results, and troubleshooting.

## examples

```bash
# single batch, best quality, preview
python nimslo_cli.py ./nimslo_raw/01/ -o my_photo.gif -q best --preview

# batch with mask debug images
python nimslo_cli.py ./nimslo_raw/ --batch -o ./outputs/ --show-masks

# alignment debug visualizations
python nimslo_visualize.py ./nimslo_raw/01/ -o output.gif --viz-dir ./debug_viz/
```

the pipeline automatically handles:

- different image sizes (normalizes to smallest)
- exposure differences (brightness normalization)
- black borders from warping (auto-cropping)
- subject centering (mask-centroid alignment before sift)
