# Nimslo Image Alignment Pipeline

a python pipeline for aligning 4-lens nimslo camera photos into smooth stereoscopic boomerang gifs. takes 4 slightly offset images and aligns them so the subject stays in place while the background shifts, creating that classic nimslo parallax effect.

## what it does

1. **preprocessing** - reduces film grain and normalizes exposure across frames
2. **segmentation** - detects the main subject using u²-net (or depth-based fallback)
3. **alignment** - uses sift feature matching to align frames, centering on the subject
4. **gif generation** - creates a boomerang gif (1→2→3→4→3→2→1) with brightness normalization

## structure

```
code/
├── nimslo_cli.py          # main command-line interface
├── nimslo_visualize.py    # pipeline with matplotlib visualizations
├── nimslo_core/           # core library modules
│   ├── preprocessing.py   # film grain reduction, exposure balancing
│   ├── segmentation.py    # u²-net subject detection
│   ├── alignment.py       # sift matching, homography estimation
│   └── gif_generator.py   # boomerang gif creation
└── notes.md               # development notes & troubleshooting
```

## quick start

### single batch

```bash
python nimslo_cli.py ./nimslo_raw/01/ -o output.gif
```

### batch processing

```bash
python nimslo_cli.py ./nimslo_raw/ --batch -o ./outputs/
```

### with visualizations

```bash
python nimslo_visualize.py ./nimslo_raw/01/ -o output.gif --viz-dir ./viz/
```

## usage

### cli options

```bash
python nimslo_cli.py INPUT [-o OUTPUT] [OPTIONS]

positional:
  INPUT                 path to batch directory (4 images) or parent dir (with --batch)

options:
  -o, --output PATH     output path (file for single, directory for batch)
  --batch               process all subdirectories as batches
  -q, --quality         quality preset: fast, balanced (default), best
  --show-masks          save segmentation mask visualization
  --preview             open result after processing (single mode only)
  -v, --verbose         enable verbose output
```

### quality presets

- **fast**: 500 features, 400px max, no denoising
- **balanced**: 1000 features, 600px max, denoising enabled (default)
- **best**: 2000 features, 800px max, denoising enabled

## core modules

### `preprocessing.py`

handles image preprocessing:
- `preprocess_image()` - main preprocessing pipeline
- `denoise_film_grain()` - pyramid mean shift filtering
- `balance_exposure()` - histogram equalization
- `normalize_sizes()` - ensures all images are same dimensions

### `segmentation.py`

subject detection with multiple fallback methods:
- **u²-net** (primary) - deep learning segmentation via rembg
- **depth-based** (fallback) - intel dpt depth estimation
- **grabcut** (refinement) - opencv-based refinement

exports: `get_segmentation_mask()` - returns mask, confidence, method used

### `alignment.py`

feature matching and image alignment:
- `extract_features()` - sift/orb feature extraction
- `match_features()` - feature matching with ratio test
- `align_images()` - full alignment pipeline with iou optimization
- `center_images_on_subject()` - centers images on detected subject

uses homography estimation to warp images into alignment, optimizing for intersection-over-union (iou) of the segmented subject.

### `gif_generator.py`

boomerang gif creation:
- `create_boomerang_gif()` - main gif generation
- `resize_for_web()` - resizes images for reasonable file sizes
- `_crop_to_valid_region()` - removes black borders from warped images
- `_normalize_brightness()` - prevents flashing from exposure differences

creates the classic boomerang pattern: forward (1→2→3→4) then reverse (4→3→2→1).

## dependencies

see `requirements.txt` in parent directory. key deps:
- `opencv-python` - image processing
- `numpy<2.0` - array operations (numpy 2.x breaks onnxruntime)
- `rembg` + `onnxruntime` - u²-net segmentation
- `pillow` - image i/o
- `matplotlib` - visualizations (optional)

## known issues

### jupyter kernel crashes

⚠️ **rembg/onnxruntime causes jupyter kernel crashes** on macos due to openmp conflicts. 

**workaround**: use the cli instead of notebooks:
```bash
python nimslo_cli.py ./nimslo_raw/01/ -o output.gif
```

the cli works perfectly because it runs in a regular python process, not a jupyter kernel.

### numpy 2.x incompatibility

onnxruntime (used by rembg) isn't fully compatible with numpy 2.x yet. requirements.txt constrains to `numpy<2.0`.

## development notes

see `notes.md` for detailed troubleshooting, test results, and development history.

## examples

```bash
# process a single batch with best quality
python nimslo_cli.py ./nimslo_raw/01/ -o my_photo.gif -q best --preview

# batch process with mask visualizations
python nimslo_cli.py ./nimslo_raw/ --batch -o ./outputs/ --show-masks

# generate visualizations for debugging
python nimslo_visualize.py ./nimslo_raw/01/ -o output.gif --viz-dir ./debug_viz/
```

the pipeline automatically handles:
- different image sizes (normalizes to smallest)
- exposure differences (brightness normalization)
- black borders from warping (auto-cropping)
- subject centering (aligns on detected subject)

