# nap development notes

## 2026-08-27 — alignment experiment (`experiment/procrustes-alignment`)

### goal

test whether a procrustes/svd fit slot (after sift matching) improves alignment vs the existing translation-only baseline, while keeping rotation locked to zero for nimslo stereo fidelity.

### branch

- branched from `main` → `experiment/procrustes-alignment`
- dropped the earlier `codex/unicorn` work (stashed separately)

### pipeline tested

shared upstream for all variants:

1. u²-net segmentation + mask-centroid centering
2. sift feature extraction (masked)
3. flann descriptor matching + lowe ratio test (0.75)
4. ransac inlier rejection (model varies)
5. translation-only fit on inliers (centroid alignment, 0° rotation)
6. warp original scans, mask iou + confidence scoring

### variants benchmarked (12 rolls, stride 7)

| variant | ransac inlier model | fit |
|---|---|---|
| `baseline_translation` | translation (median) | median shift |
| `proc_trans` | translation | centroid |
| `proc_affine` | **affine partial** | centroid |
| `proc_homog` | homography | centroid |

canonical test set: `~/path/to/nimslo/` (84 four-image batches).

### results (mean across 12 rolls)

| variant | IoU | confidence | inliers |
|---|---|---|---|
| baseline_translation | **0.742** | 0.637 | 151 |
| proc_trans | **0.742** | 0.643 | 151 |
| **proc_affine** | 0.741 | **0.656** | 159 |
| proc_homog | 0.734 | 0.674 | 170 |

### conclusions

- **rotation is out.** nimslo lenses are fixed on a horizontal plane; any rotation in the final warp destroys the 3d parallax effect. rotation variants were tested and removed.
- **gifs looked nearly identical by IoU** but `proc_affine` subjectively minimized jitter — matches the confidence metric bump (construct validity win).
- **affine partial ransac is the sweet spot** for inlier selection: looser than translation ransac (handles depth parallax during voting) but stricter than homography (rejects bad perspective matches).
- the win is **decoupling inlier rejection from the final warp**: affine model for voting → translation-only output.
- batch **107** is a known failure case (one frame ~50% vertically offset); all methods fail equally — don't overfit.

### production decision

consolidated to a single path in `align_pair()`:

```
sift → flann/lowe → affine_partial ransac → centroid translation → warp
```

removed experiment cli flags (`--transform-model`, `--ransac-model`, `--max-rotation-deg`).

### gif boomerang fix

- **gif loop order**: `1→2→3→4→3→2` (no trailing frame 1 — avoids `1→1` pause on loop)
- **mp4 order**: `1→2→3→4→3→2→1` when concatenating loops
- `encode_gif` uses `disposal=2` + consecutive-frame dedup

### benchmark tooling

```bash
python benchmark_alignment.py \
  --input ~/path/to/nimslo \
  --output benchmark_output \
  --write-gifs --limit 12 --stride 7
```

outputs: `benchmark_output/alignment_benchmark.csv`, `benchmark_output/gifs/`

### pipeline profiler (`profile_pipeline.py`)

quantifies per-stage wall time on real batches. run:

```bash
python profile_pipeline.py ./nimslo_raw/61/
python profile_pipeline.py --input ~/path/to/nimslo --limit 5
python profile_pipeline.py ./batch/ --segmentation-only --runs 3
```

sample batch `61` (balanced quality, ~16s total):

| stage | ~time | % of total |
|---|---|---|
| preprocess | 4.5s | 28% |
| segment (u²-net) | 6.6s | 42% |
| alignment (sift) | 4.3s | 27% |
| export (boomerang etc) | <0.5s | ~2% |

within segmentation, **~82% is onnx inference**; session init ~16%, preprocess/postprocess negligible.

within alignment, **~99% is sift feature extraction**; matching/ransac/warp are cheap.

optimization targets: downscale before rembg, batch onnx runs, or lighter segmentation model.

### 2026-08-27 — segmentation optimization benchmark

ran `benchmark_optimizations.py` on batches 46, 37, 60, 70. full results in `benchmark_output/optimization_benchmark.csv`, gifs in `benchmark_output/opt_gifs/`.

| step | mean total | mean seg | mean iou | mean align_conf |
|---|---|---|---|---|
| 00 baseline | 17.9s | 5.6s | 0.772 | 0.757 |
| 01 downscale | 17.1s | 5.2s | 0.780 | 0.771 |
| 02 parallel | 13.3s | 1.4s | 0.779 | 0.767 |
| 03 coreml | 13.1s | 0.8s | 0.779 | 0.770 |
| 04 u2netp | 12.4s | 0.5s | 0.651* | 0.684 |
| **05 omp_cli** | **12.0s** | **0.2s** | 0.651* | 0.691 |
| 06a with_align | 11.9s | 0.2s | 0.651* | 0.681 |
| 06b centroid_only | 7.7s | 0.2s | 0.614 | 0.717 |

\*batch 70 failure case drags u2netp iou down; batches 46/37/60 gifs still look fine.

**production config:** `FAST_SEGMENTATION` = u2net + downscale 1024 + parallel + coreml (~13s vs 18s baseline). **do not use u2netp** — tighter/wrong masks break sift alignment on some rolls (batch 46: 44px→192px shift), which drives extra `_crop_to_valid_region` cropping. centroid-only alignment unusable.

added `reference_guard` crop (off by default): loosens AND-crop using frame-0 bbox. helps u2netp framing but on batch 60 with good u2net alignment it caused black bars (edge_black 0.04→0.50). production uses AND-crop only.

### open questions

- try `affine_partial ransac → median translation` (baseline fit + affine inliers) as a follow-up — may be redundant with centroid on affine inliers
- iou refinement pass skipped for now; revisit if failure rate climbs on hard rolls
- molab dashboard still uses its own inline alignment — could sync to production path
