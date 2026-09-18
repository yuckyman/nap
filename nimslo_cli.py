#!/usr/bin/env python3
"""
Nimslo Image Aligner CLI

Command-line tool for aligning Nimslo 4-lens camera images
and generating boomerang GIFs.

Usage:
    nimslo-align ./nimslo_raw/01/ -o output.gif
    nimslo-align ./nimslo_raw/ --batch -o ./outputs/
"""

import sys
import argparse
import os
from pathlib import Path
from typing import Optional, List
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

RANGE_PATH_VARIABLES = (
    "NAP_INPUT_DIR",
    "NAP_GIF_OUTPUT_DIR",
    "NAP_MP4_OUTPUT_DIR",
)


def setup_path():
    """Add the code directory to the path for imports."""
    code_dir = Path(__file__).parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


def load_dotenv(path: Optional[Path] = None) -> None:
    """Load simple KEY=VALUE settings without overriding the process environment."""
    env_path = path or Path(__file__).with_name(".env")
    if not env_path.is_file():
        return

    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not separator or not key.isidentifier():
            raise ValueError(f"Invalid .env entry on line {line_number}: {raw_line}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def configured_range_paths() -> tuple[Path, Path, Path]:
    """Resolve the input, GIF, and MP4 directories used by range mode."""
    missing = [name for name in RANGE_PATH_VARIABLES if not os.environ.get(name)]
    if missing:
        names = ", ".join(missing)
        raise ValueError(
            f"Range mode requires {names}. Copy .env.example to .env and configure it."
        )

    return tuple(
        Path(os.path.expandvars(os.environ[name])).expanduser()
        for name in RANGE_PATH_VARIABLES
    )


def resolve_numbered_batch(input_dir: Path, number: int) -> Path:
    """Resolve a numeric batch while supporting padded and unpadded folders."""
    candidates = [
        input_dir / str(number),
        input_dir / f"{number:02d}",
        input_dir / f"{number:03d}",
    ]
    return next((path for path in candidates if path.is_dir()), candidates[0])


def process_single_batch(
    batch_path: Path,
    output_path: Path,
    output_format: str,
    quality: str = "best",
    show_masks: bool = False,
    interactive: bool = False,
    preview: bool = False,
    mp4_loops: Optional[int] = None,
    mp4_output_path: Optional[Path] = None,
) -> dict:
    """
    Process a single batch of 4 images.
    
    Args:
        batch_path: Path to directory containing 4 images
        output_path: Path for output GIF
        quality: Quality preset ("fast", "balanced", "best")
        show_masks: Whether to save mask visualization
        interactive: Whether to select and review subject anchors in the terminal
        preview: Whether to open result after processing
        
    Returns:
        Dictionary with processing results
    """
    import cv2
    import numpy as np
    from nimslo_core.preprocessing import preprocess_image, normalize_sizes
    from nimslo_core.segmentation import FAST_SEGMENTATION, segment_images
    from nimslo_core.alignment import align_images, center_images_on_subject
    from nimslo_core.gif_generator import make_boomerang_frames, encode_gif, encode_mp4, resize_for_web
    
    # Quality presets
    quality_settings = {
        "fast": {"n_features": 500, "max_dimension": 400, "denoise": False},
        "balanced": {"n_features": 1000, "max_dimension": 600, "denoise": True},
        "best": {"n_features": 2000, "max_dimension": 800, "denoise": True},
    }
    settings = quality_settings.get(quality, quality_settings["balanced"])
    
    result = {
        "batch": batch_path.name,
        "success": False,
        "output_path": None,
        "error": None
    }
    
    try:
        # Find and load source scans. Keep ordering stable across all exports.
        image_files = sorted(
            f for f in batch_path.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
        ) if batch_path.is_dir() else []
        if len(image_files) < 3:
            result["error"] = f"Need at least 3 images, found {len(image_files)}"
            return result
        
        # Use 4 images if available, fallback to 3 (mechanical/dev issues)
        if len(image_files) >= 4:
            image_files = image_files[:4]
        else:
            image_files = image_files[:3]
            logger.info("  ⚠ Only 3 images found - using 1→2→3→2 boomerang")
        logger.info(f"  Loading {len(image_files)} images...")
        
        images_original = []
        for f in image_files:
            img = cv2.imread(str(f))
            if img is None:
                result["error"] = f"Failed to load {f.name}"
                return result
            images_original.append(img)
        
        # Normalize originals first so any later warps match shapes
        images_original = normalize_sizes(images_original)
        
        # Preprocess
        logger.info("  Preprocessing...")
        # Preprocess (used for segmentation + robust alignment), but keep originals for final render
        preprocessed = [preprocess_image(img, denoise=settings["denoise"]) for img in images_original]
        preprocessed = normalize_sizes(preprocessed)
        
        subject_centers = None
        if interactive:
            from nimslo_core.terminal_picker import (
                create_subject_roi_masks,
                select_subject_points,
            )

            logger.info("  Select a subject in the terminal...")
            subject_centers = select_subject_points(images_original)
            masks = create_subject_roi_masks(preprocessed, subject_centers)
            for i, (x, y) in enumerate(subject_centers):
                logger.info(f"    Frame {i+1}: anchor ({x:.1f}, {y:.1f})")
        else:
            # Segment (batched, downscaled, coreml on mac)
            logger.info("  Segmenting subjects...")
            seg_results = segment_images(preprocessed, FAST_SEGMENTATION)
            masks = []
            for i, (mask, conf) in enumerate(seg_results):
                masks.append(mask)
                logger.info(f"    Frame {i+1}: {FAST_SEGMENTATION.model} (conf: {conf:.2f})")
        
        # Save mask visualization if requested
        if show_masks:
            mask_output = output_path.parent / f"{output_path.stem}_masks.jpg"
            # Simple mask overlay visualization
            overlays = []
            for img, mask in zip(preprocessed, masks):
                overlay = img.copy()
                mask_bool = mask > 127
                overlay[mask_bool] = (0.7 * overlay[mask_bool] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)
                overlays.append(overlay)
            combined = np.hstack(overlays)
            # Resize for reasonable file size
            h, w = combined.shape[:2]
            if w > 2000:
                scale = 2000 / w
                combined = cv2.resize(combined, (int(w * scale), int(h * scale)))
            cv2.imwrite(str(mask_output), combined)
            logger.info(f"  Saved masks to: {mask_output}")
        
        # Center images on subjects
        logger.info("  Centering images on subjects...")
        centered_images, centered_masks, center_transforms = center_images_on_subject(
            preprocessed,
            masks,
            subject_centers=subject_centers,
        )
        
        # Align (on centered images)
        logger.info("  Aligning frames...")
        _, results = align_images(
            centered_images, centered_masks,
            n_features=settings["n_features"],
        )
        
        # Log alignment results
        for i, r in enumerate(results):
            if i > 0:  # Skip reference
                logger.info(
                    f"    Frame {i+1}: {r.total_matches} matches, {r.inliers} inliers, "
                    f"IoU: {r.iou:.2f}"
                )
        
        # Apply the *same* transforms to the original (non-denoised) scans to preserve film grain.
        h, w = images_original[0].shape[:2]
        aligned_originals = []
        for img_orig, center_T, align_r in zip(images_original, center_transforms, results):
            combined_T = align_r.transform @ center_T
            warped = cv2.warpPerspective(img_orig, combined_T, (w, h))
            aligned_originals.append(warped)
        
        # Build both exports from the same aligned originals. This keeps GIF and
        # MP4 settings consistent while avoiding a second expensive alignment run.
        logger.info("  Building boomerang frames...")
        gif_frames = make_boomerang_frames(
            aligned_originals,
            crop_valid_region=True,
            normalize_brightness=True,
            brightness_strength=0.5,
            end_on_first=False,
            force_even_dimensions=False,
        )

        output_paths = {}
        if output_format in {"gif", "both"}:
            logger.info("  Generating GIF...")
            web_frames = resize_for_web(gif_frames, max_dimension=settings["max_dimension"])
            gif_path = output_path.with_suffix(".gif")
            output_paths["gif"] = encode_gif(web_frames, gif_path)

        if output_format in {"mp4", "both"}:
            logger.info("  Generating MP4...")
            # MP4 ends on frame 1 for seamless concatenation. Rebuild only the
            # frame order; alignment, crop, and brightness settings are shared.
            mp4_frames = make_boomerang_frames(
                aligned_originals,
                crop_valid_region=True,
                normalize_brightness=True,
                brightness_strength=0.5,
                end_on_first=True,
                force_even_dimensions=True,
            )
            mp4_path = (mp4_output_path or output_path).with_suffix(".mp4")
            output_paths["mp4"] = encode_mp4(
                mp4_frames,
                mp4_path,
                loops=mp4_loops if mp4_loops is not None else 1,
            )

        if not output_paths:
            raise ValueError(f"Unsupported format: {output_format}")

        result["success"] = True
        result["output_path"] = output_paths.get("gif") or output_paths.get("mp4")
        result["output_paths"] = output_paths
        result["size_kb"] = sum(path.stat().st_size for path in output_paths.values()) / 1024
        result["avg_iou"] = np.mean([r.iou for r in results[1:]])

        for kind, path in output_paths.items():
            logger.info(f"  ✓ Saved {kind.upper()}: {path} ({path.stat().st_size / 1024:.1f} KB)")
        
        # Preview if requested
        if preview:
            import subprocess
            subprocess.run(["open", str(output_path)], check=False)
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"  ✗ Error: {e}")
    
    return result


def process_batch_range(
    start: int,
    end: int,
    input_dir: Path,
    gif_output_dir: Path,
    mp4_output_dir: Path,
    quality: str = "best",
    show_masks: bool = False,
    mp4_loops: Optional[int] = None,
) -> List[dict]:
    """Generate one GIF and one MP4 for every numbered batch in an inclusive range."""
    if start < 0 or end < 0:
        raise ValueError("Batch numbers must be non-negative")
    if start > end:
        raise ValueError("The first batch number must be less than or equal to the second")

    gif_output_dir.mkdir(parents=True, exist_ok=True)
    mp4_output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for number in range(start, end + 1):
        batch_path = resolve_numbered_batch(input_dir, number)
        # Preserve the user's numeric spelling for output names (01 stays 01).
        name = batch_path.name if batch_path.is_dir() else str(number)
        logger.info(f"\n[{number - start + 1}/{end - start + 1}] Processing batch {name}...")
        result = process_single_batch(
            batch_path,
            gif_output_dir / name,
            output_format="both",
            quality=quality,
            show_masks=show_masks,
            mp4_loops=mp4_loops,
            mp4_output_path=mp4_output_dir / name,
        )
        result["batch_number"] = number
        if not batch_path.is_dir():
            result["error"] = f"Batch directory not found: {batch_path}"
            result["success"] = False
        results.append(result)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    logger.info(f"\n{'=' * 50}\nRange processing complete\n{'=' * 50}")
    logger.info(f"Successful: {len(successful)}/{len(results)}")
    if failed:
        logger.info("Failed batches:")
        for result in failed:
            logger.info(f"  - {result.get('batch', result.get('batch_number'))}: {result['error']}")
    return results


def process_batch_directory(
    input_dir: Path,
    output_dir: Path,
    output_format: str,
    quality: str = "best",
    show_masks: bool = False,
    mp4_loops: Optional[int] = None,
) -> List[dict]:
    """
    Process all batch directories within input_dir.
    
    Args:
        input_dir: Parent directory containing numbered batch folders
        output_dir: Directory for output GIFs
        quality: Quality preset
        show_masks: Whether to save mask visualizations
        
    Returns:
        List of processing results
    """
    # Find batch directories (numbered folders)
    batch_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and (d.name.isdigit() or d.name.replace("-", "").isdigit())
    ])
    
    if not batch_dirs:
        logger.error(f"No batch directories found in {input_dir}")
        return []
    
    logger.info(f"Found {len(batch_dirs)} batches to process")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for i, batch_dir in enumerate(batch_dirs):
        logger.info(f"\n[{i+1}/{len(batch_dirs)}] Processing {batch_dir.name}...")
        output_path = output_dir / f"{batch_dir.name}.{output_format}"
        result = process_single_batch(
            batch_dir, output_path,
            output_format=output_format,
            quality=quality,
            show_masks=show_masks,
            mp4_loops=mp4_loops,
        )
        results.append(result)
    
    # Summary
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing Complete")
    logger.info(f"{'='*50}")
    logger.info(f"Successful: {len(successful)}/{len(results)}")
    
    if successful:
        avg_size = sum(r["size_kb"] for r in successful) / len(successful)
        avg_iou = sum(r.get("avg_iou", 0) for r in successful) / len(successful)
        logger.info(f"Average size: {avg_size:.1f} KB")
        logger.info(f"Average IoU: {avg_iou:.2f}")
    
    if failed:
        logger.info(f"\nFailed batches:")
        for r in failed:
            logger.info(f"  - {r['batch']}: {r['error']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Align Nimslo 4-lens camera images and generate boomerang GIFs or MP4s",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./nimslo_raw/01/ -o my_photo.gif
  %(prog)s ./nimslo_raw/01/ -o my_photo.mp4 --format mp4
  %(prog)s ./nimslo_raw/ --batch -o ./outputs/
  %(prog)s ./batch/ -q best --show-masks --preview
        """
    )
    
    parser.add_argument(
        "input",
        help="Batch directory, parent directory, or first batch number in range mode"
    )

    parser.add_argument(
        "end",
        type=int,
        nargs="?",
        help="Inclusive final batch number; enables `nap START END` range mode"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path (file for single, directory for batch)"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all subdirectories as batches"
    )
    
    parser.add_argument(
        "-q", "--quality",
        choices=["fast", "balanced", "best"],
        default="best",
        help="Quality preset (default: best)"
    )

    parser.add_argument(
        "--format",
        choices=["gif", "mp4"],
        default=None,
        help="Output format (gif or mp4). If omitted, inferred from output extension."
    )
    
    parser.add_argument(
        "--show-masks",
        action="store_true",
        help="Save segmentation mask visualization"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Select and review subject anchors with the mouse in a supported terminal"
    )
    
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Open result after processing (single mode only)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--loops",
        type=int,
        default=None,
        help="MP4 only: number of boomerang loops to concatenate (default: 1)"
    )

    parser.add_argument(
        "--longer",
        type=int,
        default=None,
        help="MP4 only: alias for --loops (e.g. --longer 6)"
    )
    
    args = parser.parse_args()
    
    # Set up imports
    setup_path()
    try:
        load_dotenv()
    except (OSError, ValueError) as exc:
        logger.error(f"Could not load .env: {exc}")
        sys.exit(2)

    # `nap START END` is the convenient range command. It intentionally uses
    # fixed project paths so every batch gets the same best-quality treatment.
    if args.batch and args.end is not None:
        logger.error("Use either `--batch` or range mode (`nap START END`), not both")
        sys.exit(2)
    if args.interactive and (args.batch or args.end is not None):
        logger.error("--interactive currently supports one batch at a time")
        sys.exit(2)

    numeric_interactive = args.interactive and args.end is None and args.input.isdigit()
    numeric_interactive_paths = None
    if numeric_interactive:
        if args.output is not None or args.format is not None:
            logger.error(
                "`nap --interactive NUMBER` writes both configured outputs; "
                "omit --output and --format"
            )
            sys.exit(2)
        try:
            input_root, gif_root, mp4_root = configured_range_paths()
        except ValueError as exc:
            logger.error(str(exc))
            sys.exit(2)
        batch_path = resolve_numbered_batch(input_root, int(args.input))
        numeric_interactive_paths = (
            batch_path,
            gif_root / batch_path.name,
            mp4_root / batch_path.name,
        )

    if args.end is not None:
        try:
            start = int(args.input)
        except ValueError:
            logger.error("Range mode requires numeric batch numbers: nap START END")
            sys.exit(2)

        try:
            input_root, gif_root, mp4_root = configured_range_paths()
        except ValueError as exc:
            logger.error(str(exc))
            sys.exit(2)
        if not input_root.is_dir():
            logger.error(f"Nimslo input directory does not exist: {input_root}")
            sys.exit(1)

        if args.show_masks:
            logger.warning("--show-masks is ignored in range mode to keep the output directories clean")

        try:
            results = process_batch_range(
                start,
                args.end,
                input_root,
                gif_root,
                mp4_root,
                quality="best",
                show_masks=False,
                mp4_loops=args.loops if args.loops is not None else args.longer,
            )
        except ValueError as exc:
            logger.error(str(exc))
            sys.exit(2)
        if any(not result["success"] for result in results):
            sys.exit(1)
        return

    args.input = (
        numeric_interactive_paths[0]
        if numeric_interactive_paths is not None
        else Path(args.input)
    )
    # Validate input
    if not args.input.exists():
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)
    
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    def resolve_output_format(output_path: Optional[Path]) -> str:
        if args.format:
            return args.format
        if output_path and output_path.suffix.lower() in {".gif", ".mp4"}:
            return output_path.suffix.lower().lstrip(".")
        return "gif"
    
    if args.batch:
        # Batch mode
        output_dir = args.output or args.input / "aligned_output"
        output_format = resolve_output_format(args.output)
        mp4_loops = args.loops if args.loops is not None else (args.longer if args.longer is not None else None)
        results = process_batch_directory(
            args.input,
            output_dir,
            output_format=output_format,
            quality=args.quality,
            show_masks=args.show_masks,
            mp4_loops=mp4_loops,
        )
        
        # Exit with error if any failed
        if any(not r["success"] for r in results):
            sys.exit(1)
    else:
        # Single batch mode
        if numeric_interactive_paths is not None:
            _, output_path, mp4_output_path = numeric_interactive_paths
            output_path.parent.mkdir(parents=True, exist_ok=True)
            mp4_output_path.parent.mkdir(parents=True, exist_ok=True)
            output_format = "both"
        else:
            output_path = args.output
            output_format = resolve_output_format(output_path)
            mp4_output_path = None
            if output_path is None:
                output_path = args.input.parent / f"{args.input.name}_aligned.{output_format}"
        
        if output_path.is_dir():
            output_path = output_path / f"{args.input.name}.{output_format}"
        
        logger.info(f"Processing {args.input.name}...")
        result = process_single_batch(
            args.input,
            output_path,
            output_format=output_format,
            quality=args.quality,
            show_masks=args.show_masks,
            interactive=args.interactive,
            preview=args.preview,
            mp4_loops=(args.loops if args.loops is not None else (args.longer if args.longer is not None else None)),
            mp4_output_path=mp4_output_path,
        )
        
        if not result["success"]:
            logger.error(f"Failed: {result['error']}")
            sys.exit(1)


if __name__ == "__main__":
    main()
