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
from pathlib import Path
from typing import Optional, List
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def setup_path():
    """Add the code directory to the path for imports."""
    code_dir = Path(__file__).parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


def process_single_batch(
    batch_path: Path,
    output_path: Path,
    output_format: str,
    quality: str = "best",
    show_masks: bool = False,
    preview: bool = False,
    mp4_loops: Optional[int] = None
) -> dict:
    """
    Process a single batch of 4 images.
    
    Args:
        batch_path: Path to directory containing 4 images
        output_path: Path for output GIF
        quality: Quality preset ("fast", "balanced", "best")
        show_masks: Whether to save mask visualization
        preview: Whether to open result after processing
        
    Returns:
        Dictionary with processing results
    """
    import cv2
    import numpy as np
    from nimslo_core.preprocessing import preprocess_image, normalize_sizes
    from nimslo_core.segmentation import get_segmentation_mask
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
        # Find and load images
        image_files = sorted(batch_path.glob("*.jpg")) + sorted(batch_path.glob("*.JPG"))
        if len(image_files) < 3:
            result["error"] = f"Need at least 3 images, found {len(image_files)}"
            return result
        
        # Use 4 images if available, fallback to 3 (mechanical/dev issues)
        if len(image_files) >= 4:
            image_files = image_files[:4]
        else:
            image_files = image_files[:3]
            logger.info("  ⚠ Only 3 images found - using 1→2→3→2→1 boomerang")
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
        
        # Segment
        logger.info("  Segmenting subjects...")
        masks = []
        for i, img in enumerate(preprocessed):
            mask, conf, method = get_segmentation_mask(img)
            masks.append(mask)
            logger.info(f"    Frame {i+1}: {method} (conf: {conf:.2f})")
        
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
            preprocessed, masks
        )
        
        # Align (on centered images)
        logger.info("  Aligning frames...")
        _, results = align_images(
            centered_images, centered_masks,
            n_features=settings["n_features"]
        )
        
        # Log alignment results
        for i, r in enumerate(results):
            if i > 0:  # Skip reference
                logger.info(f"    Frame {i+1}: {r.total_matches} matches, {r.inliers} inliers, IoU: {r.iou:.2f}")
        
        # Apply the *same* transforms to the original (non-denoised) scans to preserve film grain.
        h, w = images_original[0].shape[:2]
        aligned_originals = []
        for img_orig, center_T, align_r in zip(images_original, center_transforms, results):
            combined_T = align_r.transform @ center_T
            warped = cv2.warpPerspective(img_orig, combined_T, (w, h))
            aligned_originals.append(warped)
        
        # Build boomerang frames from grain-preserving aligned originals
        logger.info("  Building boomerang frames...")
        boomerang_frames = make_boomerang_frames(
            aligned_originals,
            crop_valid_region=True,
            normalize_brightness=True,  # Prevent flashing from exposure differences
            brightness_strength=0.5,  # Moderate correction (0.0-1.0)
            end_on_first=(output_format == "mp4"),
            force_even_dimensions=(output_format == "mp4")
        )
        
        if output_format == "gif":
            logger.info("  Generating GIF...")
            web_frames = resize_for_web(boomerang_frames, max_dimension=settings["max_dimension"])
            output_path = output_path.with_suffix(".gif")
            output_path = encode_gif(web_frames, output_path)
        elif output_format == "mp4":
            logger.info("  Generating MP4...")
            output_path = output_path.with_suffix(".mp4")
            # Default: make MP4 longer by repeating the boomerang sequence.
            # Use grain-preserving encoder settings by default.
            output_path = encode_mp4(boomerang_frames, output_path, loops=mp4_loops if mp4_loops is not None else 1)
        else:
            raise ValueError(f"Unsupported format: {output_format}")
        
        result["success"] = True
        result["output_path"] = output_path
        result["size_kb"] = output_path.stat().st_size / 1024
        result["avg_iou"] = np.mean([r.iou for r in results[1:]])
        
        logger.info(f"  ✓ Saved: {output_path} ({result['size_kb']:.1f} KB)")
        
        # Preview if requested
        if preview:
            import subprocess
            subprocess.run(["open", str(output_path)], check=False)
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"  ✗ Error: {e}")
    
    return result


def process_batch_directory(
    input_dir: Path,
    output_dir: Path,
    output_format: str,
    quality: str = "best",
    show_masks: bool = False,
    mp4_loops: Optional[int] = None
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
            mp4_loops=mp4_loops
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
        type=Path,
        help="Path to batch directory (4 images) or parent directory (with --batch)"
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
    
    # Validate input
    if not args.input.exists():
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Set up imports
    setup_path()
    
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
            mp4_loops=mp4_loops
        )
        
        # Exit with error if any failed
        if any(not r["success"] for r in results):
            sys.exit(1)
    else:
        # Single batch mode
        output_path = args.output
        output_format = resolve_output_format(output_path)
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
            preview=args.preview,
            mp4_loops=(args.loops if args.loops is not None else (args.longer if args.longer is not None else None))
        )
        
        if not result["success"]:
            logger.error(f"Failed: {result['error']}")
            sys.exit(1)


if __name__ == "__main__":
    main()
