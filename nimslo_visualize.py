#!/usr/bin/env python3
"""
Nimslo Alignment Pipeline with Visualizations

Runs the full alignment pipeline with matplotlib visualizations,
saving all plots to files (non-interactive backend to avoid crashes).

Usage:
    python nimslo_visualize.py ./nimslo_raw/01/ -o output.gif
    python nimslo_visualize.py ./nimslo_raw/01/ -o output.gif --viz-dir ./viz_output/
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List
import logging

# Set up matplotlib with non-interactive backend BEFORE importing pyplot
# This prevents GUI-related crashes
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - saves to files only
import matplotlib.pyplot as plt

import cv2
import numpy as np

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


def visualize_masks(
    images: List,
    masks: List,
    output_path: Path,
    mask_type: str = "center"
):
    """Visualize segmentation masks overlaid on images."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Top row: original preprocessed images
    for i, (ax, img) in enumerate(zip(axes[0], images)):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Frame {i+1} (Preprocessed)")
        ax.axis('off')
    
    # Bottom row: images with masks overlaid
    for i, (ax, img, mask) in enumerate(zip(axes[1], images, masks)):
        overlay = visualize_mask_simple(img, mask, alpha=0.3, color=(0, 255, 0))
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Frame {i+1} ({mask_type} mask)")
        ax.axis('off')
    
    plt.suptitle("Subject Segmentation Results", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved mask visualization: {output_path}")


def visualize_matches(
    img1,
    img2,
    kp1,
    kp2,
    matches: List,
    output_path: Path,
    max_matches: int = 50
):
    """Visualize feature matches between two images."""
    if len(matches) == 0:
        logger.warning("  No matches to visualize")
        return
    
    sorted_matches = sorted(matches, key=lambda x: x.distance)[:max_matches]
    
    img_matches = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        sorted_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f"Feature Matches Between Frame 1 and Frame 2\n(showing best {len(sorted_matches)} of {len(matches)})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved matches visualization: {output_path}")


def visualize_aligned(
    aligned_images: List,
    output_path: Path
):
    """Visualize aligned images side by side."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, (ax, img) in enumerate(zip(axes, aligned_images)):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Aligned Frame {i+1}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved aligned images visualization: {output_path}")


def visualize_mask_simple(img, mask, alpha=0.4, color=(0, 255, 0)):
    """Overlay a mask on an image with transparency."""
    overlay = img.copy()
    mask_bool = mask > 127
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + 
        alpha * np.array(color)
    ).astype(np.uint8)
    return overlay


def visualize_iou_alignment(
    aligned_images: List,
    masks: List,
    alignment_results: List,
    output_path: Path
):
    """Visualize IoU alignment quality with anaglyph-style mask overlap."""
    from nimslo_core.alignment import AlignmentResult
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Anaglyph-style mask overlap visualization
    ref_mask = masks[0]
    ref_img = aligned_images[0]
    
    # Reference frame: show reference mask only
    ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    ref_mask_vis = np.zeros_like(ref_rgb)
    ref_mask_vis[ref_mask > 127] = [255, 0, 0]  # Red for reference mask
    # Blend with original image
    ref_anaglyph = cv2.addWeighted(ref_rgb, 0.5, ref_mask_vis, 0.5, 0)
    axes[0].imshow(ref_anaglyph)
    axes[0].set_title("Frame 1\n(Reference Mask)", fontsize=12)
    axes[0].axis('off')
    
    # Other frames: anaglyph-style with current frame mask (cyan) + reference mask (red)
    for i in range(1, 4):
        ax = axes[i]
        img = aligned_images[i]
        mask = masks[i]
        result = alignment_results[i]
        
        # Warp the current frame mask to reference frame using the transformation
        transform = result.transform
        h, w = ref_img.shape[:2]
        
        # Apply transformation to current frame mask
        if transform.shape == (2, 3):
            # Affine transform
            warped_mask = cv2.warpAffine(mask, transform[:2], (w, h))
        else:
            # Homography
            warped_mask = cv2.warpPerspective(mask, transform, (w, h))
        
        # Create anaglyph-style visualization
        # Convert to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create anaglyph: red channel = reference mask, cyan channels (green+blue) = current frame mask
        anaglyph = np.zeros_like(img_rgb)
        
        # Red channel: reference mask
        anaglyph[:, :, 0] = np.where(ref_mask > 127, 255, img_rgb[:, :, 0])
        
        # Green and Blue channels: current frame mask (cyan)
        anaglyph[:, :, 1] = np.where(warped_mask > 127, 255, img_rgb[:, :, 1])
        anaglyph[:, :, 2] = np.where(warped_mask > 127, 255, img_rgb[:, :, 2])
        
        # Where both masks overlap, blend to show white/yellow
        intersection = cv2.bitwise_and(ref_mask, warped_mask)
        overlap_region = intersection > 127
        
        # In overlap regions, show white/yellow (full RGB)
        anaglyph[overlap_region] = [255, 255, 200]  # Light yellow/white for overlap
        
        # Blend with original image for better visibility
        anaglyph = cv2.addWeighted(img_rgb, 0.4, anaglyph, 0.6, 0)
        
        ax.imshow(anaglyph)
        ax.set_title(f"Frame {i+1} vs Reference\nIoU: {result.iou:.3f}", fontsize=11)
        ax.axis('off')
    
    plt.suptitle("IoU Alignment Quality (Anaglyph Style)\n(Red=Reference Mask, Cyan=Current Frame Mask, Yellow/White=Overlap)", 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved IoU alignment visualization: {output_path}")


def process_single_batch(
    batch_path: Path,
    output_path: Path,
    viz_dir: Optional[Path] = None,
    quality: str = "balanced",
    show_masks: bool = False,
    preview: bool = False,
    seg_method: str = "unet"
) -> dict:
    """
    Process a single batch of 4 images with visualizations.
    
    Args:
        batch_path: Path to directory containing 4 images
        output_path: Path for output GIF
        viz_dir: Directory for visualization outputs (default: same as output)
        quality: Quality preset ("fast", "balanced", "best")
        show_masks: Whether to save mask visualization
        preview: Whether to open result after processing
        
    Returns:
        Dictionary with processing results
    """
    from nimslo_core.preprocessing import preprocess_image, normalize_sizes
    from nimslo_core.segmentation import get_segmentation_mask, visualize_mask
    from nimslo_core.alignment import align_images, extract_features, match_features
    from nimslo_core.gif_generator import create_boomerang_gif, resize_for_web
    
    # Set up visualization directory
    if viz_dir is None:
        viz_dir = output_path.parent / f"{output_path.stem}_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
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
        if len(image_files) < 4:
            result["error"] = f"Expected 4 images, found {len(image_files)}"
            return result
        
        image_files = image_files[:4]  # Take first 4 if more exist
        logger.info(f"  Loading {len(image_files)} images...")
        
        images = []
        for f in image_files:
            img = cv2.imread(str(f))
            if img is None:
                result["error"] = f"Failed to load {f.name}"
                return result
            images.append(img)
        
        # Store originals
        original_images = normalize_sizes(images.copy())
        
        # Preprocess
        logger.info("  Preprocessing...")
        preprocessed = [preprocess_image(img, denoise=settings["denoise"]) for img in images]
        preprocessed = normalize_sizes(preprocessed)
        
        # Segment
        if seg_method == "auto":
            logger.info("  Segmenting subjects (auto method selection)...")
            masks = []
            mask_methods = []
            for i, img in enumerate(preprocessed):
                mask, conf, method = get_segmentation_mask(img)
                masks.append(mask)
                mask_methods.append(method)
                logger.info(f"    Frame {i+1}: {method} (conf: {conf:.2f})")
        else:
            logger.info(f"  Segmenting subjects with {seg_method.upper()}...")
            from nimslo_core.segmentation import segment_subject
            masks = []
            mask_methods = []
            for i, img in enumerate(preprocessed):
                try:
                    mask, conf = segment_subject(img, method=seg_method, return_confidence=True)
                    method = seg_method
                except Exception as e:
                    logger.warning(f"    Frame {i+1}: {seg_method} failed ({e}), falling back to auto method")
                    mask, conf, method = get_segmentation_mask(img)
                masks.append(mask)
                mask_methods.append(method)
                logger.info(f"    Frame {i+1}: {method} (conf: {conf:.2f})")
        
        # Visualize masks
        mask_type = mask_methods[0] if mask_methods else "unknown"
        visualize_masks(
            preprocessed, masks,
            viz_dir / "01_masks.png",
            mask_type=mask_type
        )
        
        # Center images on subjects
        logger.info("  Centering images on subjects...")
        from nimslo_core.alignment import center_images_on_subject
        centered_images, centered_masks, center_transforms = center_images_on_subject(
            preprocessed, masks
        )
        logger.info("    Images centered on subject centroids")
        
        # Extract features (needed for alignment, but don't visualize keypoints)
        logger.info("  Extracting features...")
        kp1, des1 = extract_features(centered_images[0], mask=centered_masks[0], n_features=settings["n_features"])
        kp2, des2 = extract_features(centered_images[1], mask=centered_masks[1], n_features=settings["n_features"])
        
        logger.info(f"    Frame 1: {len(kp1)} keypoints")
        logger.info(f"    Frame 2: {len(kp2)} keypoints")
        
        # Match features
        matches = match_features(des1, des2)
        logger.info(f"    Matches: {len(matches)}")
        
        visualize_matches(
            centered_images[0], centered_images[1],
            kp1, kp2, matches,
            viz_dir / "02_matches.png"
        )
        
        # Align (on centered images)
        logger.info("  Aligning frames...")
        aligned, results = align_images(
            centered_images, centered_masks,
            n_features=settings["n_features"]
        )
        
        # Log alignment results
        for i, r in enumerate(results):
            if i > 0:  # Skip reference
                logger.info(f"    Frame {i+1}: {r.total_matches} matches, {r.inliers} inliers, IoU: {r.iou:.2f}")
        
        # Visualize IoU alignment quality
        visualize_iou_alignment(
            aligned, centered_masks, results,
            viz_dir / "03_iou_alignment.png"
        )
        
        # Apply transformations to original images for final output
        # Compose centering transform with alignment transform
        logger.info("  Applying transformations to original images...")
        from nimslo_core.alignment import compose_transforms
        aligned_originals = []
        ref_h, ref_w = original_images[0].shape[:2]
        
        for i, (orig_img, result_obj, center_trans) in enumerate(zip(original_images, results, center_transforms)):
            if i == 0:
                # Reference: only apply centering transform (center_trans is always 3x3)
                aligned_originals.append(cv2.warpPerspective(orig_img, center_trans, (ref_w, ref_h)))
            else:
                # Compose: center transform first, then alignment transform
                # alignment transform is relative to centered images, so:
                # final = alignment @ center (apply center first, then alignment)
                composed_transform = compose_transforms(center_trans, result_obj.transform)
                aligned_originals.append(cv2.warpPerspective(orig_img, composed_transform, (ref_w, ref_h)))
        
        # Crop to remove vertical black bars from horizontal translation
        logger.info("  Cropping to remove black bars...")
        from nimslo_core.gif_generator import _crop_to_valid_region
        aligned_originals = _crop_to_valid_region(aligned_originals, threshold=15, margin=10)
        logger.info(f"    Cropped to {aligned_originals[0].shape[1]}x{aligned_originals[0].shape[0]}")
        
        # Visualize aligned original images (after cropping)
        visualize_aligned(
            aligned_originals,
            viz_dir / "04_aligned_originals.png"
        )
        
        # Generate GIF
        logger.info("  Generating GIF...")
        web_images = resize_for_web(aligned_originals, max_dimension=settings["max_dimension"])
        gif_path = create_boomerang_gif(
            web_images, 
            output_path,
            crop_valid_region=True,
            normalize_brightness=True,
            brightness_strength=0.5
        )
        
        result["success"] = True
        result["output_path"] = gif_path
        result["size_kb"] = gif_path.stat().st_size / 1024
        result["avg_iou"] = np.mean([r.iou for r in results[1:]])
        result["viz_dir"] = viz_dir
        
        logger.info(f"  ✓ Saved: {gif_path} ({result['size_kb']:.1f} KB)")
        logger.info(f"  ✓ Visualizations saved to: {viz_dir}")
        
        # Preview if requested
        if preview:
            import subprocess
            subprocess.run(["open", str(gif_path)], check=False)
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Align Nimslo 4-lens camera images with visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./nimslo_raw/01/ -o my_photo.gif
  %(prog)s ./nimslo_raw/01/ -o my_photo.gif --viz-dir ./viz/
  %(prog)s ./batch/ -q best --preview
        """
    )
    
    parser.add_argument(
        "input",
        type=Path,
        help="Path to batch directory (4 images)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for GIF"
    )
    
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=None,
        help="Directory for visualization outputs (default: {output_stem}_viz/)"
    )
    
    parser.add_argument(
        "-q", "--quality",
        choices=["fast", "balanced", "best"],
        default="balanced",
        help="Quality preset (default: balanced)"
    )
    
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Open result after processing"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--seg-method",
        choices=["unet", "u2net", "depth", "opencv_dnn", "saliency", "grabcut", "auto"],
        default="unet",
        help="Segmentation method to use (default: unet)"
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
    
    # Single batch mode
    output_path = args.output
    if output_path is None:
        output_path = args.input.parent / f"{args.input.name}_aligned.gif"
    
    if output_path.is_dir():
        output_path = output_path / f"{args.input.name}.gif"
    
    logger.info(f"Processing {args.input.name}...")
    result = process_single_batch(
        args.input,
        output_path,
        viz_dir=args.viz_dir,
        quality=args.quality,
        preview=args.preview,
        seg_method=args.seg_method
    )
    
    if not result["success"]:
        logger.error(f"Failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()

