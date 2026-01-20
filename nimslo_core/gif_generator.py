"""
GIF generation module for Nimslo images.

Creates smooth boomerang-style animated GIFs from aligned frames.
"""

import cv2
import numpy as np
from PIL import Image
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union


def make_boomerang_frames(
    images: List[np.ndarray],
    crop_valid_region: bool = True,
    normalize_brightness: bool = True,
    brightness_strength: float = 0.5
) -> List[np.ndarray]:
    """
    Create boomerang frame sequence from aligned images.
    
    The sequence plays forward then backward: 1→2→3→4→3→2→1
    
    Args:
        images: List of aligned BGR images
        crop_valid_region: Whether to crop to valid (non-black) region
        normalize_brightness: Whether to normalize brightness across frames to prevent flashing
        brightness_strength: Strength of brightness normalization (0.0-1.0, default 0.5)
                             0.0 = no correction, 1.0 = full correction to median
        
    Returns:
        List of BGR frames in boomerang order
    """
    if not images:
        raise ValueError("No images provided")
    
    if crop_valid_region:
        images = _crop_to_valid_region(images)
    
    if normalize_brightness:
        images = _normalize_brightness(images, strength=brightness_strength)
    
    if len(images) > 1:
        return images + images[-2:0:-1]
    return images


def encode_gif(
    frames: List[np.ndarray],
    output_path: Union[str, Path],
    duration: int = 100,
    loop: int = 0,
    optimize: bool = True
) -> Path:
    """
    Encode a list of frames into a GIF.
    
    Args:
        frames: List of BGR frames
        output_path: Path for output GIF
        duration: Duration per frame in milliseconds
        loop: Number of loops (0 = infinite)
        optimize: Whether to optimize GIF file size
        
    Returns:
        Path to created GIF
    """
    if not frames:
        raise ValueError("No frames provided")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pil_frames = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(rgb))
    
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=loop,
        optimize=optimize
    )
    
    return output_path


def encode_mp4(
    frames: List[np.ndarray],
    output_path: Union[str, Path],
    fps: Optional[float] = None,
    duration: int = 100,
    crf: int = 18,
    preset: str = "slow"
) -> Path:
    """
    Encode a list of frames into an MP4 using ffmpeg (H.264).
    
    Args:
        frames: List of BGR frames
        output_path: Path for output MP4
        fps: Frames per second (if None, derived from duration)
        duration: Duration per frame in milliseconds (used if fps is None)
        crf: H.264 quality factor (lower is higher quality)
        preset: H.264 encoding preset (e.g., slow, medium, fast)
        
    Returns:
        Path to created MP4
    """
    if not frames:
        raise ValueError("No frames provided")
    
    if fps is None:
        fps = 1000 / duration if duration > 0 else 10
    
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode MP4 outputs")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for i, frame in enumerate(frames, start=1):
            frame_path = temp_path / f"frame_{i:04d}.png"
            cv2.imwrite(str(frame_path), frame)
        
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:.3f}",
            "-i",
            str(temp_path / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    return output_path


def create_boomerang_gif(
    images: List[np.ndarray],
    output_path: Union[str, Path],
    duration: int = 100,
    loop: int = 0,
    optimize: bool = True,
    crop_valid_region: bool = True,
    normalize_brightness: bool = True,
    brightness_strength: float = 0.5
) -> Path:
    """
    Create a boomerang-style GIF from aligned images.
    
    The sequence plays forward then backward: 1→2→3→4→3→2→1
    
    Args:
        images: List of aligned BGR images
        output_path: Path for output GIF
        duration: Duration per frame in milliseconds
        loop: Number of loops (0 = infinite)
        optimize: Whether to optimize GIF file size
        crop_valid_region: Whether to crop to valid (non-black) region
        normalize_brightness: Whether to normalize brightness across frames to prevent flashing
        brightness_strength: Strength of brightness normalization (0.0-1.0, default 0.5)
                             0.0 = no correction, 1.0 = full correction to median
        
    Returns:
        Path to created GIF
    """
    frames = make_boomerang_frames(
        images,
        crop_valid_region=crop_valid_region,
        normalize_brightness=normalize_brightness,
        brightness_strength=brightness_strength
    )
    return encode_gif(frames, output_path, duration=duration, loop=loop, optimize=optimize)


def _normalize_brightness(images: List[np.ndarray], strength: float = 0.5) -> List[np.ndarray]:
    """
    Normalize brightness across all frames to prevent flashing in GIFs.
    
    Uses LAB color space to adjust lightness while preserving color.
    Applies partial correction based on strength parameter.
    
    Args:
        images: List of BGR images
        strength: Correction strength (0.0-1.0). 0.0 = no change, 1.0 = full match to median
        
    Returns:
        List of brightness-normalized images
    """
    if not images or len(images) < 2:
        return images
    
    strength = np.clip(strength, 0.0, 1.0)  # Ensure valid range
    
    # Calculate average brightness for each frame (using LAB L channel)
    brightnesses = []
    lab_images = []
    
    for img in images:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_images.append(lab)
        # Calculate mean brightness (L channel), excluding very dark pixels (likely black borders)
        l_channel = lab[:, :, 0]
        mask = l_channel > 10  # Exclude black borders
        if np.any(mask):
            brightnesses.append(np.mean(l_channel[mask]))
        else:
            brightnesses.append(np.mean(l_channel))
    
    # Target brightness: use median to avoid outliers
    target_brightness = np.median(brightnesses)
    
    # Normalize each frame to target brightness (with strength blending)
    normalized = []
    for lab, brightness in zip(lab_images, brightnesses):
        if brightness > 0:
            # Calculate adjustment factor
            full_adjustment = target_brightness / brightness
            
            # Blend between original (1.0) and full adjustment based on strength
            adjustment = 1.0 + (full_adjustment - 1.0) * strength
            
            # Apply adjustment to L channel
            l_original = lab[:, :, 0].astype(np.float32)
            l_adjusted = l_original * adjustment
            l_adjusted = np.clip(l_adjusted, 0, 255).astype(np.uint8)
            
            # Reconstruct LAB image
            lab_normalized = lab.copy()
            lab_normalized[:, :, 0] = l_adjusted
            
            # Convert back to BGR
            normalized.append(cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR))
        else:
            normalized.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
    
    return normalized


def _crop_based_on_transforms(images: List[np.ndarray], transforms: List[np.ndarray], margin: int = 10) -> List[np.ndarray]:
    """
    Crop images based on actual translation amounts from alignment transforms.
    
    For each frame that was translated, crops from the opposite side by that amount.
    For example, if a frame moved 50px left, crop 50px from the right side.
    
    Args:
        images: List of aligned images
        transforms: List of 3x3 transformation matrices (one per image)
        margin: Additional margin to add (in pixels)
        
    Returns:
        List of cropped images
    """
    if not images or len(images) != len(transforms):
        return images
    
    h, w = images[0].shape[:2]
    
    # Extract translation components from each transform
    # Transform matrix: [a b tx]
    #                   [c d ty]
    #                   [0 0 1 ]
    # tx = horizontal translation, ty = vertical translation
    max_left_translation = 0   # Negative tx (moved left) -> black bars on right -> crop from right
    max_right_translation = 0  # Positive tx (moved right) -> black bars on left -> crop from left
    max_up_translation = 0     # Negative ty (moved up) -> black bars on bottom -> crop from bottom
    max_down_translation = 0   # Positive ty (moved down) -> black bars on top -> crop from top
    
    for transform in transforms:
        tx = transform[0, 2]  # Horizontal translation
        ty = transform[1, 2]  # Vertical translation
        
        if tx < 0:  # Moved left, black bars appear on right
            max_left_translation = max(max_left_translation, abs(tx))
        elif tx > 0:  # Moved right, black bars appear on left
            max_right_translation = max(max_right_translation, tx)
        
        if ty < 0:  # Moved up, black bars appear on bottom
            max_up_translation = max(max_up_translation, abs(ty))
        elif ty > 0:  # Moved down, black bars appear on top
            max_down_translation = max(max_down_translation, ty)
    
    # Calculate crop amounts from each side
    crop_left = int(max_right_translation) + margin if max_right_translation > 0 else margin
    crop_right = int(max_left_translation) + margin if max_left_translation > 0 else margin
    crop_top = int(max_down_translation) + margin if max_down_translation > 0 else margin
    crop_bottom = int(max_up_translation) + margin if max_up_translation > 0 else margin
    
    # Calculate final crop region
    x = crop_left
    y = crop_top
    valid_w = w - crop_left - crop_right
    valid_h = h - crop_top - crop_bottom
    
    # Ensure we have a valid region (at least 100x100)
    if valid_w < 100 or valid_h < 100:
        # Fallback: use smaller margins
        crop_left = min(crop_left, w // 4)
        crop_right = min(crop_right, w // 4)
        crop_top = min(crop_top, h // 4)
        crop_bottom = min(crop_bottom, h // 4)
        x = crop_left
        y = crop_top
        valid_w = w - crop_left - crop_right
        valid_h = h - crop_top - crop_bottom
    
    # Final safety check
    if valid_w <= 0 or valid_h <= 0 or x < 0 or y < 0 or x + valid_w > w or y + valid_h > h:
        return images
    
    # Crop all images
    cropped = [img[y:y+valid_h, x:x+valid_w] for img in images]
    
    return cropped


def _crop_to_valid_region(images: List[np.ndarray], threshold: int = 15, margin: int = 10) -> List[np.ndarray]:
    """
    Crop all images to the common valid (non-black) region.
    
    After warping, some images may have black borders. This finds
    the intersection of all valid regions and crops to it.
    Optimized to handle vertical black bars from horizontal translation:
    - Aggressively crops vertically (top/bottom black bars)
    - Preserves horizontal dimensions (minimal horizontal cropping)
    
    Args:
        images: List of aligned images
        threshold: Gray value threshold (pixels darker than this are considered black)
        margin: Margin to add around the valid region (in pixels)
        
    Returns:
        List of cropped images
    """
    if not images:
        return images
    
    h, w = images[0].shape[:2]
    
    # Find valid region for each image
    valid_masks = []
    for img in images:
        # Consider pixel valid if not black/dark (more aggressive threshold)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use higher threshold to exclude dark borders
        valid = (gray > threshold).astype(np.uint8) * 255
        valid_masks.append(valid)
    
    # Intersection of all valid regions (all frames must have content)
    combined_mask = valid_masks[0].copy()
    for mask in valid_masks[1:]:
        combined_mask = cv2.bitwise_and(combined_mask, mask)
    
    # Find bounding box of valid region
    coords = cv2.findNonZero(combined_mask)
    if coords is None:
        # Fallback: use union instead of intersection if intersection is empty
        combined_mask = valid_masks[0].copy()
        for mask in valid_masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        coords = cv2.findNonZero(combined_mask)
        if coords is None:
            return images  # No valid region found at all
    
    x, y, valid_w, valid_h = cv2.boundingRect(coords)
    
    # Analyze row sums to detect horizontal black bars (top/bottom)
    # This helps us be more aggressive with vertical cropping
    gray_combined = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    row_sums = np.sum(gray_combined > threshold, axis=1)  # Sum of valid pixels per row
    
    # Find top and bottom boundaries more precisely
    # Use a threshold: rows with less than 5% valid pixels are considered black bars
    min_valid_pixels_per_row = w * 0.05
    valid_rows = np.where(row_sums > min_valid_pixels_per_row)[0]
    
    if len(valid_rows) > 0:
        # Update y and valid_h based on row analysis (more aggressive vertical cropping)
        new_y = max(0, valid_rows[0] - margin // 2)  # Smaller margin for vertical
        new_h = min(h - new_y, valid_rows[-1] - valid_rows[0] + 1 + margin)  # Smaller margin
        
        # Use row-based cropping if it removes more black bars
        if new_y > y or (new_y + new_h) < (y + valid_h):
            y = new_y
            valid_h = new_h
    
    # For horizontal dimension, be conservative - only crop if there are clear vertical bars
    # Analyze column sums to detect vertical black bars (left/right)
    col_sums = np.sum(gray_combined > threshold, axis=0)  # Sum of valid pixels per column
    min_valid_pixels_per_col = h * 0.10  # Higher threshold - only crop if 10%+ of height is black
    valid_cols = np.where(col_sums > min_valid_pixels_per_col)[0]
    
    if len(valid_cols) > 0:
        # Only crop horizontally if there are significant vertical bars
        # Be conservative: use larger margin and only if it's clearly a black bar
        new_x = max(0, valid_cols[0] - margin * 2)  # Larger margin to preserve width
        new_w = min(w - new_x, valid_cols[-1] - valid_cols[0] + 1 + margin * 4)  # Preserve more width
        
        # Only use column-based cropping if it's clearly removing black bars
        # and doesn't crop too aggressively (preserve at least 80% of width)
        if (new_x > x or (new_x + new_w) < (x + valid_w)) and new_w > w * 0.8:
            x = new_x
            valid_w = new_w
    
    # Add margin around valid region
    # Use smaller vertical margin (more aggressive cropping) and larger horizontal margin (preserve width)
    x = max(0, x - margin * 2)  # Larger horizontal margin to preserve width
    y = max(0, y - margin // 2)  # Smaller vertical margin for aggressive cropping
    valid_w = min(w - x, valid_w + margin * 4)  # Preserve more horizontal space
    valid_h = min(h - y, valid_h + margin)  # Less vertical padding
    
    # Ensure minimum size (don't crop to nothing)
    # Preserve more width, allow more aggressive height cropping
    min_width = max(100, int(w * 0.75))  # Preserve at least 75% of original width
    min_height = max(100, int(h * 0.3))  # Allow cropping to 30% of original height
    
    if valid_w < min_width or valid_h < min_height:
        # If too aggressive, use original bounding box with adjusted margins
        x, y, valid_w, valid_h = cv2.boundingRect(coords)
        x = max(0, x - margin * 2)  # Preserve width
        y = max(0, y - margin // 2)  # Aggressive vertical cropping
        valid_w = min(w - x, valid_w + margin * 2)  # Preserve width
        valid_h = min(h - y, valid_h + margin)  # Less vertical padding
    
    # Crop all images
    cropped = [img[y:y+valid_h, x:x+valid_w] for img in images]
    
    return cropped


def create_side_by_side(
    images: List[np.ndarray],
    max_width: int = 1920
) -> np.ndarray:
    """
    Create a side-by-side comparison of multiple images.
    
    Args:
        images: List of images
        max_width: Maximum total width (will resize if exceeded)
        
    Returns:
        Combined image
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Resize to same height
    target_h = min(img.shape[0] for img in images)
    resized = []
    for img in images:
        h, w = img.shape[:2]
        if h != target_h:
            scale = target_h / h
            new_w = int(w * scale)
            resized.append(cv2.resize(img, (new_w, target_h)))
        else:
            resized.append(img)
    
    # Concatenate horizontally
    combined = np.hstack(resized)
    
    # Resize if too wide
    if combined.shape[1] > max_width:
        scale = max_width / combined.shape[1]
        new_h = int(combined.shape[0] * scale)
        combined = cv2.resize(combined, (max_width, new_h))
    
    return combined


def create_comparison_gif(
    original_images: List[np.ndarray],
    aligned_images: List[np.ndarray],
    output_path: Union[str, Path],
    duration: int = 150
) -> Path:
    """
    Create a GIF showing before/after alignment comparison.
    
    Args:
        original_images: List of original images
        aligned_images: List of aligned images
        output_path: Path for output GIF
        duration: Duration per frame in milliseconds
        
    Returns:
        Path to created GIF
    """
    output_path = Path(output_path)
    
    # Create side-by-side for each frame
    frames = []
    for orig, aligned in zip(original_images, aligned_images):
        # Add labels
        orig_labeled = _add_label(orig, "Original")
        aligned_labeled = _add_label(aligned, "Aligned")
        combined = np.hstack([orig_labeled, aligned_labeled])
        frames.append(combined)
    
    return create_boomerang_gif(frames, output_path, duration=duration, crop_valid_region=False)


def _add_label(img: np.ndarray, text: str) -> np.ndarray:
    """Add a text label to the top of an image."""
    labeled = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    cv2.rectangle(labeled, (0, 0), (text_w + 20, text_h + 20), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(labeled, text, (10, text_h + 10), font, font_scale, (255, 255, 255), thickness)
    
    return labeled


def resize_for_web(
    images: List[np.ndarray],
    max_dimension: int = 800
) -> List[np.ndarray]:
    """
    Resize images for web-friendly GIF output.
    
    Args:
        images: List of images
        max_dimension: Maximum width or height
        
    Returns:
        List of resized images
    """
    if not images:
        return images
    
    h, w = images[0].shape[:2]
    scale = min(max_dimension / max(h, w), 1.0)
    
    if scale >= 1.0:
        return images
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return [cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA) for img in images]


def get_gif_info(gif_path: Union[str, Path]) -> dict:
    """
    Get information about a GIF file.
    
    Args:
        gif_path: Path to GIF file
        
    Returns:
        Dictionary with GIF metadata
    """
    gif_path = Path(gif_path)
    
    if not gif_path.exists():
        return {"error": "File not found"}
    
    with Image.open(gif_path) as img:
        info = {
            "path": str(gif_path),
            "size_bytes": gif_path.stat().st_size,
            "size_kb": gif_path.stat().st_size / 1024,
            "width": img.width,
            "height": img.height,
            "n_frames": getattr(img, 'n_frames', 1),
            "is_animated": getattr(img, 'is_animated', False),
        }
        
        # Try to get duration
        try:
            info["duration_ms"] = img.info.get('duration', 0)
        except Exception:
            info["duration_ms"] = 0
    
    return info
