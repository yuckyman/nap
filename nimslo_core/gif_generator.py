"""
GIF generation module for Nimslo images.

Creates smooth boomerang-style animated GIFs from aligned frames.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple, Union


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
    if not images:
        raise ValueError("No images provided")
    
    output_path = Path(output_path)
    
    # Optionally crop to valid region
    if crop_valid_region:
        images = _crop_to_valid_region(images)
    
    # Normalize brightness across frames to prevent flashing
    if normalize_brightness:
        images = _normalize_brightness(images, strength=brightness_strength)
    
    # Convert to PIL images (RGB)
    pil_images = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(rgb))
    
    # Create boomerang sequence: forward + reverse (excluding endpoints)
    if len(pil_images) > 1:
        boomerang = pil_images + pil_images[-2:0:-1]
    else:
        boomerang = pil_images
    
    # Save as GIF
    boomerang[0].save(
        output_path,
        save_all=True,
        append_images=boomerang[1:],
        duration=duration,
        loop=loop,
        optimize=optimize
    )
    
    return output_path


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


def _crop_to_valid_region(images: List[np.ndarray], threshold: int = 15, margin: int = 10) -> List[np.ndarray]:
    """
    Crop all images to the common valid (non-black) region.
    
    After warping, some images may have black borders. This finds
    the intersection of all valid regions and crops to it.
    Optimized to handle vertical black bars from horizontal translation.
    
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
    
    # For vertical bars (horizontal translation), be more aggressive on horizontal cropping
    # Check if we have significant vertical bars by analyzing column sums
    gray_combined = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    col_sums = np.sum(gray_combined > threshold, axis=0)  # Sum of valid pixels per column
    
    # Find left and right boundaries more precisely
    # Use a threshold: columns with less than 5% valid pixels are considered black bars
    min_valid_pixels = h * 0.05
    valid_cols = np.where(col_sums > min_valid_pixels)[0]
    
    if len(valid_cols) > 0:
        # Update x and valid_w based on column analysis
        new_x = max(0, valid_cols[0] - margin)
        new_w = min(w - new_x, valid_cols[-1] - valid_cols[0] + 1 + 2 * margin)
        
        # Only use column-based cropping if it's more aggressive (removes more black bars)
        if new_x > x or (new_x + new_w) < (x + valid_w):
            x = new_x
            valid_w = new_w
    
    # Add margin around valid region
    x = max(0, x - margin)
    y = max(0, y - margin)
    valid_w = min(w - x, valid_w + 2 * margin)
    valid_h = min(h - y, valid_h + 2 * margin)
    
    # Ensure minimum size (don't crop to nothing)
    # But allow more aggressive cropping for width (vertical bars)
    min_width = max(100, int(w * 0.3))  # At least 30% of original width
    min_height = max(100, int(h * 0.5))  # At least 50% of original height
    
    if valid_w < min_width or valid_h < min_height:
        # If too aggressive, use original bounding box with smaller margin
        x, y, valid_w, valid_h = cv2.boundingRect(coords)
        x = max(0, x - margin // 2)
        y = max(0, y - margin // 2)
        valid_w = min(w - x, valid_w + margin)
        valid_h = min(h - y, valid_h + margin)
    
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

