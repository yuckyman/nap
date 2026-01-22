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
    brightness_strength: float = 0.5,
    end_on_first: bool = True,
    force_even_dimensions: bool = False
) -> List[np.ndarray]:
    """
    Create boomerang frame sequence from aligned images.
    
    The sequence plays forward then backward.
    
    - If end_on_first=True:  1→2→3→4→3→2→1  (stack/concat friendly for video)
    - If end_on_first=False: 1→2→3→4→3→2    (GIF-loop friendly; avoids duplicating frame 1)
    
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
        if end_on_first:
            # 4 frames: 1,2,3,4,3,2,1
            # 3 frames: 1,2,3,2,1
            frames = images + images[-2::-1]
        else:
            # 4 frames: 1,2,3,4,3,2
            # 3 frames: 1,2,3,2
            frames = images + images[-2:0:-1]
    else:
        frames = images
    
    if force_even_dimensions:
        # H.264 yuv420p requires even width/height.
        # Crop by 1px if needed; avoids resampling blur.
        frames = [_crop_to_even_dimensions(f) for f in frames]
    return frames


def _crop_to_even_dimensions(img: np.ndarray) -> np.ndarray:
    """Crop image by 1px if needed to make width/height even (no resampling)."""
    h, w = img.shape[:2]
    new_w = w - (w % 2)
    new_h = h - (h % 2)
    if new_w == w and new_h == h:
        return img
    return img[:new_h, :new_w]


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
    fps: Optional[float] = 10.0,
    duration: int = 100,
    loops: int = 1,
    crf: int = 18,
    preset: str = "slow",
    tune: str = "grain"
) -> Path:
    """
    Encode a list of frames into an MP4 using ffmpeg (H.264).
    
    Args:
        frames: List of BGR frames
        output_path: Path for output MP4
        fps: Frames per second (if None, derived from duration)
        duration: Duration per frame in milliseconds (used if fps is None)
        loops: Number of times to repeat the boomerang sequence (default: 10)
        crf: H.264 quality factor (lower is higher quality)
        preset: H.264 encoding preset (e.g., slow, medium, fast)
        tune: x264 tuning (e.g. grain, film, animation)
        
    Returns:
        Path to created MP4
    """
    if not frames:
        raise ValueError("No frames provided")

    # Default behavior: make MP4s longer by repeating the loop.
    # This is cheap because our sequences are short (typically 4-6 frames).
    if loops is None:
        loops = 1
    loops = int(max(1, loops))
    
    # Make single-loop MP4s "stackable": avoid ending on the same frame we start with.
    # Our boomerang sequences typically start and end on frame 1 (e.g. 1→2→3→4→3→2→1).
    # Dropping the trailing frame prevents a visible pause when concatenating MP4s externally.
    if loops == 1 and len(frames) > 1:
        try:
            if (frames[0] is frames[-1]) or np.array_equal(frames[0], frames[-1]):
                frames = frames[:-1]
        except Exception:
            # If anything goes sideways (unlikely), just keep original frames.
            pass

    if loops > 1:
        # Avoid a visible "pause" at loop boundaries by not duplicating the first frame
        # on each repetition. This assumes the boomerang sequence ends on frame 1.
        base = frames
        frames = list(base)
        for _ in range(loops - 1):
            frames.extend(base[1:])
    
    if fps is None:
        # Default to 10fps (slower, more filmic). Duration is ignored when fps is set.
        fps = 10.0
    
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
            "-hide_banner",
            "-framerate",
            f"{fps:.3f}",
            "-start_number",
            "1",
            "-i",
            str(temp_path / "frame_%04d.png"),
            # Frames are pre-cropped to even dimensions; keep filter as CFR only.
            "-vf",
            f"fps={fps:.3f}",
            # Force constant frame rate output (prevents some players from “hanging”).
            "-fps_mode",
            "cfr",
            "-r",
            f"{fps:.3f}",
            "-c:v",
            "libx264",
            "-tune",
            tune,
            "-preset",
            preset,
            "-crf",
            str(crf),
            # Avoid some grain-smearing ratecontrol heuristics; keep grain texture.
            "-x264-params",
            "mbtree=0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            stderr = ""
            try:
                stderr = (e.stderr or b"").decode("utf-8", errors="replace")
            except Exception:
                stderr = str(e)
            raise RuntimeError(f"ffmpeg mp4 encode failed: {stderr.strip()}") from e
    
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


def _crop_to_valid_region(
    images: List[np.ndarray],
    threshold: int = 20,
    margin: int = 2,
    erode_px: int = 2
) -> List[np.ndarray]:
    """
    Crop all images to the common valid (non-border) region.
    
    Robust against thin black strips along entire edges after warping:
    - build a valid mask per frame (gray > threshold)
    - keep only the main connected component (center-connected / largest)
    - AND masks across frames so any border in any frame is removed
    - erode slightly to remove residual edge pixels
    
    Returns original images if no safe crop found.
    """
    if not images:
        return images
    
    h, w = images[0].shape[:2]
    if any(img.shape[:2] != (h, w) for img in images):
        return images
    
    cy, cx = h // 2, w // 2
    masks: List[np.ndarray] = []
    
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        valid = (gray > threshold).astype(np.uint8)
        
        num_labels, labels = cv2.connectedComponents(valid, connectivity=8)
        if num_labels <= 1:
            masks.append(valid * 255)
            continue
        
        center_label = labels[cy, cx]
        if center_label != 0:
            keep_label = int(center_label)
        else:
            counts = np.bincount(labels.reshape(-1))
            counts[0] = 0
            keep_label = int(np.argmax(counts))
        
        main = (labels == keep_label).astype(np.uint8) * 255
        masks.append(main)
    
    combined = masks[0].copy()
    for m in masks[1:]:
        combined = cv2.bitwise_and(combined, m)
    
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erode_px + 1, 2 * erode_px + 1))
        combined = cv2.erode(combined, k, iterations=1)
    
    coords = cv2.findNonZero(combined)
    if coords is None:
        return images
    
    x, y, cw, ch = cv2.boundingRect(coords)
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(w, x + cw + margin)
    y1 = min(h, y + ch + margin)
    
    # Safety: don't crop too aggressively
    if (x1 - x0) < int(w * 0.5) or (y1 - y0) < int(h * 0.5):
        return images
    
    return [img[y0:y1, x0:x1] for img in images]


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
