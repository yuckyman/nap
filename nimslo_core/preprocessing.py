"""
Preprocessing module for Nimslo images.

Handles film grain reduction and exposure balancing to prepare
images for segmentation and feature extraction.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def denoise_film_grain(
    img: np.ndarray,
    h: float = 10.0,
    hColor: float = 10.0,
    templateWindowSize: int = 7,
    searchWindowSize: int = 21,
    **kwargs
) -> np.ndarray:
    """
    Reduce film grain using non-local means denoising.
    
    This approach preserves edges while smoothing grain artifacts,
    and is more stable in Jupyter environments than pyramid mean shift filtering.
    Uses fastNlMeansDenoisingColored for color image denoising.
    
    Args:
        img: Input BGR image
        h: Filter strength for luminance component (default: 10.0, good for film grain)
        hColor: Filter strength for color components (default: 10.0, same as h)
        templateWindowSize: Size of template patch (default: 7, must be odd)
        searchWindowSize: Size of search window (default: 21, must be odd)
        **kwargs: Additional parameters (for backward compatibility, ignored)
        
    Returns:
        Denoised BGR image
    """
    # Ensure window sizes are odd (required by OpenCV)
    if templateWindowSize % 2 == 0:
        templateWindowSize += 1
    if searchWindowSize % 2 == 0:
        searchWindowSize += 1
    
    return cv2.fastNlMeansDenoisingColored(
        img,
        h=h,
        hColor=hColor,
        templateWindowSize=templateWindowSize,
        searchWindowSize=searchWindowSize
    )


def balance_exposure(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Balance exposure using CLAHE on the L channel in LAB color space.
    
    This addresses exposure variations across the four Nimslo lenses
    due to manufacturing tolerances.
    
    Args:
        img: Input BGR image
        clip_limit: Contrast limiting threshold
        tile_size: Size of grid for histogram equalization
        
    Returns:
        Exposure-balanced BGR image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to the L (lightness) channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to BGR
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_image(
    img: np.ndarray,
    denoise: bool = True,
    balance: bool = True,
    denoise_params: Optional[dict] = None,
    balance_params: Optional[dict] = None
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single Nimslo image.
    
    Args:
        img: Input BGR image
        denoise: Whether to apply film grain reduction
        balance: Whether to apply exposure balancing
        denoise_params: Optional params for denoise_film_grain
        balance_params: Optional params for balance_exposure
        
    Returns:
        Preprocessed BGR image
    """
    result = img.copy()
    
    if denoise:
        params = denoise_params or {}
        result = denoise_film_grain(result, **params)
    
    if balance:
        params = balance_params or {}
        result = balance_exposure(result, **params)
    
    return result


def resize_for_processing(
    img: np.ndarray,
    max_dimension: int = 1024
) -> Tuple[np.ndarray, float]:
    """
    Resize image for faster processing while maintaining aspect ratio.
    
    Args:
        img: Input image
        max_dimension: Maximum width or height
        
    Returns:
        Tuple of (resized image, scale factor)
    """
    h, w = img.shape[:2]
    scale = min(max_dimension / max(h, w), 1.0)
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    
    return img, 1.0


def normalize_sizes(images: list) -> list:
    """
    Ensure all images in a batch have the same dimensions.
    
    Crops to the smallest common size if there are slight differences.
    
    Args:
        images: List of BGR images
        
    Returns:
        List of size-normalized images
    """
    if not images:
        return images
    
    # Find minimum dimensions
    min_h = min(img.shape[0] for img in images)
    min_w = min(img.shape[1] for img in images)
    
    # Crop all images to minimum size (centered crop)
    normalized = []
    for img in images:
        h, w = img.shape[:2]
        start_y = (h - min_h) // 2
        start_x = (w - min_w) // 2
        cropped = img[start_y:start_y + min_h, start_x:start_x + min_w]
        normalized.append(cropped)
    
    return normalized

