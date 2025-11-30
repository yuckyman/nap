"""
Alignment module for Nimslo images.

Uses SIFT feature matching with homography estimation
to align stereoscopic image pairs.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AlignmentResult:
    """Container for alignment results."""
    transform: np.ndarray  # 3x3 homography or 2x3 affine matrix
    inliers: int  # Number of RANSAC inliers
    total_matches: int  # Total number of matches
    iou: float  # Intersection over Union of masks
    confidence: float  # Overall alignment confidence
    

def extract_features(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    n_features: int = 1000
) -> Tuple[list, np.ndarray]:
    """
    Extract SIFT features from an image, optionally within a mask region.
    
    Args:
        img: Input BGR image
        mask: Optional binary mask to restrict feature detection
        n_features: Maximum number of features to detect
        
    Returns:
        Tuple of (keypoints, descriptors)
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply mask if provided
    if mask is not None:
        # Ensure mask is binary
        mask_binary = (mask > 127).astype(np.uint8) * 255
    else:
        mask_binary = None
    
    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=n_features)
    
    # Detect and compute
    keypoints, descriptors = sift.detectAndCompute(gray, mask_binary)
    
    return keypoints, descriptors


def match_features(
    des1: np.ndarray,
    des2: np.ndarray,
    ratio_threshold: float = 0.75
) -> List[cv2.DMatch]:
    """
    Match SIFT descriptors using FLANN with Lowe's ratio test.
    
    Args:
        des1: Descriptors from first image
        des2: Descriptors from second image
        ratio_threshold: Lowe's ratio test threshold
        
    Returns:
        List of good matches
    """
    if des1 is None or des2 is None:
        return []
    
    if len(des1) < 2 or len(des2) < 2:
        return []
    
    # FLANN parameters for SIFT
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except cv2.error:
        return []
    
    # Apply ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    return good_matches


def estimate_homography(
    kp1: list,
    kp2: list,
    matches: List[cv2.DMatch],
    ransac_threshold: float = 5.0,
    min_matches: int = 10
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate homography matrix from matched keypoints.
    
    Args:
        kp1: Keypoints from first image
        kp2: Keypoints from second image
        matches: List of matches
        ransac_threshold: RANSAC reprojection threshold in pixels
        min_matches: Minimum matches required for reliable estimation
        
    Returns:
        Tuple of (homography matrix, inlier mask) or (None, None) if failed
    """
    if len(matches) < min_matches:
        return None, None
    
    # Extract matched point coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Estimate homography with RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
    
    return H, mask


def estimate_translation(
    kp1: list,
    kp2: list,
    matches: List[cv2.DMatch],
    ransac_threshold: float = 5.0,
    min_matches: int = 3
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate translation-only transformation from matched keypoints.
    
    Only allows translation (x, y shifts), no rotation, scale, or shear.
    Uses RANSAC to robustly estimate translation in the presence of outliers.
    
    Args:
        kp1: Keypoints from first image
        kp2: Keypoints from second image
        matches: List of matches
        ransac_threshold: RANSAC reprojection threshold in pixels
        min_matches: Minimum matches required
        
    Returns:
        Tuple of (2x3 translation matrix, inlier mask) or (None, None) if failed
        Translation matrix is [[1, 0, tx], [0, 1, ty]]
    """
    if len(matches) < min_matches:
        return None, None
    
    # Extract matched point coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Compute translation vectors for each match
    translations = dst_pts - src_pts  # Shape: (n_matches, 2)
    
    # Use RANSAC to find the best translation
    # Sample matches, compute median translation, count inliers
    best_inliers = 0
    best_translation = None
    best_mask = None
    
    # Adaptive number of iterations based on expected outlier ratio
    # Formula: log(1-p) / log(1-(1-eps)^s) where p=0.99, eps=0.5, s=3
    n_iterations = min(100, max(10, int(np.log(0.01) / np.log(0.125))))
    
    for _ in range(n_iterations):
        # Sample a small subset of matches (at least 2, but prefer 3+)
        sample_size = min(max(2, len(matches) // 4), len(matches))
        if sample_size < len(matches):
            sample_indices = np.random.choice(len(matches), size=sample_size, replace=False)
        else:
            sample_indices = np.arange(len(matches))
        
        sample_translations = translations[sample_indices]
        
        # Use median translation (more robust than mean to outliers)
        candidate_tx = np.median(sample_translations[:, 0])
        candidate_ty = np.median(sample_translations[:, 1])
        
        # Count inliers: matches where reprojection error < threshold
        predicted_dst = src_pts + np.array([candidate_tx, candidate_ty])
        errors = np.linalg.norm(predicted_dst - dst_pts, axis=1)
        inlier_mask = errors < ransac_threshold
        n_inliers = np.sum(inlier_mask)
        
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_translation = (candidate_tx, candidate_ty)
            best_mask = inlier_mask
    
    # If we found a good translation, refine using all inliers
    if best_translation is not None and best_inliers >= min_matches:
        # Refine using median of all inlier translations
        inlier_translations = translations[best_mask]
        final_tx = np.median(inlier_translations[:, 0])
        final_ty = np.median(inlier_translations[:, 1])
        
        # Create 2x3 translation matrix
        M = np.float32([[1, 0, final_tx], [0, 1, final_ty]])
        
        return M, best_mask.astype(np.uint8)
    
    return None, None


def estimate_affine(
    kp1: list,
    kp2: list,
    matches: List[cv2.DMatch],
    ransac_threshold: float = 5.0,
    min_matches: int = 6
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate affine transformation from matched keypoints.
    
    More robust than homography for roughly planar scenes.
    
    Args:
        kp1: Keypoints from first image
        kp2: Keypoints from second image
        matches: List of matches
        ransac_threshold: RANSAC reprojection threshold
        min_matches: Minimum matches required
        
    Returns:
        Tuple of (2x3 affine matrix, inlier mask) or (None, None) if failed
    """
    if len(matches) < min_matches:
        return None, None
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Estimate affine with RANSAC
    M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, 
                                           ransacReprojThreshold=ransac_threshold)
    
    return M, mask


def center_images_on_subject(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    target_center: Optional[Tuple[int, int]] = None
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Center all images on their detected subjects.
    
    Translates each image so that the centroid of its mask is at the target center.
    This helps with alignment by ensuring subjects start in similar positions.
    
    Args:
        images: List of BGR images
        masks: List of binary masks (255 for subject, 0 for background)
        target_center: Target center point (cx, cy). If None, uses center of first image.
        
    Returns:
        Tuple of (centered_images, centered_masks, translation_matrices)
        Each translation matrix is 3x3 (homogeneous coordinates)
    """
    if len(images) != len(masks):
        raise ValueError("images and masks must have same length")
    
    if not images:
        return [], [], []
    
    h, w = images[0].shape[:2]
    if target_center is None:
        target_center = (w // 2, h // 2)
    
    centered_images = []
    centered_masks = []
    translations = []
    
    for img, mask in zip(images, masks):
        # Find mask centroid
        M = cv2.moments(mask)
        if M["m00"] == 0:
            # No subject detected, use image center (no translation needed)
            cx, cy = w // 2, h // 2
        else:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        
        # Calculate translation needed
        dx = target_center[0] - cx
        dy = target_center[1] - cy
        
        # Create translation matrix (2x3 affine)
        M_trans_affine = np.float32([[1, 0, dx], [0, 1, dy]])
        
        # Convert to 3x3 homogeneous for composition
        M_trans = np.vstack([M_trans_affine, [0, 0, 1]])
        
        # Apply translation
        centered_img = cv2.warpAffine(img, M_trans_affine, (w, h), 
                                      borderMode=cv2.BORDER_CONSTANT, 
                                      borderValue=0)
        centered_mask = cv2.warpAffine(mask, M_trans_affine, (w, h),
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0)
        
        centered_images.append(centered_img)
        centered_masks.append(centered_mask)
        translations.append(M_trans)
    
    return centered_images, centered_masks, translations


def compose_transforms(
    transform1: np.ndarray,
    transform2: np.ndarray
) -> np.ndarray:
    """
    Compose two transformation matrices.
    
    Result is transform2 @ transform1 (apply transform1 first, then transform2).
    
    Args:
        transform1: First transformation (3x3 or 2x3)
        transform2: Second transformation (3x3 or 2x3)
        
    Returns:
        Composed transformation (3x3)
    """
    # Convert to 3x3 if needed
    if transform1.shape == (2, 3):
        T1 = np.vstack([transform1, [0, 0, 1]])
    else:
        T1 = transform1
    
    if transform2.shape == (2, 3):
        T2 = np.vstack([transform2, [0, 0, 1]])
    else:
        T2 = transform2
    
    # Compose: T2 @ T1 (apply T1 first, then T2)
    return T2 @ T1


def calculate_iou(
    mask1: np.ndarray,
    mask2: np.ndarray,
    transform: np.ndarray
) -> float:
    """
    Calculate Intersection over Union of two masks after transformation.
    
    Args:
        mask1: Binary mask for first image
        mask2: Binary mask for second image (reference)
        transform: Transformation matrix to apply to mask1
        
    Returns:
        IoU score between 0 and 1
    """
    h, w = mask2.shape[:2]
    
    # Warp mask1 using the transformation
    if transform.shape == (2, 3):
        # Affine transform
        warped_mask1 = cv2.warpAffine(mask1, transform, (w, h))
    else:
        # Homography
        warped_mask1 = cv2.warpPerspective(mask1, transform, (w, h))
    
    # Calculate intersection and union
    intersection = cv2.bitwise_and(warped_mask1, mask2)
    union = cv2.bitwise_or(warped_mask1, mask2)
    
    intersection_area = np.sum(intersection > 0)
    union_area = np.sum(union > 0)
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def align_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    mask1: Optional[np.ndarray] = None,
    mask2: Optional[np.ndarray] = None,
    use_translation_only: bool = True,
    n_features: int = 1000,
    ratio_threshold: float = 0.75,
    ransac_threshold: float = 5.0
) -> Tuple[Optional[np.ndarray], AlignmentResult]:
    """
    Align img1 to img2 using feature-based matching.
    
    Args:
        img1: Source image to be warped
        img2: Target/reference image
        mask1: Optional mask for img1 subject region
        mask2: Optional mask for img2 subject region
        use_translation_only: If True, only allow translation (no rotation). 
                             If False, use affine transformation.
        n_features: Maximum SIFT features
        ratio_threshold: Lowe's ratio test threshold
        ransac_threshold: RANSAC reprojection threshold
        
    Returns:
        Tuple of (warped image, AlignmentResult)
    """
    # Extract features
    kp1, des1 = extract_features(img1, mask1, n_features)
    kp2, des2 = extract_features(img2, mask2, n_features)
    
    # Match features
    matches = match_features(des1, des2, ratio_threshold)
    
    # Estimate transformation
    if use_translation_only:
        transform, inlier_mask = estimate_translation(kp1, kp2, matches, ransac_threshold)
    else:
        transform, inlier_mask = estimate_affine(kp1, kp2, matches, ransac_threshold)
    
    if transform is None:
        result = AlignmentResult(
            transform=np.eye(3),
            inliers=0,
            total_matches=len(matches),
            iou=0.0,
            confidence=0.0
        )
        return None, result
    
    # Warp image (translation and affine both use warpAffine)
    h, w = img2.shape[:2]
    warped = cv2.warpAffine(img1, transform, (w, h))
    
    # Calculate metrics
    n_inliers = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
    
    # Calculate IoU if masks provided
    if mask1 is not None and mask2 is not None:
        iou = calculate_iou(mask1, mask2, transform)
    else:
        iou = 0.0
    
    # Confidence based on inlier ratio and IoU
    inlier_ratio = n_inliers / len(matches) if matches else 0
    confidence = 0.5 * inlier_ratio + 0.5 * iou
    
    # Convert affine to 3x3 for consistent storage
    if transform.shape == (2, 3):
        full_transform = np.vstack([transform, [0, 0, 1]])
    else:
        full_transform = transform
    
    result = AlignmentResult(
        transform=full_transform,
        inliers=n_inliers,
        total_matches=len(matches),
        iou=iou,
        confidence=confidence
    )
    
    return warped, result


def align_images(
    images: List[np.ndarray],
    masks: Optional[List[np.ndarray]] = None,
    reference_idx: int = 0,
    use_translation_only: bool = True,
    **kwargs
) -> Tuple[List[np.ndarray], List[AlignmentResult]]:
    """
    Align multiple images to a reference image.
    
    Args:
        images: List of BGR images
        masks: Optional list of masks for each image
        reference_idx: Index of reference image (default: first image)
        use_translation_only: If True, only allow translation (no rotation)
        **kwargs: Additional arguments passed to align_pair
        
    Returns:
        Tuple of (list of aligned images, list of alignment results)
    """
    if masks is None:
        masks = [None] * len(images)
    
    ref_img = images[reference_idx]
    ref_mask = masks[reference_idx]
    
    aligned_images = []
    results = []
    
    for i, (img, mask) in enumerate(zip(images, masks)):
        if i == reference_idx:
            # Reference image stays as-is
            aligned_images.append(img.copy())
            results.append(AlignmentResult(
                transform=np.eye(3),
                inliers=0,
                total_matches=0,
                iou=1.0,
                confidence=1.0
            ))
        else:
            warped, result = align_pair(
                img, ref_img,
                mask, ref_mask,
                use_translation_only=use_translation_only,
                **kwargs
            )
            
            if warped is not None:
                aligned_images.append(warped)
            else:
                aligned_images.append(img.copy())
            results.append(result)
    
    return aligned_images, results


def create_anaglyph(
    img1: np.ndarray,
    img2: np.ndarray,
    mode: str = "red-cyan"
) -> np.ndarray:
    """
    Create an anaglyph image for stereo visualization.
    
    Args:
        img1: Left image (BGR)
        img2: Right image (BGR)
        mode: Anaglyph mode ("red-cyan", "green-magenta")
        
    Returns:
        Anaglyph image (BGR)
    """
    # Convert to grayscale
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
        
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2
    
    # Create anaglyph
    h, w = gray1.shape[:2]
    anaglyph = np.zeros((h, w, 3), dtype=np.uint8)
    
    if mode == "red-cyan":
        anaglyph[:, :, 2] = gray1  # Red channel from left image
        anaglyph[:, :, 1] = gray2  # Green channel from right image
        anaglyph[:, :, 0] = gray2  # Blue channel from right image
    elif mode == "green-magenta":
        anaglyph[:, :, 1] = gray1  # Green from left
        anaglyph[:, :, 2] = gray2  # Red from right
        anaglyph[:, :, 0] = gray2  # Blue from right
    
    return anaglyph


def visualize_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    kp1: list,
    kp2: list,
    matches: List[cv2.DMatch],
    max_matches: int = 50
) -> np.ndarray:
    """
    Visualize feature matches between two images.
    
    Args:
        img1: First image
        img2: Second image
        kp1: Keypoints from first image
        kp2: Keypoints from second image
        matches: List of matches
        max_matches: Maximum matches to display
        
    Returns:
        Visualization image
    """
    # Sort by distance and take top matches
    sorted_matches = sorted(matches, key=lambda x: x.distance)[:max_matches]
    
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2,
        sorted_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return vis


