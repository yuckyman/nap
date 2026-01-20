"""
Image rectification module for Nimslo stereo pairs.

Based on Zhang (1999) and Hartley (1999) rectification methods.
Rectifies stereo pairs so epipolar lines are horizontal, simplifying
feature matching to 1D search along scanlines.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def estimate_fundamental_matrix(
    pts1: np.ndarray,
    pts2: np.ndarray,
    method: int = cv2.FM_RANSAC,
    ransac_threshold: float = 3.0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate fundamental matrix from point correspondences.
    
    Args:
        pts1: Points in first image (Nx2)
        pts2: Points in second image (Nx2)
        method: Method for estimation (FM_RANSAC, FM_8POINT, etc.)
        ransac_threshold: RANSAC threshold in pixels
        
    Returns:
        Tuple of (fundamental matrix, inlier mask) or (None, None) if failed
    """
    if len(pts1) < 8:
        return None, None
    
    # OpenCV requires points in shape (N, 1, 2)
    pts1_reshaped = pts1.reshape(-1, 1, 2).astype(np.float32)
    pts2_reshaped = pts2.reshape(-1, 1, 2).astype(np.float32)
    
    # Estimate fundamental matrix
    F, mask = cv2.findFundamentalMat(
        pts1_reshaped,
        pts2_reshaped,
        method=method,
        ransacReprojThreshold=ransac_threshold,
        confidence=0.99
    )
    
    if F is None or F.shape != (3, 3):
        return None, None
    
    return F, mask


def stereo_rectify_uncalibrated(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: Optional[np.ndarray] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Rectify stereo pair using uncalibrated method (Hartley, 1999).
    
    This method only requires the fundamental matrix, not full camera calibration.
    Uses cv2.stereoRectifyUncalibrated which implements Hartley's method.
    
    Args:
        img1: First image
        img2: Second image
        pts1: Corresponding points in first image (Nx2)
        pts2: Corresponding points in second image (Nx2)
        F: Optional pre-computed fundamental matrix
        
    Returns:
        Tuple of (rectified_img1, rectified_img2, H1, H2) or (None, ...) if failed
        H1, H2 are rectification homographies
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Estimate fundamental matrix if not provided
    if F is None:
        F, mask = estimate_fundamental_matrix(pts1, pts2)
        if F is None:
            return None, None, None, None
        # Filter points to inliers only
        if mask is not None:
            inlier_mask = mask.ravel() == 1
            pts1 = pts1[inlier_mask]
            pts2 = pts2[inlier_mask]
    
    # Rectify using uncalibrated method
    # This implements Hartley's rectification algorithm
    ret, H1, H2 = cv2.stereoRectifyUncalibrated(
        pts1.reshape(-1, 1, 2).astype(np.float32),
        pts2.reshape(-1, 1, 2).astype(np.float32),
        F,
        (w1, h1),
        threshold=5.0
    )
    
    if not ret or H1 is None or H2 is None:
        return None, None, None, None
    
    # Apply rectification homographies
    rectified_img1 = cv2.warpPerspective(img1, H1, (w1, h1))
    rectified_img2 = cv2.warpPerspective(img2, H2, (w2, h2))
    
    return rectified_img1, rectified_img2, H1, H2


def stereo_rectify_calibrated(
    img1: np.ndarray,
    img2: np.ndarray,
    K1: np.ndarray,
    D1: np.ndarray,
    K2: np.ndarray,
    D2: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    image_size: Tuple[int, int],
    alpha: float = 0.0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Rectify stereo pair using calibrated method (Zhang, 1999).
    
    This method requires full camera calibration (intrinsics, extrinsics).
    Produces better results than uncalibrated method but requires more setup.
    
    Args:
        img1: First image
        img2: Second image
        K1: Camera matrix for first camera (3x3)
        D1: Distortion coefficients for first camera
        K2: Camera matrix for second camera (3x3)
        D2: Distortion coefficients for second camera
        R: Rotation matrix from camera 1 to camera 2 (3x3)
        T: Translation vector from camera 1 to camera 2 (3x1)
        image_size: (width, height) of images
        alpha: Free scaling parameter (0=no black areas, 1=all pixels valid)
        
    Returns:
        Tuple of (rectified_img1, rectified_img2, R1, R2, P1, P2) or (None, ...) if failed
        R1, R2 are rectification rotations
        P1, P2 are projection matrices
    """
    # Rectify using calibrated method
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1,
        K2, D2,
        image_size,
        R, T,
        alpha=alpha,
        flags=cv2.CALIB_ZERO_DISPARITY
    )
    
    # Compute rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
    
    # Apply rectification
    rectified_img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    rectified_img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    
    return rectified_img1, rectified_img2, R1, R2, P1, P2


def verify_rectification(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    H1: np.ndarray,
    H2: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Verify rectification quality by checking epipolar line alignment.
    
    After rectification, corresponding points should have the same y-coordinate
    (epipolar lines should be horizontal).
    
    Args:
        img1: First rectified image
        img2: Second rectified image
        pts1: Points in original first image (Nx2)
        pts2: Points in original second image (Nx2)
        H1: Rectification homography for first image
        H2: Rectification homography for second image
        
    Returns:
        Tuple of (mean_vertical_error, error_per_point)
        Lower error means better rectification
    """
    # Transform points to rectified space
    pts1_homogeneous = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_homogeneous = np.hstack([pts2, np.ones((len(pts2), 1))])
    
    pts1_rect = (H1 @ pts1_homogeneous.T).T
    pts2_rect = (H2 @ pts2_homogeneous.T).T
    
    # Normalize homogeneous coordinates
    pts1_rect = pts1_rect[:, :2] / pts1_rect[:, 2:3]
    pts2_rect = pts2_rect[:, :2] / pts2_rect[:, 2:3]
    
    # Compute vertical error (should be ~0 for good rectification)
    vertical_errors = np.abs(pts1_rect[:, 1] - pts2_rect[:, 1])
    mean_error = np.mean(vertical_errors)
    
    return mean_error, vertical_errors


def draw_epipolar_lines(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: np.ndarray,
    num_lines: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw epipolar lines on images to visualize epipolar geometry.
    
    Args:
        img1: First image
        img2: Second image
        pts1: Points in first image (Nx2)
        pts2: Points in second image (Nx2)
        F: Fundamental matrix
        num_lines: Number of epipolar lines to draw
        
    Returns:
        Tuple of (img1_with_lines, img2_with_lines)
    """
    img1_vis = img1.copy()
    img2_vis = img2.copy()
    
    # Sample points to draw lines for
    if len(pts1) > num_lines:
        indices = np.linspace(0, len(pts1) - 1, num_lines, dtype=int)
        sample_pts1 = pts1[indices]
    else:
        sample_pts1 = pts1
    
    # Draw epipolar lines
    for pt in sample_pts1:
        # Epipolar line in second image: l2 = F * pt1
        pt_homogeneous = np.array([pt[0], pt[1], 1.0])
        line = F @ pt_homogeneous
        
        # Draw line in second image
        h, w = img2.shape[:2]
        if abs(line[1]) > 1e-6:  # Not vertical line
            y1 = int(-line[2] / line[1])
            y2 = int(-(line[0] * w + line[2]) / line[1])
            cv2.line(img2_vis, (0, y1), (w, y2), (0, 255, 0), 1)
        else:  # Vertical line
            x = int(-line[2] / line[0])
            cv2.line(img2_vis, (x, 0), (x, h), (0, 255, 0), 1)
        
        # Draw point in first image
        cv2.circle(img1_vis, tuple(pt.astype(int)), 5, (0, 0, 255), -1)
    
    return img1_vis, img2_vis


