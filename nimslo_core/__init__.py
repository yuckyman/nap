"""
Nimslo Core Library
===================
Shared modules for Nimslo 4-lens camera image alignment.

Modules:
    - preprocessing: Film grain reduction and exposure balancing
    - segmentation: U²-Net based subject detection
    - alignment: SIFT feature matching and homography estimation
    - gif_generator: Boomerang GIF creation
"""

from .preprocessing import preprocess_image, denoise_film_grain, balance_exposure
from .alignment import align_images, extract_features, match_features
from .gif_generator import create_boomerang_gif, make_boomerang_frames, encode_gif, encode_mp4

# DO NOT import segmentation here - it causes kernel crashes
# Import it directly when needed: from nimslo_core.segmentation import segment_subject

__version__ = "0.1.0"
__all__ = [
    "preprocess_image",
    "denoise_film_grain", 
    "balance_exposure",
    "align_images",
    "extract_features",
    "match_features",
    "create_boomerang_gif",
    "make_boomerang_frames",
    "encode_gif",
    "encode_mp4",
]
