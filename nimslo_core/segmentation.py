"""
Segmentation module for Nimslo images.

Uses U²-Net (via rembg) for salient object detection,
with optional depth-based fallback for difficult cases.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from PIL import Image
import io
import os
import warnings
from pathlib import Path
import urllib.request
import tarfile
import tempfile

# Configure OpenMP BEFORE importing rembg/onnxruntime
# This prevents the deprecated omp_set_nested warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # Limit threads to avoid conflicts
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'  # Use max_active_levels instead of nested

# Try to configure OpenMP programmatically if possible
try:
    import ctypes
    # Try to set max_active_levels directly if OpenMP is available
    try:
        libomp = ctypes.CDLL(None)
        if hasattr(libomp, 'omp_set_max_active_levels'):
            libomp.omp_set_max_active_levels(1)
    except:
        pass
except:
    pass

warnings.filterwarnings('ignore', message='.*omp_set_nested.*')
warnings.filterwarnings('ignore', message='.*omp_set_max_active_levels.*')

# Lazy imports for heavy dependencies
_rembg_session = None
_rembg_available = None

# OpenCV DNN model cache
_opencv_dnn_net = None
_opencv_dnn_available = None

# U-Net model cache
_unet_model = None
_unet_available = None
_unet_processor = None


def _check_rembg_available():
    """Check if rembg can be imported without crashing."""
    global _rembg_available
    if _rembg_available is None:
        try:
            import rembg
            _rembg_available = True
        except Exception as e:
            _rembg_available = False
            print(f"Warning: rembg not available: {e}")
    return _rembg_available


def _get_rembg_session():
    """Lazy-load rembg session to avoid startup overhead."""
    global _rembg_session
    if _rembg_session is None:
        if not _check_rembg_available():
            raise RuntimeError("rembg is not available - cannot perform U²-Net segmentation")
        try:
            from rembg import new_session
            # u2net is the default, good balance of quality and speed
            _rembg_session = new_session("u2net")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize rembg session: {e}")
    return _rembg_session


def segment_subject(
    img: np.ndarray,
    method: str = "u2net",
    return_confidence: bool = False
) -> np.ndarray | Tuple[np.ndarray, float]:
    """
    Segment the main subject from the background.
    
    Args:
        img: Input BGR image
        method: Segmentation method ("u2net", "unet", "depth", "opencv_dnn", "saliency", or "grabcut")
        return_confidence: Whether to return confidence score
        
    Returns:
        Binary mask (255 for subject, 0 for background)
        If return_confidence=True, returns (mask, confidence)
    """
    if method == "u2net":
        mask, confidence = _segment_u2net(img)
    elif method == "unet":
        mask, confidence = _segment_unet(img)
    elif method == "depth":
        mask, confidence = _segment_depth(img)
    elif method == "opencv_dnn":
        mask, confidence = _segment_opencv_dnn(img)
    elif method == "saliency":
        mask, confidence = _segment_saliency(img)
    elif method == "grabcut":
        mask, confidence = _segment_grabcut_improved(img)
    else:
        raise ValueError(f"Unknown segmentation method: {method}. Choose from: u2net, unet, depth, opencv_dnn, saliency, grabcut")
    
    if return_confidence:
        return mask, confidence
    return mask


def _segment_u2net(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Segment using U²-Net via rembg library.
    
    Returns:
        Tuple of (binary mask, confidence score)
    """
    try:
        from rembg import remove
    except ImportError as e:
        raise RuntimeError(f"rembg not available: {e}. Install with: pip install rembg onnxruntime")
    
    # Convert BGR to RGB for PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Get session and remove background
    try:
        session = _get_rembg_session()
    except Exception as e:
        raise RuntimeError(f"Failed to get rembg session: {e}")
    
    try:
        # Remove returns RGBA image with alpha channel as mask
        result = remove(pil_img, session=session, only_mask=True)
    except Exception as e:
        raise RuntimeError(f"rembg segmentation failed: {e}")
    
    # Convert mask to numpy
    mask = np.array(result)
    
    # Ensure binary mask
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate confidence based on mask properties
    confidence = _calculate_mask_confidence(binary_mask)
    
    return binary_mask, confidence


def _get_unet_model():
    """Lazy-load U-Net model for segmentation."""
    global _unet_model, _unet_processor, _unet_available
    
    if _unet_available is False:
        return None, None
    
    if _unet_model is None:
        try:
            # Try using segmentation_models_pytorch (most common)
            try:
                import segmentation_models_pytorch as smp
                import torch
                
                # Use a pre-trained U-Net with efficientnet encoder
                # This is lightweight and works well for person/object segmentation
                _unet_model = smp.Unet(
                    encoder_name="efficientnet-b0",  # Lightweight encoder
                    encoder_weights="imagenet",      # Pre-trained weights
                    in_channels=3,
                    classes=1,                       # Binary segmentation
                    activation=None,                 # Raw logits
                )
                _unet_model.eval()
                
                # Try to load pre-trained segmentation weights if available
                # Otherwise, we'll use the encoder weights only
                _unet_available = True
                _unet_processor = None  # No special processor needed
                return _unet_model, _unet_processor
                
            except ImportError:
                # Fallback: try using torchvision's DeepLabV3 (which uses similar architecture)
                # or a simple U-Net from scratch
                try:
                    import torch
                    import torchvision.transforms as transforms
                    from torchvision.models.segmentation import deeplabv3_resnet50
                    
                    # Use DeepLabV3 as a U-Net alternative (it's actually better for segmentation)
                    # Use weights parameter instead of deprecated pretrained
                    try:
                        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
                        _unet_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
                    except (ImportError, AttributeError):
                        # Fallback for older torchvision versions
                        _unet_model = deeplabv3_resnet50(pretrained=True)
                    _unet_model.eval()
                    
                    _unet_processor = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((520, 520)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )
                    ])
                    
                    _unet_available = True
                    return _unet_model, _unet_processor
                    
                except ImportError:
                    _unet_available = False
                    return None, None
                    
        except Exception as e:
            print(f"Warning: U-Net model not available: {e}")
            _unet_available = False
            return None, None
    
    return _unet_model, _unet_processor


def _segment_unet(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Segment using U-Net architecture.
    
    Uses segmentation_models_pytorch or torchvision DeepLabV3 as fallback.
    More accurate than depth-based methods for person/object segmentation.
    
    Returns:
        Tuple of (binary mask, confidence score)
    """
    model, processor = _get_unet_model()
    
    if model is None:
        # Fallback to saliency if U-Net not available
        return _segment_saliency(img)
    
    try:
        import torch
        from PIL import Image
        
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Check if using segmentation_models_pytorch or torchvision
        model_name = type(model).__name__.lower()
        
        if 'unet' in model_name or 'smp' in str(type(model)):
            # segmentation_models_pytorch U-Net
            # Prepare input
            import torchvision.transforms as transforms
            img_pil = Image.fromarray(img_rgb)
            img_tensor = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(img_pil).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                output = model(img_tensor)
                if isinstance(output, dict):
                    output = output['out']
                elif isinstance(output, tuple):
                    output = output[0]
                
                # Apply sigmoid for binary segmentation
                mask_logits = torch.sigmoid(output[0, 0])
                mask_np = (mask_logits.cpu().numpy() * 255).astype(np.uint8)
                
        else:
            # torchvision DeepLabV3 (U-Net-like architecture)
            import torchvision.transforms as transforms
            if processor is not None:
                img_tensor = processor(img_rgb).unsqueeze(0)
            else:
                # Default preprocessing
                img_pil = Image.fromarray(img_rgb)
                img_tensor = transforms.Compose([
                    transforms.Resize((520, 520)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])(img_pil).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                output = model(img_tensor)['out'][0]
                
                # Get person class (class 15 in COCO) or largest foreground class
                # For binary segmentation, use argmax and threshold
                probs = torch.softmax(output, dim=0)
                
                # Person class is typically 15 in COCO, but we'll use the largest non-background class
                foreground_probs = probs[1:].max(dim=0)[0]  # Max of all non-background classes
                mask_np = (foreground_probs.cpu().numpy() * 255).astype(np.uint8)
        
        # Resize mask back to original size
        mask = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Threshold to binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        confidence = _calculate_mask_confidence(mask)
        return mask, confidence
        
    except Exception as e:
        print(f"Warning: U-Net segmentation failed: {e}, falling back to saliency")
        return _segment_saliency(img)


def _segment_depth(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Segment using monocular depth estimation.
    
    Fallback method when U²-Net doesn't work well.
    Uses Intel DPT model for depth estimation.
    
    Returns:
        Tuple of (binary mask, confidence score)
    """
    try:
        from transformers import DPTImageProcessor, DPTForDepthEstimation
        import torch
    except ImportError:
        raise ImportError("Depth segmentation requires transformers and torch")
    
    # Load model (cached after first call)
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
    
    # Prepare image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = processor(images=img_rgb, return_tensors="pt")
    
    # Get depth prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    )
    
    # Convert to numpy and normalize
    depth = prediction.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = (depth * 255).astype(np.uint8)
    
    # Threshold using Otsu's method (foreground is typically closer/brighter)
    _, mask = cv2.threshold(depth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    confidence = _calculate_mask_confidence(mask)
    
    return mask, confidence


def _download_opencv_dnn_model(model_dir: Optional[Path] = None) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Download and extract DeepLabV3 MobileNetV2 model for OpenCV DNN.
    
    Args:
        model_dir: Directory to save model files (default: temp directory)
        
    Returns:
        Tuple of (model_pb_path, model_pbtxt_path) or (None, None) if download fails
    """
    if model_dir is None:
        # Use a cache directory in the project or temp
        model_dir = Path.home() / ".nimslo_models"
    else:
        model_dir = Path(model_dir)
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_pb = model_dir / "frozen_inference_graph.pb"
    model_pbtxt = model_dir / "frozen_inference_graph.pbtxt"
    
    # Check if model already exists
    if model_pb.exists() and model_pbtxt.exists():
        return model_pb, model_pbtxt
    
    # Try to download model
    model_url = "http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz"
    tar_path = model_dir / "deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz"
    
    try:
        print(f"Downloading DeepLabV3 model from {model_url}...")
        urllib.request.urlretrieve(model_url, tar_path)
        
        # Extract frozen graph
        with tarfile.open(tar_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if 'frozen_inference_graph.pb' in member.name:
                    tar.extract(member, model_dir)
                    extracted_pb = model_dir / Path(member.name).name
                    if extracted_pb != model_pb:
                        extracted_pb.rename(model_pb)
        
        # Generate pbtxt if needed (OpenCV DNN can work with just .pb for some models)
        # For DeepLabV3, we may need to create a simple pbtxt or use readNetFromTensorflow
        # For now, return the .pb file and None for pbtxt
        if model_pb.exists():
            return model_pb, None
        
    except Exception as e:
        print(f"Warning: Could not download OpenCV DNN model: {e}")
        return None, None
    
    return None, None


def _get_opencv_dnn_net():
    """Lazy-load OpenCV DNN network."""
    global _opencv_dnn_net, _opencv_dnn_available
    
    if _opencv_dnn_available is False:
        return None
    
    if _opencv_dnn_net is None:
        try:
            model_pb, model_pbtxt = _download_opencv_dnn_model()
            
            if model_pb is None:
                _opencv_dnn_available = False
                return None
            
            # Try to load the model
            if model_pbtxt is not None:
                _opencv_dnn_net = cv2.dnn.readNetFromTensorflow(str(model_pb), str(model_pbtxt))
            else:
                # Try loading with just .pb file (some models work this way)
                try:
                    _opencv_dnn_net = cv2.dnn.readNetFromTensorflow(str(model_pb))
                except:
                    # If that fails, try using a generated pbtxt
                    # For DeepLabV3, we can create a minimal pbtxt
                    _opencv_dnn_available = False
                    return None
            
            _opencv_dnn_available = True
        except Exception as e:
            print(f"Warning: OpenCV DNN model not available: {e}")
            _opencv_dnn_available = False
            return None
    
    return _opencv_dnn_net


def _segment_opencv_dnn(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Segment using OpenCV DNN with DeepLabV3.
    
    This is a lightweight alternative that avoids onnxruntime.
    
    Returns:
        Tuple of (binary mask, confidence score)
    """
    net = _get_opencv_dnn_net()
    
    if net is None:
        # Fallback to saliency if model not available
        return _segment_saliency(img)
    
    try:
        h, w = img.shape[:2]
        
        # DeepLabV3 expects input size 513x513
        input_size = 513
        blob = cv2.dnn.blobFromImage(img, 1.0/127.5, (input_size, input_size), 
                                     (127.5, 127.5, 127.5), swapRB=True, crop=False)
        
        net.setInput(blob)
        output = net.forward()
        
        # Output shape: (1, num_classes, height, width)
        # Get class predictions (assuming class 0 is background, class 15 is person/foreground)
        predictions = np.argmax(output[0], axis=0)
        
        # Create mask: foreground classes (typically 15 for person, or we can use all non-background)
        # For Pascal VOC: 0=background, 15=person, others are various objects
        # We'll treat person and common foreground objects as foreground
        foreground_classes = [15]  # Person class in Pascal VOC
        mask = np.zeros(predictions.shape, dtype=np.uint8)
        for cls in foreground_classes:
            mask[predictions == cls] = 255
        
        # If no person found, use largest non-background region
        if np.sum(mask) == 0:
            # Find largest non-background component
            for cls in range(1, predictions.max() + 1):
                cls_mask = (predictions == cls).astype(np.uint8) * 255
                if np.sum(cls_mask) > np.sum(mask):
                    mask = cls_mask
        
        # Resize mask back to original image size
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        confidence = _calculate_mask_confidence(mask)
        return mask, confidence
        
    except Exception as e:
        print(f"Warning: OpenCV DNN segmentation failed: {e}, falling back to saliency")
        return _segment_saliency(img)


def _segment_saliency(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Segment using OpenCV's built-in saliency detection.
    
    This is a lightweight method that requires no external models.
    Uses fine-grained saliency detection to find the main subject.
    
    Returns:
        Tuple of (binary mask, confidence score)
    """
    try:
        # Try fine-grained saliency (more accurate but slower)
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        success, saliency_map = saliency.computeSaliency(img)
        
        if not success:
            # Fallback to spectral residual
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliency_map = saliency.computeSaliency(img)
        
        if not success:
            # Last resort: use center-weighted approach
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            y1, y2 = int(h*0.2), int(h*0.8)
            x1, x2 = int(w*0.2), int(w*0.8)
            mask[y1:y2, x1:x2] = 255
            return mask, 0.3
        
        # Convert saliency map to binary mask
        saliency_map = (saliency_map * 255).astype(np.uint8)
        
        # Use Otsu's method to threshold
        _, mask = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        confidence = _calculate_mask_confidence(mask)
        return mask, confidence
        
    except Exception as e:
        # Fallback to center mask
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        y1, y2 = int(h*0.2), int(h*0.8)
        x1, x2 = int(w*0.2), int(w*0.8)
        mask[y1:y2, x1:x2] = 255
        return mask, 0.3


def _segment_grabcut_improved(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Segment using GrabCut with saliency-based initialization.
    
    This improves upon the simple center mask by using saliency
    to initialize GrabCut, resulting in better segmentation.
    
    Returns:
        Tuple of (binary mask, confidence score)
    """
    # Get initial mask from saliency
    initial_mask, _ = _segment_saliency(img)
    
    # Refine with GrabCut
    refined_mask = refine_mask_grabcut(img, initial_mask, iterations=5)
    
    confidence = _calculate_mask_confidence(refined_mask)
    return refined_mask, confidence


def _calculate_mask_confidence(mask: np.ndarray) -> float:
    """
    Calculate confidence score for a segmentation mask.
    
    Based on:
    - Area ratio (subject should cover 10-40% of image)
    - Compactness (well-defined subjects have good area/perimeter ratio)
    
    Returns:
        Confidence score between 0 and 1
    """
    h, w = mask.shape[:2]
    total_pixels = h * w
    
    # Calculate area ratio
    mask_pixels = np.sum(mask > 0)
    area_ratio = mask_pixels / total_pixels
    
    # Ideal range is 10-40% of image
    if 0.1 <= area_ratio <= 0.4:
        area_score = 1.0
    elif area_ratio < 0.1:
        area_score = area_ratio / 0.1
    elif area_ratio > 0.4:
        area_score = max(0, 1 - (area_ratio - 0.4) / 0.3)
    else:
        area_score = 0.5
    
    # Calculate compactness
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter ** 2)
            # Normalize compactness (circle = 1, more complex = lower)
            compactness_score = min(compactness * 2, 1.0)
        else:
            compactness_score = 0.5
    else:
        compactness_score = 0.0
    
    # Combined confidence
    confidence = 0.6 * area_score + 0.4 * compactness_score
    
    return confidence


def get_segmentation_mask(
    img: np.ndarray,
    confidence_threshold: float = 0.5,
    fallback_to_depth: bool = True,
    use_subprocess: bool = True,
    prefer_lightweight: bool = True
) -> Tuple[np.ndarray, float, str]:
    """
    Get the best segmentation mask, with automatic method selection.
    
    Tries lightweight methods first (opencv_dnn, saliency) to avoid kernel crashes,
    falls back to U²-Net or depth-based if needed.
    
    Args:
        img: Input BGR image
        confidence_threshold: Minimum confidence to accept result
        fallback_to_depth: Whether to try depth method if other methods fail
        use_subprocess: Use subprocess isolation for rembg (prevents kernel crashes)
        prefer_lightweight: If True, prefer lightweight methods (opencv_dnn, saliency) over rembg
        
    Returns:
        Tuple of (mask, confidence, method_used)
    """
    mask = None
    confidence = 0.0
    method_used = "fallback"
    
    # Try lightweight methods first (safe for Jupyter)
    if prefer_lightweight:
        # Try OpenCV DNN (lightweight, no onnxruntime)
        try:
            mask, confidence = segment_subject(img, method="opencv_dnn", return_confidence=True)
            if confidence >= confidence_threshold:
                return mask, confidence, "opencv_dnn"
            method_used = "opencv_dnn"
        except Exception:
            pass
        
        # Try saliency-based (pure OpenCV, very lightweight)
        try:
            saliency_mask, saliency_conf = segment_subject(img, method="saliency", return_confidence=True)
            if saliency_conf > confidence:
                mask = saliency_mask
                confidence = saliency_conf
                method_used = "saliency"
        except Exception:
            pass
        
        # Try improved GrabCut (uses saliency for initialization)
        try:
            grabcut_mask, grabcut_conf = segment_subject(img, method="grabcut", return_confidence=True)
            if grabcut_conf > confidence:
                mask = grabcut_mask
                confidence = grabcut_conf
                method_used = "grabcut"
        except Exception:
            pass
    
    # If lightweight methods didn't meet threshold, try heavier methods
    if confidence < confidence_threshold:
        # Try U-Net (better than depth-based, uses PyTorch)
        try:
            unet_mask, unet_conf = segment_subject(img, method="unet", return_confidence=True)
            if unet_conf > confidence:
                mask = unet_mask
                confidence = unet_conf
                method_used = "unet"
        except Exception:
            pass
        
        # Try U²-Net first (with subprocess isolation if requested)
        if use_subprocess:
            try:
                from .segmentation_subprocess import segment_u2net_subprocess
                u2net_mask, u2net_conf = segment_u2net_subprocess(img)
                if u2net_conf > confidence:
                    mask = u2net_mask
                    confidence = u2net_conf
                    method_used = "u2net_subprocess"
            except Exception:
                pass
        
        # Try direct U²-Net (may crash kernel in Jupyter)
        try:
            u2net_mask, u2net_conf = segment_subject(img, method="u2net", return_confidence=True)
            if u2net_conf > confidence:
                mask = u2net_mask
                confidence = u2net_conf
                method_used = "u2net"
        except Exception:
            pass
        
        # Fall back to depth-based segmentation (last resort)
        if fallback_to_depth:
            try:
                depth_mask, depth_conf = segment_subject(img, method="depth", return_confidence=True)
                if depth_conf > confidence:
                    mask = depth_mask
                    confidence = depth_conf
                    method_used = "depth"
            except Exception:
                pass
    
    # Last resort: center mask
    if mask is None or confidence < 0.2:
        h, w = img.shape[:2]
        fallback_mask = np.zeros((h, w), dtype=np.uint8)
        y1, y2 = int(h*0.2), int(h*0.8)
        x1, x2 = int(w*0.2), int(w*0.8)
        fallback_mask[y1:y2, x1:x2] = 255
        return fallback_mask, 0.3, "fallback"
    
    return mask, confidence, method_used


def refine_mask_grabcut(
    img: np.ndarray,
    initial_mask: np.ndarray,
    iterations: int = 5
) -> np.ndarray:
    """
    Refine a segmentation mask using GrabCut.
    
    Args:
        img: Input BGR image
        initial_mask: Initial binary mask (255=foreground, 0=background)
        iterations: Number of GrabCut iterations
        
    Returns:
        Refined binary mask
    """
    # Create GrabCut mask
    gc_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    gc_mask[initial_mask > 127] = cv2.GC_PR_FGD  # Probable foreground
    gc_mask[initial_mask <= 127] = cv2.GC_PR_BGD  # Probable background
    
    # Erode mask to get definite foreground
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    definite_fg = cv2.erode(initial_mask, kernel, iterations=2)
    gc_mask[definite_fg > 127] = cv2.GC_FGD
    
    # Dilate inverse mask to get definite background
    definite_bg = cv2.dilate(initial_mask, kernel, iterations=2)
    gc_mask[definite_bg == 0] = cv2.GC_BGD
    
    # Apply GrabCut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(img, gc_mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        # GrabCut can fail on certain images, return original mask
        return initial_mask
    
    # Extract foreground
    output_mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255, 0
    ).astype(np.uint8)
    
    return output_mask


def visualize_mask(
    img: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Overlay segmentation mask on image for visualization.
    
    Args:
        img: Input BGR image
        mask: Binary mask
        alpha: Transparency of overlay
        color: BGR color for mask overlay
        
    Returns:
        Image with mask overlay
    """
    overlay = img.copy()
    mask_bool = mask > 127
    
    # Apply color to mask region
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + 
        np.array(color) * alpha
    ).astype(np.uint8)
    
    return overlay

