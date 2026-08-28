"""
Segmentation module for Nimslo images.

Uses U²-Net (via rembg) for salient object detection.
U²-Net or bust - no fallbacks.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
from PIL import Image
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
import urllib.request
import tarfile
import tempfile

U2NET_MEAN = (0.485, 0.456, 0.406)
U2NET_STD = (0.229, 0.224, 0.225)
U2NET_INPUT_SIZE = (320, 320)


def _is_notebook_kernel() -> bool:
    """True in Jupyter, IPython, or local marimo kernels (not CLI)."""
    if "marimo" in sys.modules:
        return True
    if os.environ.get("MARIMO_APP_ROOT") or os.environ.get("MARIMO_BRANCH"):
        return True
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            shell_name = ip.__class__.__name__
            if shell_name in ("ZMQInteractiveShell", "GoogleColabShell", "TerminalInteractiveShell"):
                return True
    except ImportError:
        pass
    return False


def configure_openmp(force_single_thread: Optional[bool] = None) -> None:
    """
    Limit OpenMP threads in notebook kernels to avoid onnx/jupyter crashes.

    CLI runs use available CPU cores. Marimo WASM in the browser does not
    import this module.
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if force_single_thread is None:
        force_single_thread = _is_notebook_kernel()
    if force_single_thread:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
        try:
            import ctypes
            libomp = ctypes.CDLL(None)
            if hasattr(libomp, "omp_set_max_active_levels"):
                libomp.omp_set_max_active_levels(1)
        except Exception:
            pass
    else:
        n_threads = str(os.cpu_count() or 4)
        os.environ["OMP_NUM_THREADS"] = n_threads
        os.environ["OPENBLAS_NUM_THREADS"] = n_threads
        os.environ["MKL_NUM_THREADS"] = n_threads
        os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"


configure_openmp()

warnings.filterwarnings("ignore", message=".*omp_set_nested.*")
warnings.filterwarnings("ignore", message=".*omp_set_max_active_levels.*")

# Lazy imports for heavy dependencies
_rembg_sessions: Dict[Tuple[str, str], object] = {}
_rembg_available = None

# OpenCV DNN model cache
_opencv_dnn_net = None
_opencv_dnn_available = None

# U-Net model cache
_unet_model = None
_unet_available = None
_unet_processor = None


@dataclass(frozen=True)
class SegmentationOptions:
    """Tunable segmentation path for benchmarks and production."""
    model: str = "u2net"
    max_dimension: Optional[int] = None
    parallel: bool = False
    use_coreml: bool = False
    force_omp_single_thread: Optional[bool] = None


# Production-fast path: u2net (not u2netp) keeps alignment/framing stable.
FAST_SEGMENTATION = SegmentationOptions(
    model="u2net",
    max_dimension=1024,
    parallel=True,
    use_coreml=True,
    force_omp_single_thread=False,
)


def _check_rembg_available():
    """Check if rembg can be imported without crashing."""
    global _rembg_available
    if _rembg_available is None:
        try:
            import rembg  # noqa: F401
            _rembg_available = True
        except Exception as e:
            _rembg_available = False
            print(f"Warning: rembg not available: {e}")
    return _rembg_available


def _session_cache_key(model: str, use_coreml: bool) -> Tuple[str, str]:
    if use_coreml:
        return model, "coreml"
    return model, "cpu"


def _get_rembg_session(model: str = "u2net", use_coreml: bool = False):
    """Lazy-load rembg session keyed by model and execution provider."""
    key = _session_cache_key(model, use_coreml)
    if key in _rembg_sessions:
        return _rembg_sessions[key]

    if not _check_rembg_available():
        raise RuntimeError("rembg is not available - cannot perform U²-Net segmentation")

    try:
        import onnxruntime as ort
        from rembg import new_session

        kwargs = {}
        if use_coreml and "CoreMLExecutionProvider" in ort.get_available_providers():
            kwargs["providers"] = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        _rembg_sessions[key] = new_session(model, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize rembg session ({model}): {e}")
    return _rembg_sessions[key]


def _legacy_get_rembg_session():
    """Backwards-compatible default session accessor."""
    return _get_rembg_session("u2net", use_coreml=False)


def _prepare_inference_image(
    img: np.ndarray,
    max_dimension: Optional[int],
) -> Tuple[Image.Image, Tuple[int, int]]:
    """Optionally downscale, return PIL RGB and original (w, h) for mask upscale."""
    h, w = img.shape[:2]
    orig_size = (w, h)
    work = img
    if max_dimension is not None:
        scale = min(max_dimension / max(h, w), 1.0)
        if scale < 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            work = cv2.resize(work, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), orig_size


def _predict_mask_pil(pil_img: Image.Image, session) -> Image.Image:
    """Run u²-net via rembg session and return grayscale mask PIL image."""
    return session.predict(pil_img)[0]


def _mask_pil_to_binary(mask_pil: Image.Image, orig_size: Tuple[int, int]) -> np.ndarray:
    mask = np.array(mask_pil)
    if mask_pil.size != orig_size:
        mask = cv2.resize(mask, orig_size, interpolation=cv2.INTER_LINEAR)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask


def segment_images(
    images: List[np.ndarray],
    options: Optional[SegmentationOptions] = None,
) -> List[Tuple[np.ndarray, float]]:
    """
    Segment one or more frames with shared session and optional parallelism.

    Note: exported u²-net onnx graphs are fixed batch=1, so ``parallel=True``
    runs concurrent single-frame inferences on a thread-safe shared session.
    """
    if not images:
        return []

    opts = options or SegmentationOptions()
    configure_openmp(opts.force_omp_single_thread)
    session = _get_rembg_session(opts.model, opts.use_coreml)
    prepared = [_prepare_inference_image(img, opts.max_dimension) for img in images]

    if opts.parallel and len(prepared) > 1:
        with ThreadPoolExecutor(max_workers=len(prepared)) as executor:
            mask_pils = list(
                executor.map(lambda item: _predict_mask_pil(item[0], session), prepared)
            )
    else:
        mask_pils = [_predict_mask_pil(pil_img, session) for pil_img, _ in prepared]

    results = []
    for (_, orig_size), mask_pil in zip(prepared, mask_pils):
        binary_mask = _mask_pil_to_binary(mask_pil, orig_size)
        confidence = _calculate_mask_confidence(binary_mask)
        results.append((binary_mask, confidence))
    return results


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


def _segment_u2net(
    img: np.ndarray,
    options: Optional[SegmentationOptions] = None,
) -> Tuple[np.ndarray, float]:
    """
    Segment using U²-Net via rembg library.

    Returns:
        Tuple of (binary mask, confidence score)
    """
    mask, confidence = segment_images([img], options)[0]
    return mask, confidence


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
    options: Optional[SegmentationOptions] = None,
) -> Tuple[np.ndarray, float, str]:
    """
    Get segmentation mask using U²-Net (via rembg).

    Args:
        img: Input BGR image
        options: Optional segmentation tuning (model, downscale, parallel, coreml)

    Returns:
        Tuple of (mask, confidence, method_used)

    Raises:
        RuntimeError: If U²-Net/rembg is not available
    """
    opts = options or SegmentationOptions()
    mask, confidence = _segment_u2net(img, opts)
    return mask, confidence, opts.model


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

