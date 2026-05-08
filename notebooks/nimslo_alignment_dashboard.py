# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "opencv-python",
#     "pillow",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell
def _():
    import io
    from dataclasses import dataclass
    from pathlib import Path
    from typing import List, Optional, Tuple

    import cv2
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    return Image, Path, cv2, dataclass, io, mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # nimslo alignment pipeline

    a hosted walkthrough of how the nimslo alignment pipeline turns real four-lens
    film scans into a stable wigglegram: preprocess, segment the subject, center
    on the mask, match features, estimate transforms, then render the original
    grain-preserving scans.
    """)
    return


@app.cell
def _(Path, mo):
    SAMPLE_SCAN_GROUPS = {
        "web-scans/44": [
            "723594009607-R1-064-30A.jpg",
            "723594009607-R1-065-31.jpg",
            "723594009607-R1-066-31A.jpg",
            "723594009607-R1-067-32.jpg",
        ],
        "web-scans/45": [
            "723594009607-R1-068-32A.jpg",
            "723594009607-R1-069-33.jpg",
            "723594009607-R1-070-33A.jpg",
            "723594009607-R1-071-34.jpg",
        ],
        "web-scans/49": [
            "378424001385-R1-000-19A-20.jpg",
            "378424001385-R1-000-20-20A.jpg",
            "378424001385-R1-000-20A-21.jpg",
            "378424001385-R1-000-21-21A.jpg",
        ],
        "web-scans/56": [
            "378424001387-R1-059-28.jpg",
            "378424001387-R1-060-28A.jpg",
            "378424001387-R1-061-29.jpg",
            "378424001387-R1-062-29A.jpg",
        ],
        "web-scans/60": [
            "378424001385-R1-014-5A.jpg",
            "378424001385-R1-015-6.jpg",
            "378424001385-R1-016-6A.jpg",
            "378424001385-R1-017-7.jpg",
        ],
        "web-scans/74": [
            "213547008873-R1-025.jpg",
            "213547008873-R1-026.jpg",
            "213547008873-R1-027.jpg",
            "213547008873-R1-028.jpg",
        ],
    }

    notebook_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    scan_root_candidates = [
        notebook_dir / "web-scans",
        notebook_dir / "scans",
        Path("notebooks/web-scans"),
        Path("notebooks/scans"),
    ]
    scans_root = next((path for path in scan_root_candidates if path.exists()), scan_root_candidates[0])
    if scans_root.exists():
        scan_folders = [
            str(path)
            for path in sorted(scans_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name)
            if path.is_dir()
        ]
    else:
        scan_folders = list(SAMPLE_SCAN_GROUPS)
    selected_folder = mo.ui.dropdown(
        options=scan_folders,
        value=scan_folders[0] if scan_folders else None,
        label="scan folder",
    )
    quality = mo.ui.dropdown(
        options=["fast", "balanced", "best"],
        value="balanced",
        label="quality preset",
    )
    segmentation = mo.ui.dropdown(
        options=["saliency", "grabcut"],
        value="grabcut",
        label="segmentation method",
    )
    transform_model = mo.ui.dropdown(
        options=["translation", "affine"],
        value="translation",
        label="alignment model",
    )
    max_dimension = mo.ui.slider(
        600,
        1800,
        value=1000,
        step=100,
        label="browser processing max dimension",
    )
    show_features = mo.ui.checkbox(value=True, label="show feature matches")
    mo.vstack(
        [
            mo.hstack([selected_folder, quality, segmentation, transform_model, max_dimension, show_features], wrap=True),
        ]
    )
    return (
        max_dimension,
        quality,
        SAMPLE_SCAN_GROUPS,
        segmentation,
        selected_folder,
        show_features,
        transform_model,
    )


@app.cell
def _(cv2, dataclass, np):
    @dataclass
    class AlignmentMetrics:
        frame: int
        keypoints_source: int
        keypoints_reference: int
        matches: int
        inliers: int
        iou: float
        confidence: float
        tx: float
        ty: float

    def bgr_to_rgb(img):
        return img[:, :, ::-1] if img.ndim == 3 else img

    def normalize_sizes(images):
        min_h = min(img.shape[0] for img in images)
        min_w = min(img.shape[1] for img in images)
        normalized = []
        for img in images:
            h, w = img.shape[:2]
            y = (h - min_h) // 2
            x = (w - min_w) // 2
            normalized.append(img[y : y + min_h, x : x + min_w])
        return normalized

    def resize_for_browser(images, max_dimension):
        resized = []
        for img in images:
            h, w = img.shape[:2]
            scale = min(max_dimension / max(h, w), 1.0)
            if scale < 1.0:
                resized.append(
                    cv2.resize(
                        img,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                )
            else:
                resized.append(img.copy())
        return resized

    def mask_centroid(mask):
        ys, xs = np.where(mask > 127)
        if len(xs) == 0:
            h, w = mask.shape[:2]
            return w // 2, h // 2
        return int(xs.mean()), int(ys.mean())

    return (
        AlignmentMetrics,
        bgr_to_rgb,
        mask_centroid,
        normalize_sizes,
        resize_for_browser,
    )


@app.cell
def _(Path, SAMPLE_SCAN_GROUPS, cv2, np):
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    RAW_SCAN_BASE = "https://raw.githubusercontent.com/yuckyman/nap/main/notebooks"

    def fetch_bytes(url):
        try:
            from pyodide.http import open_url

            data = open_url(url).read()
        except Exception:
            from urllib.request import urlopen

            with urlopen(url) as response:
                data = response.read()
        if isinstance(data, str):
            data = data.encode("latin-1")
        return data

    def decode_image(data):
        array = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)

    def load_remote_scans(group):
        frames = []
        names = []
        for name in SAMPLE_SCAN_GROUPS.get(group, [])[:4]:
            url = f"{RAW_SCAN_BASE}/{group}/{name}"
            img = decode_image(fetch_bytes(url))
            if img is not None:
                frames.append(img)
                names.append(name)
        return frames, names

    def load_directory_scans(directory):
        if not directory:
            return [], []
        path = Path(directory).expanduser()
        if not path.exists() or not path.is_dir():
            return load_remote_scans(str(directory))
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        frames = []
        names = []
        for file_path in files[:4]:
            img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
                names.append(file_path.name)
        if len(frames) < 3 and str(directory) in SAMPLE_SCAN_GROUPS:
            return load_remote_scans(str(directory))
        return frames, names

    return (load_directory_scans,)


@app.cell
def _(cv2, np):
    def preprocess_image(img, denoise=True, balance=True):
        result = img.copy()
        if denoise:
            result = cv2.fastNlMeansDenoisingColored(result, h=8, hColor=8, templateWindowSize=7, searchWindowSize=21)
        if balance:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return result

    def segment_saliency(img):
        if hasattr(cv2, "saliency"):
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            ok, saliency_map = saliency.computeSaliency(img)
            if not ok:
                return np.zeros(img.shape[:2], dtype=np.uint8)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            saliency_map = cv2.Laplacian(gray, cv2.CV_32F)
            saliency_map = cv2.convertScaleAbs(saliency_map)
        saliency_map = (saliency_map * 255).astype(np.uint8)
        saliency_map = cv2.GaussianBlur(saliency_map, (11, 11), 0)
        _, mask = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))

    def segment_grabcut(img):
        h, w = img.shape[:2]
        rect = (w // 5, h // 8, int(w * 0.6), int(h * 0.78))
        gc_mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        return np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    return preprocess_image, segment_grabcut, segment_saliency


@app.cell
def _(AlignmentMetrics, cv2, mask_centroid, np):
    def center_on_subject(images, masks):
        h, w = images[0].shape[:2]
        target = mask_centroid(masks[0])
        centered_images = []
        centered_masks = []
        transforms = []
        for img, mask in zip(images, masks):
            cx, cy = mask_centroid(mask)
            tx, ty = target[0] - cx, target[1] - cy
            matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            full = np.vstack([matrix, [0, 0, 1]])
            centered_images.append(cv2.warpAffine(img, matrix, (w, h), borderValue=0))
            centered_masks.append(cv2.warpAffine(mask, matrix, (w, h), borderValue=0))
            transforms.append(full)
        return centered_images, centered_masks, transforms

    def iou(mask_a, mask_b):
        inter = np.logical_and(mask_a > 127, mask_b > 127).sum()
        union = np.logical_or(mask_a > 127, mask_b > 127).sum()
        return float(inter / union) if union else 0.0

    def align_to_reference(images, masks, model="translation", n_features=1000):
        ref_img, ref_mask = images[0], masks[0]
        sift = cv2.SIFT_create(nfeatures=n_features)
        kp_ref, des_ref = sift.detectAndCompute(cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY), ref_mask)
        aligned = [ref_img.copy()]
        transforms = [np.eye(3)]
        metrics = [AlignmentMetrics(1, len(kp_ref), len(kp_ref), 0, 0, 1.0, 1.0, 0.0, 0.0)]
        match_debug = None
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

        for idx, (img, mask) in enumerate(zip(images[1:], masks[1:]), start=2):
            kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask)
            good = []
            if des is not None and des_ref is not None and len(des) > 2 and len(des_ref) > 2:
                for pair in matcher.knnMatch(des, des_ref, k=2):
                    if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                        good.append(pair[0])
            if idx == 2:
                match_debug = (img, ref_img, kp, kp_ref, sorted(good, key=lambda m: m.distance)[:60])

            transform = None
            inliers = 0
            if len(good) >= 3:
                src = np.float32([kp[m.queryIdx].pt for m in good])
                dst = np.float32([kp_ref[m.trainIdx].pt for m in good])
                if model == "translation":
                    deltas = dst - src
                    tx, ty = np.median(deltas[:, 0]), np.median(deltas[:, 1])
                    err = np.linalg.norm((src + [tx, ty]) - dst, axis=1)
                    keep = err < 5.0
                    inliers = int(keep.sum())
                    if inliers >= 3:
                        tx, ty = np.median(deltas[keep, 0]), np.median(deltas[keep, 1])
                        transform = np.float32([[1, 0, tx], [0, 1, ty]])
                else:
                    transform, keep = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                    inliers = int(keep.sum()) if keep is not None else 0

            if transform is None:
                transform = np.float32([[1, 0, 0], [0, 1, 0]])
            h, w = ref_img.shape[:2]
            warped = cv2.warpAffine(img, transform, (w, h))
            warped_mask = cv2.warpAffine(mask, transform, (w, h))
            score = iou(warped_mask, ref_mask)
            full = np.vstack([transform, [0, 0, 1]])
            aligned.append(warped)
            transforms.append(full)
            match_count = len(good)
            confidence = 0.5 * (inliers / match_count if match_count else 0.0) + 0.5 * score
            metrics.append(
                AlignmentMetrics(idx, len(kp), len(kp_ref), match_count, inliers, score, confidence, full[0, 2], full[1, 2])
            )
        return aligned, transforms, metrics, match_debug

    return align_to_reference, center_on_subject


@app.cell
def _(cv2, np):
    def crop_valid_region(images):
        valid = None
        for img in images:
            frame_valid = np.any(img > 8, axis=2)
            valid = frame_valid if valid is None else np.logical_and(valid, frame_valid)
        ys, xs = np.where(valid)
        if len(xs) == 0:
            return images
        return [img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] for img in images]

    def normalize_brightness(images, strength=0.5):
        labs = [cv2.cvtColor(img, cv2.COLOR_BGR2LAB) for img in images]
        medians = [float(np.median(lab[:, :, 0])) for lab in labs]
        target = float(np.median(medians))
        normalized = []
        for img, lab, median in zip(images, labs, medians):
            shift = (target - median) * strength
            lab = lab.copy()
            lab[:, :, 0] = np.clip(lab[:, :, 0].astype(np.float32) + shift, 0, 255).astype(np.uint8)
            normalized.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
        return normalized

    def make_boomerang(images):
        frames = crop_valid_region(images)
        frames = normalize_brightness(frames)
        return frames + frames[-2:0:-1]

    return (make_boomerang,)


@app.cell
def _(
    load_directory_scans,
    max_dimension,
    mo,
    resize_for_browser,
    selected_folder,
):
    full_resolution_frames, frame_names = load_directory_scans(selected_folder.value)
    mo.stop(
        len(full_resolution_frames) < 3,
        mo.md(
            """
            ## add real nimslo scans

            add numbered folders under `notebooks/scans`, each containing 3 or 4 frame scans.
            this dashboard intentionally uses curated scans instead of fabricated samples.
            """
        ),
    )
    full_resolution_frames = full_resolution_frames[:4]
    original_frames = resize_for_browser(full_resolution_frames, max_dimension.value)
    frame_names = frame_names[:4]
    return frame_names, original_frames


@app.cell
def _(mo):
    mo.md("""
    ## 1. input frames
    the source material is the actual nimslo frame sequence. for browser responsiveness,
    this dashboard runs the walkthrough on a scaled working copy.
    """)
    return


@app.cell
def _(bgr_to_rgb, frame_names, np, original_frames, plt):
    fig_input, _axes_input = plt.subplots(1, len(original_frames), figsize=(3.6 * len(original_frames), 4))
    for _i, _ax in enumerate(np.ravel(_axes_input)):
        _ax.imshow(bgr_to_rgb(original_frames[_i]))
        _ax.set_title(frame_names[_i] if _i < len(frame_names) else f"frame {_i + 1}")
        _ax.axis("off")
    fig_input.tight_layout()
    fig_input
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. preprocess
    denoising and clahe exposure balancing make feature extraction and masks more stable. the final render still uses the original scans.
    """)
    return


@app.cell
def _(normalize_sizes, original_frames, preprocess_image, quality):
    settings = {
        "fast": {"denoise": False, "n_features": 500},
        "balanced": {"denoise": True, "n_features": 1000},
        "best": {"denoise": True, "n_features": 2000},
    }[quality.value]
    preprocessed = normalize_sizes([preprocess_image(frame, denoise=settings["denoise"]) for frame in original_frames])
    return preprocessed, settings


@app.cell
def _(bgr_to_rgb, original_frames, plt, preprocessed):
    fig_pre, _axes_pre = plt.subplots(1, 2, figsize=(10, 4))
    _axes_pre[0].imshow(bgr_to_rgb(original_frames[0]))
    _axes_pre[0].set_title("before")
    _axes_pre[0].axis("off")
    _axes_pre[1].imshow(bgr_to_rgb(preprocessed[0]))
    _axes_pre[1].set_title("after")
    _axes_pre[1].axis("off")
    fig_pre.tight_layout()
    fig_pre
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. segment the subject
    the production pipeline uses u2-net through `rembg`; this hosted notebook uses browser-friendly opencv segmentation so the walkthrough can run without model downloads.
    """)
    return


@app.cell
def _(preprocessed, segment_grabcut, segment_saliency, segmentation):
    if segmentation.value == "saliency":
        masks = [segment_saliency(frame) for frame in preprocessed]
    else:
        masks = [segment_grabcut(frame) for frame in preprocessed]
    return (masks,)


@app.cell
def _(bgr_to_rgb, masks, np, plt, preprocessed):
    fig_masks, _axes_masks = plt.subplots(1, len(preprocessed), figsize=(3.6 * len(preprocessed), 4))
    for _i, _ax in enumerate(np.ravel(_axes_masks)):
        overlay = preprocessed[_i].copy()
        overlay[masks[_i] > 127] = (0.55 * overlay[masks[_i] > 127] + 0.45 * np.array([0, 220, 60])).astype(np.uint8)
        _ax.imshow(bgr_to_rgb(overlay))
        _ax.set_title(f"mask {_i + 1}")
        _ax.axis("off")
    fig_masks.tight_layout()
    fig_masks
    return


@app.cell
def _(center_on_subject, masks, preprocessed):
    centered_frames, centered_masks, center_transforms = center_on_subject(preprocessed, masks)
    return center_transforms, centered_frames, centered_masks


@app.cell
def _(mo):
    mo.md("""
    ## 4. center on the mask
    each frame is translated so the subject centroid lands on the reference frame subject centroid before feature matching.
    """)
    return


@app.cell
def _(bgr_to_rgb, centered_frames, np, plt):
    fig_center, _axes_center = plt.subplots(1, len(centered_frames), figsize=(3.6 * len(centered_frames), 4))
    for _i, _ax in enumerate(np.ravel(_axes_center)):
        _ax.imshow(bgr_to_rgb(centered_frames[_i]))
        _ax.set_title(f"centered {_i + 1}")
        _ax.axis("off")
    fig_center.tight_layout()
    fig_center
    return


@app.cell
def _(
    align_to_reference,
    centered_frames,
    centered_masks,
    settings,
    transform_model,
):
    aligned_preprocessed, align_transforms, metrics, match_debug = align_to_reference(
        centered_frames,
        centered_masks,
        model=transform_model.value,
        n_features=settings["n_features"],
    )
    return align_transforms, aligned_preprocessed, match_debug, metrics


@app.cell
def _(mo):
    mo.md("""
    ## 5. match features and estimate transforms
    sift features are detected inside the subject mask, filtered with lowe's ratio test, then fit with a translation or affine transform.
    """)
    return


@app.cell
def _(bgr_to_rgb, cv2, match_debug, mo, plt, show_features):
    if show_features.value and match_debug is not None and len(match_debug[4]) > 0:
        source, reference, kp_source, kp_reference, matches = match_debug
        match_img = cv2.drawMatches(
            source,
            kp_source,
            reference,
            kp_reference,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        fig_matches, ax_matches = plt.subplots(figsize=(14, 5))
        ax_matches.imshow(bgr_to_rgb(match_img))
        ax_matches.set_title(f"frame 2 to frame 1: {len(matches)} strongest ratio-test matches")
        ax_matches.axis("off")
        fig_matches.tight_layout()
        match_preview = fig_matches
    else:
        match_preview = mo.md("_feature match preview hidden or no matches were found._")
    match_preview
    return


@app.cell
def _(metrics, mo):
    rows = [
        {
            "frame": m.frame,
            "matches": m.matches,
            "inliers": m.inliers,
            "IoU": round(m.iou, 3),
            "confidence": round(m.confidence, 3),
            "tx": round(m.tx, 1),
            "ty": round(m.ty, 1),
        }
        for m in metrics
    ]
    mo.ui.table(rows, label="alignment metrics")
    return


@app.cell
def _(aligned_preprocessed, bgr_to_rgb, np, plt):
    fig_aligned, _axes_aligned = plt.subplots(1, len(aligned_preprocessed), figsize=(3.6 * len(aligned_preprocessed), 4))
    for _i, _ax in enumerate(np.ravel(_axes_aligned)):
        _ax.imshow(bgr_to_rgb(aligned_preprocessed[_i]))
        _ax.set_title(f"aligned {_i + 1}")
        _ax.axis("off")
    fig_aligned.tight_layout()
    fig_aligned
    return


@app.cell
def _(align_transforms, center_transforms, cv2, original_frames):
    h, w = original_frames[0].shape[:2]
    aligned_originals = []
    for original, center_t, align_t in zip(original_frames, center_transforms, align_transforms):
        combined = align_t @ center_t
        aligned_originals.append(cv2.warpPerspective(original, combined, (w, h)))
    return (aligned_originals,)


@app.cell
def _(aligned_originals, make_boomerang):
    boomerang_frames = make_boomerang(aligned_originals)
    return (boomerang_frames,)


@app.cell
def _(mo):
    mo.md("""
    ## 6. render original scans
    the transforms estimated on preprocessed frames are applied to the original grainy frames, cropped to the shared valid region, brightness-normalized, and played forward then backward.
    """)
    return


@app.cell
def _(Image, bgr_to_rgb, boomerang_frames, io, mo):
    gif_buffer = io.BytesIO()
    pil_frames = [Image.fromarray(bgr_to_rgb(frame)) for frame in boomerang_frames]
    pil_frames[0].save(
        gif_buffer,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=120,
        loop=0,
        optimize=True,
    )
    mo.image(gif_buffer.getvalue(), width="520px")
    return


if __name__ == "__main__":
    app.run()
