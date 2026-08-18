"""
preprocess.py — Reusable preprocessing functions for VessGen-Net v2.

This module contains pure, testable preprocessing functions. It does NOT
contain any training loop, data augmentation, or stochastic transforms.

All functions follow these contracts:
  - Image inputs: numpy arrays, uint8 (H, W, 3) RGB unless noted
  - Mask inputs:  numpy arrays, uint8 (H, W), values 0–255
  - All resizing of MASKS uses NEAREST NEIGHBOUR interpolation (enforced,
    never bilinear/bicubic, to prevent creation of spurious label values)
  - All resizing of IMAGES uses bilinear interpolation

Functions:
  apply_fov_mask        -- Zero out pixels outside the field-of-view mask
  resize_image          -- Bilinear resize to target size
  resize_mask           -- NEAREST NEIGHBOUR resize to target size (enforced)
  clahe_lab             -- CLAHE on L channel in LAB colour space
  clahe_green           -- CLAHE on green channel only (alternative)
  binarize_mask         -- Threshold mask to strict {0, 1}
  normalize_image       -- Scale to [0,1], optionally apply per-channel z-score
  compute_normalization_stats -- Compute and save per-channel mean/std from DRIVE train

Usage:
  from src.preprocessing.preprocess import (
      apply_fov_mask, resize_image, resize_mask,
      clahe_lab, clahe_green, binarize_mask,
      normalize_image, compute_normalization_stats,
  )
"""

import json
import os
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────
ImageArray = np.ndarray  # (H, W, 3) uint8 RGB
MaskArray  = np.ndarray  # (H, W)    uint8


# ═══════════════════════════════════════════════════════════════════════════
# 1. FOV Handling
# ═══════════════════════════════════════════════════════════════════════════

def apply_fov_mask(
    image: ImageArray,
    fov_mask: Optional[MaskArray],
    fill_value: int = 0,
) -> ImageArray:
    """
    Zero out (or fill) pixels outside the field-of-view mask.

    DRIVE provides official circular FOV masks. CHASE_DB1 and STARE do not
    have FOV masks; pass None and this function returns the image unchanged.

    Args:
        image:      RGB image, (H, W, 3) uint8.
        fov_mask:   Binary-ish mask (H, W) uint8. Non-zero = inside FOV.
                    Pass None if no FOV mask is available (CHASE/STARE).
        fill_value: Pixel value to use outside FOV (default 0 = black).

    Returns:
        Image with pixels outside FOV set to fill_value.
    """
    if fov_mask is None:
        return image.copy()

    if image.shape[:2] != fov_mask.shape[:2]:
        raise ValueError(
            f"Image shape {image.shape[:2]} != FOV mask shape {fov_mask.shape[:2]}"
        )

    result = image.copy()
    outside = fov_mask == 0
    result[outside] = fill_value
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 2. Resizing
# ═══════════════════════════════════════════════════════════════════════════

def resize_image(
    image: ImageArray,
    size: tuple[int, int] = (512, 512),
) -> ImageArray:
    """
    Resize a retinal image using bilinear interpolation.

    Args:
        image: RGB image, (H, W, 3) uint8.
        size:  Target (width, height). Default (512, 512).

    Returns:
        Resized image, (size[1], size[0], 3) uint8.
    """
    # cv2.resize takes (width, height) = size as given
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def resize_mask(
    mask: MaskArray,
    size: tuple[int, int] = (512, 512),
) -> MaskArray:
    """
    Resize a vessel mask using NEAREST NEIGHBOUR interpolation ONLY.

    INTER_NEAREST is the ONLY acceptable interpolation for binary masks.
    Using bilinear or cubic would create intermediate values (e.g., 127)
    between vessel (255) and background (0), corrupting the ground truth.
    This is enforced in code — the interpolation flag is hardcoded.

    Args:
        mask: (H, W) uint8 mask.
        size: Target (width, height). Default (512, 512).

    Returns:
        Resized mask, (size[1], size[0]) uint8 — values remain strictly {0, 255}
        (or {0, 1} if already binarized).
    """
    # ⚠️ NEAREST NEIGHBOUR IS ENFORCED HERE. DO NOT CHANGE.
    return cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Contrast Processing
# ═══════════════════════════════════════════════════════════════════════════

def clahe_lab(
    image: ImageArray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> ImageArray:
    """
    Apply CLAHE to the L channel in LAB colour space.

    Pipeline: RGB → LAB → CLAHE on L → LAB → RGB

    This is the primary contrast enhancement method. It operates on the
    luminance channel and avoids hue/saturation distortion.

    Args:
        image:      RGB image, (H, W, 3) uint8.
        clip_limit: CLAHE clip limit (default 2.0).
        tile_grid:  CLAHE tile grid size (default (8, 8)).

    Returns:
        Contrast-enhanced RGB image, (H, W, 3) uint8.
    """
    # BGR for OpenCV colour conversions
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    rgb_enhanced = cv2.cvtColor(bgr_enhanced, cv2.COLOR_BGR2RGB)

    return rgb_enhanced


def clahe_green(
    image: ImageArray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> ImageArray:
    """
    Apply CLAHE to the green channel only.

    Alternative to clahe_lab for later experimentation. The green channel
    has the highest vessel contrast in retinal images. CLAHE is applied
    to green only; red and blue channels remain unchanged.

    Args:
        image:      RGB image, (H, W, 3) uint8.
        clip_limit: CLAHE clip limit (default 2.0).
        tile_grid:  CLAHE tile grid size (default (8, 8)).

    Returns:
        Image with CLAHE-enhanced green channel, (H, W, 3) uint8.
    """
    result = image.copy()
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    result[:, :, 1] = clahe.apply(image[:, :, 1])
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 4. Normalization
# ═══════════════════════════════════════════════════════════════════════════

def normalize_image(
    image: ImageArray,
    mean: Optional[list[float]] = None,
    std: Optional[list[float]] = None,
) -> np.ndarray:
    """
    Normalize a retinal image to [0,1] and optionally apply per-channel z-score.

    Step 1: Scale pixel values to [0.0, 1.0] (divide by 255).
    Step 2 (optional): Apply (x - mean) / std per channel using DRIVE train stats.

    Args:
        image: RGB image, (H, W, 3) uint8.
        mean:  Per-channel mean [R, G, B] in [0,1] range. If None, skip z-score.
        std:   Per-channel std  [R, G, B] in [0,1] range. If None, skip z-score.

    Returns:
        Normalized image, (H, W, 3) float32.
    """
    img_float = image.astype(np.float32) / 255.0

    if mean is not None and std is not None:
        mean_arr = np.array(mean, dtype=np.float32)
        std_arr  = np.array(std,  dtype=np.float32)
        img_float = (img_float - mean_arr) / (std_arr + 1e-8)

    return img_float


def compute_normalization_stats(
    image_paths: list[Union[str, Path]],
    output_path: Union[str, Path] = "configs/normalization_stats.json",
    load_as_rgb: bool = True,
) -> dict:
    """
    Compute per-channel mean and std from a list of images and save to JSON.

    IMPORTANT: This must ONLY be called with DRIVE training images. Never
    include test images, CHASE images, or STARE images — that would cause
    data leakage.

    Args:
        image_paths: List of paths to images (DRIVE train only).
        output_path: Where to save the JSON stats file.
        load_as_rgb: If True, convert BGR→RGB after loading (for .tif with cv2).

    Returns:
        dict with keys: mean (list[float]), std (list[float]), n_images (int)
    """
    if not image_paths:
        raise ValueError("image_paths is empty. Provide DRIVE training image paths.")

    print(f"  Computing normalization stats from {len(image_paths)} images...")

    # Welford's online algorithm for numerically stable mean/variance
    # across a large number of images without loading all into memory
    n_pixels_total = 0
    channel_sum   = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)

    for img_path in image_paths:
        img_path = Path(img_path)

        # Load: use tifffile for .tif, cv2 for others
        if img_path.suffix.lower() in (".tif", ".tiff"):
            try:
                import tifffile
                img = tifffile.imread(str(img_path))
            except (ValueError, ImportError):
                # Fallback: PIL handles LZW-compressed TIFs without imagecodecs
                from PIL import Image as _PILImg
                img = np.array(_PILImg.open(str(img_path)).convert("RGB"))
            # tifffile returns (H, W, C) in native order — assume RGB
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)

        else:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"    [WARN] Could not read: {img_path}")
                continue
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        img_float = img.astype(np.float64) / 255.0
        h, w = img_float.shape[:2]
        n_pixels = h * w

        channel_sum    += img_float.reshape(-1, 3).sum(axis=0)
        channel_sum_sq += (img_float.reshape(-1, 3) ** 2).sum(axis=0)
        n_pixels_total += n_pixels

    mean = (channel_sum / n_pixels_total).tolist()
    variance = (channel_sum_sq / n_pixels_total) - np.array(mean) ** 2
    std = np.sqrt(np.maximum(variance, 0)).tolist()

    stats = {
        "mean":     mean,
        "std":      std,
        "n_images": len(image_paths),
        "note":     "Computed from DRIVE training set ONLY. Do not recompute with test/CHASE/STARE.",
        "channels": ["R", "G", "B"],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Normalization stats saved: {output_path}")
    print(f"    Mean (RGB): {[f'{v:.4f}' for v in mean]}")
    print(f"    Std  (RGB): {[f'{v:.4f}' for v in std]}")

    return stats


def load_normalization_stats(
    stats_path: Union[str, Path] = "configs/normalization_stats.json",
) -> tuple[list[float], list[float]]:
    """
    Load pre-computed normalization statistics from JSON.

    Returns:
        (mean, std) — each is a list of 3 floats [R, G, B] in [0,1] range.
    """
    stats_path = Path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Normalization stats not found: {stats_path}\n"
            "Run compute_normalization_stats() first."
        )
    with open(stats_path) as f:
        stats = json.load(f)
    return stats["mean"], stats["std"]


# ═══════════════════════════════════════════════════════════════════════════
# 5. Mask Binarization
# ═══════════════════════════════════════════════════════════════════════════

def binarize_mask(
    mask: MaskArray,
    threshold: int = 127,
) -> MaskArray:
    """
    Binarize a vessel mask to strict {0, 1} values.

    Vessel masks from DRIVE (GIF) are typically already binary but may have
    values 0/255. CHASE (rendered from polygon JSON) may have values 0/255.
    STARE (.ah files) are PGM-style with values 0/255. This function normalizes
    all to strict {0, 1}.

    Args:
        mask:      (H, W) uint8 mask, any values.
        threshold: Pixels > threshold → 1; others → 0. Default 127.

    Returns:
        (H, W) uint8 mask with strict values in {0, 1}.
    """
    return (mask > threshold).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: Full preprocessing pipeline for a single image-mask pair
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_pair(
    image: ImageArray,
    mask: MaskArray,
    fov_mask: Optional[MaskArray] = None,
    target_size: tuple[int, int] = (512, 512),
    use_clahe_lab: bool = True,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
    normalize: bool = False,
    norm_mean: Optional[list[float]] = None,
    norm_std: Optional[list[float]] = None,
) -> tuple[np.ndarray, MaskArray]:
    """
    Apply the full preprocessing pipeline to an image-mask pair.

    Order:
      1. Apply FOV mask (zero out outside-FOV pixels)
      2. Resize image to target_size (bilinear)
      3. Resize mask  to target_size (NEAREST NEIGHBOUR — enforced)
      4. CLAHE contrast enhancement (LAB or green channel)
      5. Binarize mask to {0, 1}
      6. (Optional) Normalize image

    Args:
        image:        RGB image, (H, W, 3) uint8.
        mask:         Vessel mask, (H, W) uint8.
        fov_mask:     FOV mask, (H, W) uint8 or None.
        target_size:  (width, height). Default (512, 512).
        use_clahe_lab: If True, use CLAHE on LAB L-channel. If False, green-only.
        clip_limit:   CLAHE clip limit.
        tile_grid:    CLAHE tile grid size.
        normalize:    If True, apply per-channel normalization.
        norm_mean:    Per-channel mean [R, G, B] for normalization.
        norm_std:     Per-channel std  [R, G, B] for normalization.

    Returns:
        (preprocessed_image, binarized_mask)
        image: float32 if normalize=True, uint8 otherwise
        mask:  uint8, values in {0, 1}
    """
    # 1. FOV masking
    img = apply_fov_mask(image, fov_mask)

    # 2. Resize image (bilinear)
    img = resize_image(img, target_size)

    # 3. Resize mask (NEAREST NEIGHBOUR — non-negotiable)
    msk = resize_mask(mask, target_size)

    # 4. CLAHE
    if use_clahe_lab:
        img = clahe_lab(img, clip_limit, tile_grid)
    else:
        img = clahe_green(img, clip_limit, tile_grid)

    # 5. Binarize mask
    msk = binarize_mask(msk)

    # 6. Normalize
    if normalize:
        img = normalize_image(img, norm_mean, norm_std)

    return img, msk
