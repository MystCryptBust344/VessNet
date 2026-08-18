"""
verify_preprocessing.py — Visual verification of the preprocessing pipeline.

Runs the preprocessing pipeline on ~20 image-mask pairs across the three
datasets and saves side-by-side before/after visualizations to
results/preprocessing_check/.

Visualizations include:
  - Raw image | FOV-masked | Resized | CLAHE-LAB | CLAHE-Green
  - Raw mask  | Resized    | Binarized
  - Overlay: CLAHE image with mask overlay

Usage:
    python src/preprocessing/verify_preprocessing.py
    python src/preprocessing/verify_preprocessing.py --n 5   # fewer images
    python src/preprocessing/verify_preprocessing.py --root /path/to/project
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Add project root to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.preprocess import (
    apply_fov_mask,
    binarize_mask,
    clahe_green,
    clahe_lab,
    load_normalization_stats,
    resize_image,
    resize_mask,
)

# ─────────────────────────────────────────────────────────────────────────────
TARGET_SIZE = (512, 512)
RESULTS_DIR = project_root / "results" / "preprocessing_check"
MANIFEST_CSV = project_root / "data" / "manifest.csv"
# ─────────────────────────────────────────────────────────────────────────────


def load_image(path: Path) -> np.ndarray:
    """Load any supported image format as RGB uint8."""
    p = str(path)

    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        img = tifffile.imread(p)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        return img.astype(np.uint8)

    elif path.suffix.lower() == ".ppm":
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Could not load: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    elif path.suffix.lower() in (".jpg", ".jpeg", ".png"):
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Could not load: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    else:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Could not load: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    """Load any mask format as grayscale uint8."""
    p = str(path)

    if path.suffix.lower() == ".gif":
        from PIL import Image
        mask = np.array(Image.open(p).convert("L"))
        return mask

    elif path.suffix.lower() == ".ah":
        # STARE .ah files are PPM/PGM format
        # Try cv2 first
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
        # Fallback: read as binary and parse PPM header
        with open(p, "rb") as f:
            data = f.read()
        # Try PIL
        from PIL import Image
        import io
        try:
            mask = np.array(Image.open(io.BytesIO(data)).convert("L"))
            return mask
        except Exception:
            raise IOError(f"Could not parse .ah mask: {path}")

    elif path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Could not load mask: {path}")
        return img

    else:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            from PIL import Image
            mask = np.array(Image.open(p).convert("L"))
            return mask
        return img


def load_fov_mask(dataset: str, subject_id: str, split: str) -> np.ndarray | None:
    """Load the FOV mask for DRIVE images. Returns None for CHASE/STARE."""
    if dataset != "DRIVE":
        return None

    fov_dir = project_root / "data" / "DRIVE" / split / "fov_masks"
    if not fov_dir.exists():
        return None

    # Pattern: "21_training_mask.gif" or "01_test_mask.gif"
    candidates = list(fov_dir.glob("*.gif"))
    for c in candidates:
        if c.stem.startswith(subject_id.zfill(2)):
            return load_mask(c)
    return None


def make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay vessel mask (red) on image.

    Args:
        image: (H, W, 3) uint8 RGB image.
        mask:  (H, W) binary {0, 1} mask.
        alpha: Transparency of overlay.

    Returns:
        (H, W, 3) uint8 RGB image with vessel overlay.
    """
    overlay = image.copy().astype(np.float32)
    vessel_color = np.array([255, 0, 0], dtype=np.float32)  # Red
    vessel_pixels = mask > 0
    overlay[vessel_pixels] = (
        (1 - alpha) * overlay[vessel_pixels] + alpha * vessel_color
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def visualize_pair(
    image_path: Path,
    mask_path: Path,
    dataset: str,
    subject_id: str,
    split: str,
    save_path: Path,
) -> dict:
    """
    Run preprocessing pipeline on one image-mask pair and save visualization.

    Returns a stats dict with image metadata.
    """
    # Load
    raw_img = load_image(image_path)
    raw_mask = load_mask(mask_path)
    fov_mask = load_fov_mask(dataset, subject_id, split)

    h_orig, w_orig = raw_img.shape[:2]
    n_channels = raw_img.shape[2] if raw_img.ndim == 3 else 1

    # Pipeline
    fov_applied = apply_fov_mask(raw_img, fov_mask)
    img_resized = resize_image(fov_applied, TARGET_SIZE)
    # ⚠️ Mask uses NEAREST NEIGHBOUR (enforced in resize_mask)
    mask_resized = resize_mask(raw_mask, TARGET_SIZE)
    img_clahe_lab = clahe_lab(img_resized)
    img_clahe_green = clahe_green(img_resized)
    mask_binary = binarize_mask(mask_resized)

    overlay = make_overlay(img_clahe_lab, mask_binary)

    # ─── Build figure ───
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle(
        f"{dataset} -- {image_path.name}  |  Original: {w_orig}x{h_orig} -> 512x512",
        fontsize=13, fontweight="bold", y=1.01,
    )

    def imshow(ax, img, title, cmap=None):
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap or "gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Row 1: Image pipeline
    imshow(axes[0, 0], raw_img,         "1. Raw Image")
    imshow(axes[0, 1], fov_applied,     "2. FOV Applied" if fov_mask is not None else "2. FOV (none)")
    imshow(axes[0, 2], img_resized,     "3. Resized 512x512\n(bilinear)")
    imshow(axes[0, 3], img_clahe_lab,   "4. CLAHE-LAB\n(L channel)")
    imshow(axes[0, 4], img_clahe_green, "5. CLAHE-Green\n(G channel)")

    # Row 2: Mask pipeline + overlay
    imshow(axes[1, 0], raw_mask,                                 "1. Raw Mask", cmap="gray")
    imshow(axes[1, 1], mask_resized,                             "2. Resized\n(NEAREST NEIGHBOUR)", cmap="gray")
    imshow(axes[1, 2], mask_binary * 255,                        "3. Binarized {0,1}x255", cmap="gray")
    imshow(axes[1, 3], overlay,                                   "4. CLAHE-LAB + Mask Overlay")

    # Stats panel
    vessel_frac = mask_binary.mean() * 100
    axes[1, 4].axis("off")
    stats_text = (
        f"Dataset:     {dataset}\n"
        f"Subject:     {subject_id}\n"
        f"Split:       {split}\n"
        f"Orig size:   {w_orig} x {h_orig}\n"
        f"Channels:    {n_channels}\n"
        f"Vessel px:   {vessel_frac:.1f}%\n"
        f"Mask unique: {np.unique(mask_binary).tolist()}\n"
        f"FOV mask:    {'Yes' if fov_mask is not None else 'No'}"
    )
    axes[1, 4].text(
        0.05, 0.95, stats_text,
        transform=axes[1, 4].transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    axes[1, 4].set_title("Image Stats", fontsize=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {
        "dataset": dataset,
        "subject_id": subject_id,
        "orig_width": w_orig,
        "orig_height": h_orig,
        "channels": n_channels,
        "vessel_pct": vessel_frac,
        "fov_mask": fov_mask is not None,
        "mask_unique_vals": np.unique(mask_binary).tolist(),
    }


def print_summary(all_stats: list[dict]) -> None:
    """Print per-dataset summary statistics."""
    print("\n" + "=" * 70)
    print("  Preprocessing Verification Summary")
    print("=" * 70)

    for dataset in ["DRIVE", "CHASE", "STARE"]:
        rows = [s for s in all_stats if s["dataset"] == dataset]
        if not rows:
            continue

        widths = [r["orig_width"] for r in rows]
        heights = [r["orig_height"] for r in rows]
        vessel_pcts = [r["vessel_pct"] for r in rows]

        print(f"\n  {dataset} ({len(rows)} images verified):")
        print(f"    Width range:   {min(widths)}-{max(widths)} px")
        print(f"    Height range:  {min(heights)}-{max(heights)} px")
        print(f"    Channels:      {rows[0]['channels']}")
        print(f"    Vessel %:      {min(vessel_pcts):.1f}%-{max(vessel_pcts):.1f}% "
              f"(mean {sum(vessel_pcts)/len(vessel_pcts):.1f}%)")
        print(f"    FOV masks:     {'Yes' if all(r['fov_mask'] for r in rows) else 'No/Partial'}")
        print(f"    Mask values:   {rows[0]['mask_unique_vals']} (should be [0, 1])")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Verify preprocessing pipeline with before/after visualizations."
    )
    parser.add_argument(
        "--n", type=int, default=7,
        help="Max images per dataset to visualize (default: 7, total ~21)",
    )
    parser.add_argument(
        "--root", default=str(project_root),
        help="Project root directory",
    )
    parser.add_argument(
        "--out", default=str(RESULTS_DIR),
        help="Output directory for visualizations",
    )
    args = parser.parse_args()

    root = Path(args.root)
    manifest_path = root / "data" / "manifest.csv"
    results_dir = Path(args.out)

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        print("  Run organize_datasets.py first to generate it.")
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    print(f"\nLoaded manifest: {len(manifest)} rows")

    all_stats = []
    processed = 0

    for dataset in ["DRIVE", "CHASE", "STARE"]:
        dataset_rows = manifest[manifest["dataset"] == dataset]
        # Only use rows that have masks
        with_masks = dataset_rows[dataset_rows["mask_path"].notna() & (dataset_rows["mask_path"] != "")]
        sample = with_masks.head(args.n)

        if sample.empty:
            print(f"\n  [{dataset}] No images with masks found in manifest. Skipping.")
            continue

        print(f"\n  [{dataset}] Verifying {len(sample)} images...")

        for _, row in sample.iterrows():
            img_path  = root / row["image_path"]
            mask_path = root / row["mask_path"]

            if not img_path.exists():
                print(f"    [SKIP] Image not found: {img_path}")
                continue
            if not mask_path.exists():
                print(f"    [SKIP] Mask not found: {mask_path}")
                continue

            save_name = f"{dataset}_{row['subject_id']}_{row['split']}.png"
            save_path = results_dir / save_name

            try:
                stats = visualize_pair(
                    image_path=img_path,
                    mask_path=mask_path,
                    dataset=dataset,
                    subject_id=str(row["subject_id"]),
                    split=row["split"],
                    save_path=save_path,
                )
                all_stats.append(stats)
                print(f"    [OK] Saved: {save_name}")
                processed += 1
            except Exception as e:
                print(f"    [ERR] Error processing {img_path.name}: {e}")

    print(f"\nProcessed {processed} image-mask pairs.")
    print(f"Visualizations saved to: {results_dir.resolve()}")

    if all_stats:
        print_summary(all_stats)
    else:
        print("\n[WARN] No images were verified. Check that organize_datasets.py has been run.")


if __name__ == "__main__":
    main()
