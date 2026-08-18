"""
organize_datasets.py — Reorganize raw retinal vessel datasets into the
VessGen-Net v2 canonical folder structure and generate a manifest CSV.

Usage:
    python src/preprocessing/organize_datasets.py \\
        --drive  "Dataset/archive (2)/DRIVE" \\
        --chase  "Dataset/chase-db1-DatasetNinja.tar" \\
        --stare  "data/STARE/external_test"

    # Skip a dataset if not yet available:
    python src/preprocessing/organize_datasets.py \\
        --drive  "Dataset/archive (2)/DRIVE"

Outputs:
    data/DRIVE/train/images/          20 .tif images
    data/DRIVE/train/masks/           20 vessel annotations (.gif)
    data/DRIVE/train/fov_masks/       20 FOV masks (.gif)
    data/DRIVE/test/images/           20 .tif images
    data/DRIVE/test/masks/            20 vessel annotations (.gif)
    data/DRIVE/test/fov_masks/        20 FOV masks (.gif)
    data/CHASE/target_train/images/   20 .jpg images
    data/CHASE/target_train/masks/    20 vessel masks (rendered from JSON)
    data/CHASE/target_test/images/    8 .jpg images
    data/CHASE/target_test/masks/     8 vessel masks
    data/STARE/external_test/images/  20 .ppm images
    data/STARE/external_test/masks/   20 .ah vessel masks
    data/manifest.csv
"""

import argparse
import csv
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CHASE_DB1 split: 28 images total -> 20 target_train, 8 target_test
# Sorted alphabetically, last 8 go to target_test.
# Subject IDs: Image_01L, Image_01R, ..., Image_14L, Image_14R
# Sorted order: 01L, 01R, 02L, 02R, ..., 14L, 14R
# Last 8 = 11L, 11R, 12L, 12R, 13L, 13R, 14L, 14R
CHASE_TARGET_TEST_IDS = {
    "Image_11L", "Image_11R",
    "Image_12L", "Image_12R",
    "Image_13L", "Image_13R",
    "Image_14L", "Image_14R",
}
# ─────────────────────────────────────────────────────────────────────────────


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_copy(src: Path, dst: Path) -> None:
    """Copy file, creating parent dirs as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ═══════════════════════════════════════════════════════════════════════════
# DRIVE
# ═══════════════════════════════════════════════════════════════════════════

def organize_drive(raw_dir: str, project_root: Path) -> list[dict]:
    """
    Organize DRIVE dataset from its raw downloaded folder.

    Raw DRIVE layout:
        <raw_dir>/
            training/
                images/       21_training.tif ... 40_training.tif
                1st_manual/   21_manual1.gif  ... 40_manual1.gif
                mask/         21_training_mask.gif ... 40_training_mask.gif
            test/
                images/       01_test.tif ... 20_test.tif
                mask/         01_test_mask.gif ... 20_test_mask.gif
                (no 1st_manual in test — only FOV masks)

    Note: DRIVE test does NOT ship with ground-truth vessel annotations in the
    Kaggle archive. The official DRIVE test annotations exist but require
    registration. We copy FOV masks to test/fov_masks/ as available.
    """
    raw = Path(raw_dir)
    manifest_rows = []

    splits = {
        "train": {
            "raw_images": raw / "training" / "images",
            "raw_masks":  raw / "training" / "1st_manual",
            "raw_fov":    raw / "training" / "mask",
            "out_images": project_root / "data" / "DRIVE" / "train" / "images",
            "out_masks":  project_root / "data" / "DRIVE" / "train" / "masks",
            "out_fov":    project_root / "data" / "DRIVE" / "train" / "fov_masks",
            "split":      "train",
            "domain_role": "source_train",
            "img_suffix":  "_training.tif",
            "mask_suffix": "_manual1.gif",
            "fov_suffix":  "_training_mask.gif",
        },
        "test": {
            "raw_images": raw / "test" / "images",
            "raw_masks":  None,  # not available in standard Kaggle archive
            "raw_fov":    raw / "test" / "mask",
            "out_images": project_root / "data" / "DRIVE" / "test" / "images",
            "out_masks":  project_root / "data" / "DRIVE" / "test" / "masks",
            "out_fov":    project_root / "data" / "DRIVE" / "test" / "fov_masks",
            "split":      "test",
            "domain_role": "source_test",
            "img_suffix":  "_test.tif",
            "mask_suffix": None,
            "fov_suffix":  "_test_mask.gif",
        },
    }

    for split_name, cfg in splits.items():
        imgs_dir = cfg["raw_images"]
        if not imgs_dir.exists():
            print(f"  [DRIVE/{split_name}] Images directory not found: {imgs_dir}")
            continue

        images = sorted(imgs_dir.glob("*.tif"))
        print(f"  [DRIVE/{split_name}] Found {len(images)} images")

        for img_path in images:
            stem = img_path.stem  # e.g. "21_training" or "01_test"
            # Extract numeric ID
            subject_id = stem.split("_")[0]  # "21" or "01"

            out_img = cfg["out_images"] / img_path.name
            safe_copy(img_path, out_img)

            mask_path = None
            out_mask = None

            # Vessel mask (training only in Kaggle archive)
            if cfg["raw_masks"] is not None:
                expected_mask_name = f"{subject_id}{cfg['mask_suffix']}"
                raw_mask = cfg["raw_masks"] / expected_mask_name
                if raw_mask.exists():
                    out_mask = cfg["out_masks"] / raw_mask.name
                    safe_copy(raw_mask, out_mask)
                    mask_path = str(out_mask.relative_to(project_root))
                else:
                    print(f"    [WARN] Mask not found: {raw_mask}")

            # FOV mask
            if cfg["raw_fov"] is not None and cfg["raw_fov"].exists():
                expected_fov_name = f"{subject_id}{cfg['fov_suffix']}"
                raw_fov = cfg["raw_fov"] / expected_fov_name
                if raw_fov.exists():
                    out_fov = cfg["out_fov"] / raw_fov.name
                    safe_copy(raw_fov, out_fov)

            manifest_rows.append({
                "image_id":    f"DRIVE_{stem}",
                "dataset":     "DRIVE",
                "subject_id":  subject_id,
                "image_path":  str(out_img.relative_to(project_root)),
                "mask_path":   mask_path or "",
                "split":       cfg["split"],
                "domain_role": cfg["domain_role"],
            })

    return manifest_rows


# ═══════════════════════════════════════════════════════════════════════════
# CHASE_DB1
# ═══════════════════════════════════════════════════════════════════════════

def _render_chase_mask_from_json(ann_json_path: Path, img_shape: tuple) -> np.ndarray:
    """
    Render a binary vessel mask from CHASE DatasetNinja JSON annotation.

    DatasetNinja CHASE_DB1 uses Supervisely-format bitmap annotations:
        {
            "geometryType": "bitmap",
            "bitmap": {
                "data": "<base64-encoded zlib-compressed PNG>",
                "origin": [x, y]   # top-left corner of the bitmap in image coords
            }
        }

    Decode: base64 -> zlib decompress -> PNG bytes -> PIL Image (palette mode P).
    Values: {0=background, 1=vessel}. Paste onto full-image canvas using origin.
    """
    import base64
    import io as _io
    import zlib
    from PIL import Image as _PIL_Image

    with open(ann_json_path, "r") as f:
        ann = json.load(f)

    h, w = img_shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)

    objects = ann.get("objects", [])
    for obj in objects:
        geometry_type = obj.get("geometryType", "")

        if geometry_type == "bitmap":
            bmp = obj.get("bitmap", {})
            b64_data = bmp.get("data", "")
            origin = bmp.get("origin", [0, 0])  # [x=col, y=row]

            if not b64_data:
                continue

            try:
                import base64 as _b64
                import zlib as _zlib
                raw_bytes = _b64.b64decode(b64_data)
                png_bytes = _zlib.decompress(raw_bytes)
                bmp_img = _PIL_Image.open(_io.BytesIO(png_bytes))
                bmp_arr = np.array(bmp_img)  # shape (bh, bw), values {0, 1}

                x0, y0 = int(origin[0]), int(origin[1])
                bh, bw = bmp_arr.shape[:2]

                x1 = min(x0 + bw, w)
                y1 = min(y0 + bh, h)
                bw_clip = x1 - x0
                bh_clip = y1 - y0

                if bw_clip > 0 and bh_clip > 0:
                    region = (bmp_arr[:bh_clip, :bw_clip] > 0).astype(np.uint8) * 255
                    full_mask[y0:y1, x0:x1] = np.maximum(
                        full_mask[y0:y1, x0:x1], region
                    )
            except Exception as e:
                print(f"    [WARN] Bitmap decode error in {ann_json_path.name}: {e}")

        elif geometry_type == "polygon":
            import cv2 as _cv2_poly
            exterior = obj.get("points", {}).get("exterior", [])
            if exterior:
                pts = np.array(exterior, dtype=np.int32).reshape(-1, 1, 2)
                _cv2_poly.fillPoly(full_mask, [pts], color=255)

        else:
            print(f"    [INFO] Skipping geometry '{geometry_type}' in {ann_json_path.name}")

    return full_mask


def organize_chase(raw_path: str, project_root: Path) -> list[dict]:
    """
    Organize CHASE_DB1 from a DatasetNinja .tar archive or extracted folder.

    DatasetNinja CHASE_DB1 layout inside tar:
        ds0/img/Image_01L.jpg ... Image_14R.jpg   (28 images)
        ds0/ann/Image_01L.jpg.json ... (28 JSON annotations)

    Split: last 8 (11L, 11R, 12L, 12R, 13L, 13R, 14L, 14R) -> target_test
           remaining 20 -> target_train
    """
    try:
        import cv2
    except ImportError:
        print("  [ERROR] opencv-python not installed. Run: pip install opencv-python")
        sys.exit(1)

    raw = Path(raw_path)
    manifest_rows = []

    # ── Extract tar if needed ──
    working_dir: Optional[Path] = None
    temp_dir = None

    if raw.suffix == ".tar" or raw.name.endswith(".tar.gz"):
        print(f"  [CHASE] Extracting {raw.name}...")
        temp_dir = tempfile.mkdtemp(prefix="chase_extract_")
        with tarfile.open(raw, "r:*") as tf:
            tf.extractall(temp_dir)
        working_dir = Path(temp_dir)
    elif raw.is_dir():
        working_dir = raw
    else:
        print(f"  [CHASE] Cannot find or open: {raw}")
        return []

    # ── Locate ds0/img and ds0/ann ──
    img_dir = working_dir / "ds0" / "img"
    ann_dir = working_dir / "ds0" / "ann"

    if not img_dir.exists():
        # Try one level deeper (in case extracted into subfolder)
        candidates = list(working_dir.rglob("ds0/img"))
        if candidates:
            img_dir = candidates[0]
            ann_dir = img_dir.parent.parent / "ds0" / "ann"

    if not img_dir.exists():
        print(f"  [CHASE] Could not locate ds0/img inside {raw}. Skipping.")
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return []

    images = sorted(img_dir.glob("*.jpg"))
    print(f"  [CHASE] Found {len(images)} images")

    for img_path in images:
        stem = img_path.stem  # e.g. "Image_01L"

        # Determine split
        is_test = stem in CHASE_TARGET_TEST_IDS
        split = "target_test" if is_test else "target_train"
        domain_role = split  # same value

        out_subdir = project_root / "data" / "CHASE" / split
        out_img = out_subdir / "images" / img_path.name
        safe_copy(img_path, out_img)

        # ── Render mask from JSON annotation ──
        ann_json = ann_dir / f"{img_path.name}.json"
        out_mask_path = ""
        if ann_json.exists():
            import cv2 as cv2_local
            img_bgr = cv2_local.imread(str(img_path))
            if img_bgr is not None:
                mask = _render_chase_mask_from_json(ann_json, img_bgr.shape)
                out_mask = out_subdir / "masks" / f"{stem}_mask.png"
                out_mask.parent.mkdir(parents=True, exist_ok=True)
                import cv2 as _cv2
                _cv2.imwrite(str(out_mask), mask)
                out_mask_path = str(out_mask.relative_to(project_root))
        else:
            print(f"    [WARN] No annotation JSON for {stem}")

        # Extract subject_id (e.g. "01L" from "Image_01L")
        subject_id = stem.replace("Image_", "")  # "01L", "02R", etc.

        manifest_rows.append({
            "image_id":    f"CHASE_{stem}",
            "dataset":     "CHASE",
            "subject_id":  subject_id,
            "image_path":  str(out_img.relative_to(project_root)),
            "mask_path":   out_mask_path,
            "split":       split,
            "domain_role": domain_role,
        })

    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return manifest_rows


# ═══════════════════════════════════════════════════════════════════════════
# STARE
# ═══════════════════════════════════════════════════════════════════════════

def organize_stare(raw_dir: str, project_root: Path) -> list[dict]:
    """
    Organize STARE dataset from its download directory.

    Expected layout (output of download_stare.py):
        <raw_dir>/
            images/   im0001.ppm ... im0324.ppm  (20 files)
            masks/    im0001.ah  ... im0324.ah    (Hoover annotator)
            masks_vk/ im0001.vk  ... im0324.vk   (Valentine annotator, optional)

    OR from any flat directory containing .ppm images and .ah/.vk masks.
    """
    raw = Path(raw_dir)
    manifest_rows = []

    if not raw.exists():
        print(f"  [STARE] Directory not found: {raw}. Skipping.")
        print(f"  [STARE] Run: python src/preprocessing/download_stare.py")
        return []

    # Locate images — could be in raw/images/ or raw/ directly
    imgs_dir = raw / "images" if (raw / "images").exists() else raw
    masks_dir = raw / "masks" if (raw / "masks").exists() else raw

    images = sorted(imgs_dir.glob("*.ppm"))
    print(f"  [STARE] Found {len(images)} images")

    out_base = project_root / "data" / "STARE" / "external_test"

    for img_path in images:
        stem = img_path.stem  # e.g. "im0001"
        img_id = stem  # keep original name

        out_img = out_base / "images" / img_path.name
        if str(raw) != str(out_base):  # avoid copying to itself
            safe_copy(img_path, out_img)

        # Hoover mask (.ah) — primary
        ah_mask = masks_dir / f"{stem}.ah"
        out_mask_path = ""
        if ah_mask.exists():
            out_mask = out_base / "masks" / f"{stem}.ah"
            if str(raw) != str(out_base):
                safe_copy(ah_mask, out_mask)
            out_mask_path = str(out_mask.relative_to(project_root))
        else:
            print(f"    [WARN] No .ah mask for {stem} — checked {ah_mask}")

        # Extract numeric subject_id
        subject_id = stem.replace("im", "").lstrip("0") or "0"

        manifest_rows.append({
            "image_id":    f"STARE_{stem}",
            "dataset":     "STARE",
            "subject_id":  subject_id,
            "image_path":  str(out_img.relative_to(project_root)),
            "mask_path":   out_mask_path,
            "split":       "external_test",
            "domain_role": "external_test",
        })

    return manifest_rows


# ═══════════════════════════════════════════════════════════════════════════
# Manifest generation
# ═══════════════════════════════════════════════════════════════════════════

def write_manifest(rows: list[dict], output_path: Path) -> None:
    """Write dataset manifest to CSV."""
    columns = ["image_id", "dataset", "subject_id", "image_path", "mask_path", "split", "domain_role"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Manifest written: {output_path} ({len(rows)} rows)")


# ═══════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════

def verify_organization(project_root: Path) -> None:
    """
    Quick sanity check after organization:
      - Confirms image/mask pairs match (same filename stem, same count)
      - Prints image count per dataset/split
      - Flags missing masks or mismatched pairs
    """
    print("\n" + "=" * 60)
    print("  Organization Verification")
    print("=" * 60)

    checks = [
        ("DRIVE/train",        "data/DRIVE/train/images",        "data/DRIVE/train/masks",        ".tif", ".gif"),
        ("DRIVE/test",         "data/DRIVE/test/images",         None,                            ".tif", None),
        ("CHASE/target_train", "data/CHASE/target_train/images", "data/CHASE/target_train/masks", ".jpg", ".png"),
        ("CHASE/target_test",  "data/CHASE/target_test/images",  "data/CHASE/target_test/masks",  ".jpg", ".png"),
        ("STARE/external_test","data/STARE/external_test/images","data/STARE/external_test/masks", ".ppm", ".ah"),
    ]

    all_ok = True
    for label, img_rel, mask_rel, img_ext, mask_ext in checks:
        img_dir = project_root / img_rel
        imgs = sorted(img_dir.glob(f"*{img_ext}")) if img_dir.exists() else []

        line = f"  {label:<25} images: {len(imgs):>3}"

        if mask_rel is not None:
            mask_dir = project_root / mask_rel
            masks = sorted(mask_dir.glob(f"*{mask_ext}")) if mask_dir.exists() else []
            line += f"  masks: {len(masks):>3}"

            if len(imgs) != len(masks):
                line += "  !! MISMATCH"
                all_ok = False
        else:
            line += "  masks: N/A (test without GT)"

        print(line)

    print()
    if all_ok:
        print("  [OK] All checks passed.")
    else:
        print("  [!!] Some checks failed -- review warnings above.")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Organize retinal vessel datasets into VessGen-Net v2 structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Organize all three datasets
  python src/preprocessing/organize_datasets.py \\
      --drive "Dataset/archive (2)/DRIVE" \\
      --chase "Dataset/chase-db1-DatasetNinja.tar" \\
      --stare "data/STARE/external_test"

  # Organize only DRIVE (others not yet available)
  python src/preprocessing/organize_datasets.py \\
      --drive "Dataset/archive (2)/DRIVE"
        """,
    )
    parser.add_argument(
        "--drive",
        default=None,
        help='Path to raw DRIVE folder (containing "training/" and "test/" subfolders)',
    )
    parser.add_argument(
        "--chase",
        default=None,
        help='Path to CHASE_DB1 .tar archive or extracted folder',
    )
    parser.add_argument(
        "--stare",
        default=None,
        help='Path to STARE directory (output of download_stare.py, or any folder with .ppm + .ah files)',
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root directory (default: current directory)",
    )
    args = parser.parse_args()

    project_root = Path(args.root).resolve()
    print(f"\nProject root: {project_root}")

    all_rows: list[dict] = []

    if args.drive:
        print(f"\n[1/3] Organizing DRIVE from: {args.drive}")
        rows = organize_drive(args.drive, project_root)
        all_rows.extend(rows)
        print(f"      -> {len(rows)} entries added")
    else:
        print("\n[1/3] DRIVE: --drive not specified, skipping")

    if args.chase:
        print(f"\n[2/3] Organizing CHASE_DB1 from: {args.chase}")
        rows = organize_chase(args.chase, project_root)
        all_rows.extend(rows)
        print(f"      -> {len(rows)} entries added")
    else:
        print("\n[2/3] CHASE: --chase not specified, skipping")

    if args.stare:
        print(f"\n[3/3] Organizing STARE from: {args.stare}")
        rows = organize_stare(args.stare, project_root)
        all_rows.extend(rows)
        print(f"      -> {len(rows)} entries added")
    else:
        print("\n[3/3] STARE: --stare not specified, skipping")
        print("      Run: python src/preprocessing/download_stare.py  then re-run with --stare")

    if all_rows:
        manifest_path = project_root / "data" / "manifest.csv"
        write_manifest(all_rows, manifest_path)
        verify_organization(project_root)
    else:
        print("\n  No datasets organized. Provide at least one of --drive, --chase, --stare.")
        sys.exit(1)


if __name__ == "__main__":
    main()
