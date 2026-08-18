"""
download_stare.py — Download STARE dataset from Clemson University website.

Usage:
    python src/preprocessing/download_stare.py
    python src/preprocessing/download_stare.py --output data/STARE/external_test

Downloads:
  - 20 labeled retinal images (.ppm) from cecas.clemson.edu/~ahoover/stare/
  - Corresponding vessel annotations (.ah) from Hoover annotator
  - Corresponding vessel annotations (.vk) from Valentine annotator (bonus)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

# The 20 STARE images with vessel ground-truth labels
STARE_IMAGE_IDS = [
    "0001", "0002", "0003", "0004", "0005",
    "0044", "0077", "0081", "0082", "0139",
    "0162", "0163", "0235", "0236", "0239",
    "0240", "0255", "0291", "0319", "0324",
]

STARE_BASE_URL = "https://cecas.clemson.edu/~ahoover/stare"
IMAGE_URL_TEMPLATE = f"{STARE_BASE_URL}/images/im{{id}}.ppm"
# Hoover annotator masks (primary — used as ground truth in most papers)
MASK_AH_URL_TEMPLATE = f"{STARE_BASE_URL}/probing/im{{id}}.ah"
# Valentine annotator masks (secondary — downloaded as bonus)
MASK_VK_URL_TEMPLATE = f"{STARE_BASE_URL}/probing/im{{id}}.vk"


def download_file(url: str, dest_path: Path, retries: int = 3) -> bool:
    """Download a file with progress display and retry logic."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 404:
                print(f"  [404 Not Found] {url}")
                return False
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            with open(dest_path, "wb") as f:
                if total:
                    with tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc=dest_path.name,
                        leave=False,
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            return True

        except requests.RequestException as e:
            print(f"  [Attempt {attempt}/{retries}] Error: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)  # exponential backoff

    return False


def download_stare(output_dir: str = "data/STARE/external_test") -> None:
    """
    Download all 20 STARE labeled images and their vessel masks.

    Args:
        output_dir: Root output directory. Images go to <output_dir>/images/,
                    masks to <output_dir>/masks/.
    """
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"
    masks_vk_dir = output_path / "masks_vk"  # secondary annotator

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    masks_vk_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  STARE Dataset Downloader")
    print(f"  Target: {output_path.resolve()}")
    print(f"  Images: {len(STARE_IMAGE_IDS)}")
    print(f"{'='*60}\n")

    success_images = 0
    success_masks_ah = 0
    success_masks_vk = 0
    failed = []

    for img_id in STARE_IMAGE_IDS:
        print(f"[{img_id}] Downloading...")

        # Image
        img_url = IMAGE_URL_TEMPLATE.format(id=img_id)
        img_dest = images_dir / f"im{img_id}.ppm"
        if img_dest.exists():
            print(f"  [OK] Image already exists: {img_dest.name}")
            success_images += 1
        elif download_file(img_url, img_dest):
            print(f"  [OK] Image: {img_dest.name}")
            success_images += 1
        else:
            print(f"  [FAIL] FAILED to download image for {img_id}")
            failed.append(f"image:{img_id}")

        # Hoover mask (.ah)
        mask_url = MASK_AH_URL_TEMPLATE.format(id=img_id)
        mask_dest = masks_dir / f"im{img_id}.ah"
        if mask_dest.exists():
            print(f"  [OK] Mask (AH) already exists: {mask_dest.name}")
            success_masks_ah += 1
        elif download_file(mask_url, mask_dest):
            print(f"  [OK] Mask (AH): {mask_dest.name}")
            success_masks_ah += 1
        else:
            print(f"  [FAIL] FAILED to download AH mask for {img_id}")
            failed.append(f"mask_ah:{img_id}")

        # Valentine mask (.vk) — best-effort
        mask_vk_url = MASK_VK_URL_TEMPLATE.format(id=img_id)
        mask_vk_dest = masks_vk_dir / f"im{img_id}.vk"
        if not mask_vk_dest.exists():
            download_file(mask_vk_url, mask_vk_dest)
            if mask_vk_dest.exists():
                success_masks_vk += 1
        else:
            success_masks_vk += 1

    print(f"\n{'='*60}")
    print(f"  Download Summary")
    print(f"{'='*60}")
    print(f"  Images downloaded:        {success_images}/{len(STARE_IMAGE_IDS)}")
    print(f"  AH masks downloaded:      {success_masks_ah}/{len(STARE_IMAGE_IDS)}")
    print(f"  VK masks downloaded:      {success_masks_vk}/{len(STARE_IMAGE_IDS)}")
    if failed:
        print(f"\n  FAILURES ({len(failed)}):")
        for f in failed:
            print(f"    - {f}")
        sys.exit(1)
    else:
        print(f"\n  [OK] All files downloaded successfully!")
        print(f"  Output: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Download STARE retinal vessel dataset from Clemson University."
    )
    parser.add_argument(
        "--output",
        default="data/STARE/external_test",
        help="Output directory for STARE data (default: data/STARE/external_test)",
    )
    args = parser.parse_args()
    download_stare(args.output)


if __name__ == "__main__":
    main()
