# VessGen-Net v2

> **Retinal Vessel Segmentation with Domain Adaptation**  
> Environment Setup & Preprocessing Phase

---

## Project Overview

VessGen-Net v2 is a domain-adaptive retinal vessel segmentation framework. This repository covers the **environment setup and preprocessing phase only**. Model architecture, training loops, and domain adaptation code are out of scope for this phase.

---

## Dataset Roles

Three datasets are used, each with a distinct role in the learning pipeline:

| Dataset | Domain Role | Split | Notes |
|---------|-------------|-------|-------|
| **DRIVE** | Source domain (fully supervised) | 20 train / 20 test | Official DRIVE split preserved. `.tif` images, `.gif` vessel masks + FOV masks. |
| **CHASE_DB1** | Target domain (unsupervised adaptation) | 20 target_train / 8 target_test | **Target_train masks must NOT be used during domain-adaptation training** — they are present only for future evaluation/validation purposes. |
| **STARE** | External unseen domain (held out) | 20 external_test | No training or adaptation role whatsoever. Evaluates cross-domain generalization. |

---

## Folder Structure

```
project/                          ← (this repo root = d:\MIP)
  data/
    DRIVE/
      train/
        images/                   ← 20 .tif retinal images (21–40)
        masks/                    ← 20 .gif vessel annotations
        fov_masks/                ← 20 .gif FOV masks (circular field-of-view)
      test/
        images/                   ← 20 .tif retinal images (01–20)
        masks/                    ← 20 .gif vessel annotations (where available)
        fov_masks/                ← 20 .gif FOV masks
    CHASE/
      target_train/
        images/                   ← 20 .jpg retinal images
        masks/                    ← 20 vessel masks (DO NOT USE for DA training)
      target_test/
        images/                   ← 8 .jpg retinal images
        masks/                    ← 8 vessel masks
    STARE/
      external_test/
        images/                   ← 20 .ppm retinal images
        masks/                    ← 20 vessel masks (.ah annotator)
    manifest.csv                  ← Full dataset index with domain_role column
  configs/
    normalization_stats.json      ← Per-channel mean/std from DRIVE train only
  src/
    datasets/                     ← (placeholder — loaders come in next phase)
    preprocessing/
      organize_datasets.py        ← Dataset reorganization + manifest generation
      preprocess.py               ← Reusable preprocessing functions
      verify_preprocessing.py     ← Before/after visualization checks
      download_stare.py           ← Downloads STARE from Clemson website
    utils/
  results/
    preprocessing_check/          ← Before/after visualization PNGs
  README.md
  requirements.txt
```

> **Note**: Raw dataset files are excluded from git via `.gitignore`. Only code, configs (once generated), and the manifest CSV are tracked. Re-run `organize_datasets.py` to repopulate `data/` from raw archives.

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download STARE dataset

STARE is not included and must be fetched from the official Clemson source:

```bash
python src/preprocessing/download_stare.py --output data/STARE/external_test
```

**Manual alternative**: Download from https://cecas.clemson.edu/~ahoover/stare/probing/index.html  
The 20 labeled images are: `im0001, im0002, im0003, im0004, im0005, im0044, im0077, im0081, im0082, im0139, im0162, im0163, im0235, im0236, im0239, im0240, im0255, im0291, im0319, im0324`  
Download both `.ppm` images and `.ah` vessel masks (Hoover annotator).

### 3. Organize datasets

```bash
python src/preprocessing/organize_datasets.py \
    --drive "Dataset/archive (2)/DRIVE" \
    --chase "Dataset/chase-db1-DatasetNinja.tar" \
    --stare "data/STARE/external_test"
```

This will:
- Copy DRIVE images/masks to `data/DRIVE/train/` and `data/DRIVE/test/`
- Extract and split CHASE_DB1 (28 images → 20 target_train, 8 target_test)
- Validate STARE images/masks are present
- Generate `data/manifest.csv`

### 4. Compute normalization statistics

Run once after organizing datasets:

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.preprocessing.preprocess import compute_normalization_stats
import glob
image_paths = sorted(glob.glob('data/DRIVE/train/images/*.tif'))
compute_normalization_stats(image_paths, output_path='configs/normalization_stats.json')
"
```

### 5. Verify preprocessing

```bash
python src/preprocessing/verify_preprocessing.py
```

Check `results/preprocessing_check/` for before/after visualizations.

---

## Data Manifest

`data/manifest.csv` columns:

| Column | Description |
|--------|-------------|
| `image_id` | Unique identifier (e.g. `DRIVE_21_training`) |
| `dataset` | `DRIVE`, `CHASE`, or `STARE` |
| `subject_id` | Subject/image number where derivable |
| `image_path` | Relative path to image |
| `mask_path` | Relative path to vessel mask |
| `split` | `train`, `test`, `target_train`, `target_test`, or `external_test` |
| `domain_role` | `source_train`, `source_test`, `target_train`, `target_test`, `external_test` |

---

## Important Notes

- **CHASE target_train masks are present but MUST NOT be used during domain-adaptation training.** They exist solely for future evaluation/validation. This is enforced by design and documented in the manifest `domain_role` column.
- **Normalization stats are computed from DRIVE training images only.** Never compute from test, CHASE, or STARE images to prevent data leakage.
- **Mask resizing always uses NEAREST NEIGHBOUR interpolation.** Using bilinear or bicubic on binary masks would create spurious intermediate values — this is enforced in code in `preprocess.py`.
- **FOV masks**: DRIVE provides official circular FOV masks. CHASE_DB1 and STARE do not; FOV handling is skipped gracefully for those datasets.

---

## Out of Scope (This Phase)

- Model architecture (`src/models/` does not exist yet)
- Domain adaptation (FFT style mixing)
- Training/augmentation pipelines
- Loss functions
- Experiment tracking

---

## Citation

- **DRIVE**: Staal et al., IEEE TMI 2004
- **CHASE_DB1**: Fraz et al., IEEE TBME 2012  
- **STARE**: Hoover et al., IEEE TMI 2000
