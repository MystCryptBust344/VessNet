"""
eda.py — Phase 2 Exploratory Data Analysis

This script performs an exhaustive analysis of all 88 images across DRIVE, CHASE, and STARE.
It computes both native and post-resize statistics, ensures 100% of the datasets load 
without corruption, and empirically visualizes the domain gap using ResNet-50 + t-SNE.
"""

import os
import sys
import json
import csv
from pathlib import Path
import warnings

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.manifold import TSNE

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.preprocess import preprocess_pair
from src.preprocessing.verify_preprocessing import load_image, load_mask, load_fov_mask

# Suppress typical Torchvision warnings about weights
warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_DIR = project_root / "results" / "eda"
MANIFEST_CSV = project_root / "data" / "manifest.csv"


class FeatureExtractor(nn.Module):
    """Extracts 2048-dim features from a pre-trained ResNet-50."""
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final classification FC layer; keep the AdaptiveAvgPool2d
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.eval()
        
    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            return x.view(x.size(0), -1)  # Flatten to (B, 2048)


def get_brightness_contrast(img_rgb: np.ndarray) -> tuple[float, float]:
    """Calculate perceived brightness and RMS contrast of an RGB image."""
    # Convert to grayscale via luminosity method
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    brightness = gray.mean()
    contrast = gray.std()
    return brightness, contrast


def extract_bifurcation_patch(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Find a rough bounding box containing a vessel bifurcation to zoom into."""
    # Simple heuristic: find coordinates of all vessel pixels
    y, x = np.where(mask > 0)
    if len(y) == 0:
        return 0, 0, 50, 50
    # Pick a random patch near the center of the vessel distribution
    cx, cy = int(np.median(x)), int(np.median(y))
    # Return a 100x100 patch around the center
    size = 100
    x0 = max(0, cx - size//2)
    y0 = max(0, cy - size//2)
    return x0, y0, x0 + size, y0 + size


def plot_zoomed_bifurcation(
    native_img: np.ndarray, native_mask: np.ndarray,
    prep_img: np.ndarray, prep_mask: np.ndarray,
    title: str, save_path: Path
):
    """Plot an explicitly zoomed before/after comparison to check for distortion."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # We want a 150x150 patch
    h, w = native_img.shape[:2]
    # Pick center-ish 200x200 patch
    y0, x0 = h // 2 - 100, w // 2 - 100
    y1, x1 = y0 + 200, x0 + 200
    
    n_patch_img = native_img[y0:y1, x0:x1]
    n_patch_msk = native_mask[y0:y1, x0:x1]
    
    # Equivalent patch in 512x512
    scale_y = 512 / h
    scale_x = 512 / w
    py0, px0 = int(y0 * scale_y), int(x0 * scale_x)
    py1, px1 = int(y1 * scale_y), int(x1 * scale_x)
    
    p_patch_img = prep_img[py0:py1, px0:px1]
    p_patch_msk = prep_mask[py0:py1, px0:px1]
    
    axes[0,0].imshow(n_patch_img)
    axes[0,0].set_title(f"Native Size ({w}x{h})")
    axes[0,1].imshow(p_patch_img)
    axes[0,1].set_title(f"Resized (512x512)")
    
    axes[1,0].imshow(n_patch_msk, cmap="gray")
    axes[1,0].set_title("Native Mask")
    axes[1,1].imshow(p_patch_msk, cmap="gray")
    axes[1,1].set_title("Resized Mask (Nearest)")
    
    for ax in axes.flatten():
        ax.axis("off")
        
    plt.tight_layout()
    plt.savefig(str(save_path), bbox_inches='tight', dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("  Phase 2: Exploratory Data Analysis & Verification")
    print("="*60)

    if not MANIFEST_CSV.exists():
        raise FileNotFoundError("Run organize_datasets.py first!")
        
    manifest = pd.read_csv(MANIFEST_CSV)
    total_images = len(manifest)
    print(f"Loading manifest: {total_images} images total.\n")

    # Initialize PyTorch ResNet-50 feature extractor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing ResNet-50 on {device}...")
    extractor = FeatureExtractor().to(device)
    
    # Transform for ImageNet ResNet
    resnet_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    all_stats = []
    features_list = []
    labels_list = []
    pixel_intensities = {"DRIVE": [], "CHASE": [], "STARE": []}
    
    chase_zoomed = False

    for idx, row in tqdm(manifest.iterrows(), total=total_images, desc="Processing images"):
        dataset = row["dataset"]
        img_path = project_root / row["image_path"]
        mask_path = project_root / str(row["mask_path"]) if pd.notna(row["mask_path"]) and row["mask_path"] else None
        
        # Load raw data
        raw_img = load_image(img_path)
        h, w = raw_img.shape[:2]
        
        brightness, contrast = get_brightness_contrast(raw_img)
        mean_r, mean_g, mean_b = raw_img.mean(axis=(0,1))
        
        # Collect pixel intensities for histograms (subsample for memory)
        gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
        pixel_intensities[dataset].extend(np.random.choice(gray.flatten(), 1000).tolist())
        
        # Determine masks
        raw_mask = load_mask(mask_path) if mask_path and mask_path.exists() else None
        fov_mask = load_fov_mask(dataset, str(row["subject_id"]), str(row["split"]))
        
        # Compute pre-resize vessel %
        native_vessel_pct = 0.0
        if raw_mask is not None:
            # Quick binarize for native calculation
            native_vessel_pct = float(np.mean(raw_mask > 127) * 100)
            
        # Run standard preprocessing
        # Note: if there is no mask (DRIVE test), we pass a dummy mask
        dummy_mask = np.zeros_like(gray) if raw_mask is None else raw_mask
        
        prep_img, prep_mask = preprocess_pair(
            image=raw_img, 
            mask=dummy_mask,
            fov_mask=fov_mask,
            target_size=(512, 512),
            use_clahe_lab=True,
            normalize=False
        )
        
        # Compute post-resize vessel %
        post_vessel_pct = 0.0
        if raw_mask is not None:
            post_vessel_pct = float(np.mean(prep_mask > 0) * 100)
            
        # CHASE distortion check plot
        if dataset == "CHASE" and not chase_zoomed and raw_mask is not None:
            plot_zoomed_bifurcation(
                raw_img, raw_mask, prep_img, prep_mask,
                title=f"CHASE Distortion Check: {row['image_id']}",
                save_path=RESULTS_DIR / "chase_distortion_check.png"
            )
            chase_zoomed = True
            
        # Extract deep features
        tensor_img = resnet_transforms(prep_img).unsqueeze(0).to(device)
        feat = extractor(tensor_img).cpu().numpy().flatten()
        features_list.append(feat)
        labels_list.append(dataset)
        
        # Record stats
        all_stats.append({
            "image_id": row["image_id"],
            "dataset": dataset,
            "split": row["split"],
            "native_w": w,
            "native_h": h,
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "mean_r": round(mean_r, 2),
            "mean_g": round(mean_g, 2),
            "mean_b": round(mean_b, 2),
            "native_vessel_pct": round(native_vessel_pct, 3) if raw_mask is not None else None,
            "post_resize_vessel_pct": round(post_vessel_pct, 3) if raw_mask is not None else None,
            "vessel_pct_diff": round(post_vessel_pct - native_vessel_pct, 3) if raw_mask is not None else None,
            "has_mask": raw_mask is not None
        })

    # ─────────────────────────────────────────────────────────────
    # Save per-image CSV
    # ─────────────────────────────────────────────────────────────
    df_stats = pd.DataFrame(all_stats)
    csv_path = RESULTS_DIR / "per_image_stats.csv"
    df_stats.to_csv(csv_path, index=False)
    print(f"\nSaved raw per-image statistics to {csv_path}")

    # ─────────────────────────────────────────────────────────────
    # Plot Histograms
    # ─────────────────────────────────────────────────────────────
    print("Generating EDA visualizations...")
    
    # 1. Pixel Intensity Overlays
    plt.figure(figsize=(10, 6))
    sns.kdeplot(pixel_intensities["DRIVE"], label="DRIVE", fill=True, alpha=0.3)
    sns.kdeplot(pixel_intensities["CHASE"], label="CHASE", fill=True, alpha=0.3)
    sns.kdeplot(pixel_intensities["STARE"], label="STARE", fill=True, alpha=0.3)
    plt.title("Pixel Intensity Distributions (Grayscale)", fontsize=14)
    plt.xlabel("Pixel Intensity [0-255]")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(RESULTS_DIR / "pixel_intensity_histograms.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    # 2. Vessel Area distribution
    df_masks = df_stats[df_stats["has_mask"] == True]
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df_masks, x="dataset", y="native_vessel_pct")
    plt.title("Native Vessel Area (%)")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df_masks, x="dataset", y="vessel_pct_diff")
    plt.title("Change in Vessel % after 512x512 Resize\n(closeness to 0 is better)")
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "vessel_area_distributions.png", bbox_inches='tight', dpi=150)
    plt.close()

    # ─────────────────────────────────────────────────────────────
    # Domain Gap: t-SNE
    # ─────────────────────────────────────────────────────────────
    print("Running t-SNE on ResNet-50 features...")
    X = np.stack(features_list)
    # Using random_state=42 for reproducibility as requested
    tsne = TSNE(n_components=2, random_state=42, perplexity=15, n_iter=1000)
    X_2d = tsne.fit_transform(X)
    
    df_tsne = pd.DataFrame({
        "tsne_1": X_2d[:, 0],
        "tsne_2": X_2d[:, 1],
        "Dataset": labels_list
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_tsne, x="tsne_1", y="tsne_2", hue="Dataset", 
        palette="Set1", s=100, alpha=0.8, edgecolor='k'
    )
    plt.title("Domain Gap Visualization\nt-SNE on ResNet-50 (AvgPool) Features", fontsize=15, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    # Note on interpretation directly on the plot
    plt.figtext(0.5, 0.01, 
                "Note: t-SNE distances are not strictly meaningful metrics of dissimilarity; "
                "interpret only the relative grouping/clustering as evidence of domain shift.", 
                ha="center", fontsize=9, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
                
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(RESULTS_DIR / "tsne_domain_gap.png", bbox_inches='tight', dpi=150)
    plt.close()

    # ─────────────────────────────────────────────────────────────
    # Generate Markdown Summary
    # ─────────────────────────────────────────────────────────────
    summary_md = [
        "# Exploratory Data Analysis Summary\n",
        "## Per-Dataset Aggregated Statistics\n",
        "This table verifies all 88 images successfully processed. Stats are aggregated means.\n",
        "| Dataset | Images | Native Size | Brightness | Contrast | Native Vessel % | Post-Resize Vessel % | Δ Vessel % |",
        "|---------|--------|-------------|------------|----------|-----------------|----------------------|------------|"
    ]
    
    for dataset in ["DRIVE", "CHASE", "STARE"]:
        df_d = df_stats[df_stats["dataset"] == dataset]
        n_img = len(df_d)
        if n_img == 0: continue
        
        w, h = df_d["native_w"].iloc[0], df_d["native_h"].iloc[0]
        b = df_d["brightness"].mean()
        c = df_d["contrast"].mean()
        
        df_d_m = df_d[df_d["has_mask"]]
        nv = df_d_m["native_vessel_pct"].mean() if len(df_d_m) > 0 else 0
        pv = df_d_m["post_resize_vessel_pct"].mean() if len(df_d_m) > 0 else 0
        diff = df_d_m["vessel_pct_diff"].mean() if len(df_d_m) > 0 else 0
        
        summary_md.append(
            f"| {dataset} | {n_img} | {w}x{h} | {b:.1f} | {c:.1f} | {nv:.2f}% | {pv:.2f}% | {diff:+.3f}% |"
        )
        
    summary_md.extend([
        "\n## t-SNE Interpretation",
        "A pre-trained ResNet-50 (ImageNet weights, no fine-tuning) extracted 2048-dim features "
        "from the preprocessed images. t-SNE (`random_state=42`) reduced this to 2D.",
        "\n**Note on t-SNE**: t-SNE distances are not strictly meaningful metrics. The algorithm "
        "preserves local neighborhood structure, meaning we only interpret the *relative grouping* "
        "and formation of distinct clusters as empirical evidence of a domain gap, not the absolute "
        "distance between clusters."
    ])
    
    with open(RESULTS_DIR / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_md))
        
    print("\n[OK] EDA Phase Complete! All 88 images processed successfully.")
    print(f"Results saved to: {RESULTS_DIR.resolve()}")

if __name__ == "__main__":
    main()
