import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from src.datasets.vessel_dataset import VesselDataset
from src.metrics.metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_unetpp_grl.pth")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded GRL model from {args.checkpoint}")

    domains = ["source_test", "target_test", "external_test"]
    rows = []

    for domain in domains:
        try:
            ds = VesselDataset(manifest_path="data/manifest.csv", domain_role=domain)
        except ValueError:
            print(f"  [{domain}] No data found, skipping.")
            continue

        loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        all_probs, all_masks = [], []

        with torch.no_grad():
            for imgs, masks in loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                all_probs.append(torch.sigmoid(logits).cpu())
                all_masks.append(masks)

        metrics = compute_metrics(torch.cat(all_probs), torch.cat(all_masks))
        print(f"[{domain}] Dice={metrics['Dice']:.4f} | Sensitivity={metrics['Sensitivity']:.4f} | "
              f"Specificity={metrics['Specificity']:.4f} | Accuracy={metrics['Accuracy']:.4f}")
        rows.append({"model": "unetpp_grl", "domain": domain, **metrics})

    os.makedirs("results", exist_ok=True)
    out_path = "results/grl_evaluation_results.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Comparison vs baseline and FDA
    for ref_path, label in [("results/baselines_results.csv", "Baseline U-Net++"),
                             ("results/fda_evaluation_results.csv", "FDA U-Net++")]:
        if not os.path.exists(ref_path):
            continue
        ref_df = pd.read_csv(ref_path)
        # Handle both column naming conventions
        domain_col = "Test Domain" if "Test Domain" in ref_df.columns else "domain"
        dice_col = "Dice"
        model_col = "Model" if "Model" in ref_df.columns else "model"
        model_key = "unetpp" if label == "Baseline U-Net++" else "unetpp_fda"
        ref_filtered = ref_df[ref_df[model_col] == model_key] if model_col in ref_df.columns else ref_df

        print(f"\nGRL vs {label}:")
        for row in rows:
            domain = row["domain"]
            match = ref_filtered[ref_filtered[domain_col] == domain]
            if match.empty:
                continue
            ref_dice = match[dice_col].values[0]
            delta = row["Dice"] - ref_dice
            print(f"  {domain}: {delta:+.4f}  (GRL={row['Dice']:.4f}, {label}={ref_dice:.4f})")


if __name__ == "__main__":
    main()
