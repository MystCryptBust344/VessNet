import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.vessel_dataset import VesselDataset
from src.losses.bce_dice import BCEDiceLoss
from src.metrics.metrics import compute_metrics
from src.train import get_model


DOMAINS = ["source_test", "target_test", "external_test"]
METRIC_COLS = ["Dice", "Sensitivity", "Specificity", "Accuracy", "AUC_ROC"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unetpp")
    return p.parse_args()


def evaluate_domain(model, domain_role, device, criterion):
    try:
        ds = VesselDataset(manifest_path="data/manifest.csv", domain_role=domain_role)
    except Exception:
        return None
    if len(ds) == 0:
        return None

    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    model.eval()
    all_probs, all_masks, losses = [], [], []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            losses.append(criterion(logits, masks).item())
            all_probs.append(torch.sigmoid(logits).cpu())
            all_masks.append(masks.cpu())

    probs = torch.cat(all_probs)
    masks = torch.cat(all_masks)
    return compute_metrics(probs, masks)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"checkpoints/best_{args.model}_fda.pth"
    model = get_model(args.model, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"Loaded FDA model from {ckpt_path}")

    criterion = BCEDiceLoss(bce_weight=0.4, dice_weight=0.6)

    rows = []
    for domain in DOMAINS:
        metrics = evaluate_domain(model, domain, device, criterion)
        if metrics is None:
            print(f"[SKIP] {domain}: no data found")
            continue
        row = {"model": f"{args.model}_fda", "domain": domain}
        row.update({k: metrics.get(k, float("nan")) for k in METRIC_COLS})
        rows.append(row)
        print(f"[{domain}] " + " | ".join(f"{k}={row[k]:.4f}" for k in METRIC_COLS))

    os.makedirs("results", exist_ok=True)
    out_path = "results/fda_evaluation_results.csv"
    df = pd.DataFrame(rows, columns=["model", "domain"] + METRIC_COLS)
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

    baseline_path = "results/baselines_results.csv"
    if os.path.exists(baseline_path):
        base_df = pd.read_csv(baseline_path)
        base_df_model = base_df[base_df["Model"] == args.model]
        print("\nDice improvement (FDA - Baseline):")
        for row in rows:
            domain = row["domain"]
            base_rows = base_df_model[base_df_model["Test Domain"] == domain]
            if base_rows.empty:
                continue
            base_dice = base_rows["Dice"].values[0]
            improvement = row["Dice"] - base_dice
            print(f"  {domain}: {improvement:+.4f}  (FDA={row['Dice']:.4f}, Baseline={base_dice:.4f})")


if __name__ == "__main__":
    main()
