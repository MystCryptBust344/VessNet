import os
import torch
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.vessel_dataset import VesselDataset
from src.metrics.metrics import compute_metrics
from src.train import get_model  # Reusing the model factory from train.py

def evaluate_domain(model, device, domain_role, batch_size=2):
    dataset = VesselDataset(domain_role=domain_role)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=f"Evaluating {domain_role}"):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu())
            all_targets.append(masks.cpu())
            
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = compute_metrics(all_probs, all_targets)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'unetpp', 'swin_unet'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, Model: {args.model}")
    
    model = get_model(args.model, device)
    checkpoint_path = f"checkpoints/best_{args.model}.pth"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    domains_to_evaluate = ['source_test', 'target_test', 'external_test']
    results = []
    
    for role in domains_to_evaluate:
        metrics = evaluate_domain(model, device, role)
        
        # Add metadata for the table
        row = {"Model": args.model, "Test Domain": role}
        row.update(metrics)
        results.append(row)
        
        print(f"\nResults for {role}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(results)
    output_path = f"results/baselines_results.csv"
    
    # Append if exists, otherwise create
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        # remove previous runs of the same model so we don't duplicate
        existing_df = existing_df[existing_df['Model'] != args.model]
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
        
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved final evaluation metrics to {output_path}")

if __name__ == "__main__":
    main()
