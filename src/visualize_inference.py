import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from src.datasets.vessel_dataset import VesselDataset
from src.train import get_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unetpp', choices=['unet', 'unetpp', 'swin_unet'])
    parser.add_argument('--image_idx', type=int, default=0, help="Index of the image in the test set to visualize")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading {args.model} on {device}...")

    # Load the model and its best weights
    model = get_model(args.model, device)
    ckpt_path = f"checkpoints/best_{args.model}.pth"
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found!")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # We evaluate on source_test (DRIVE test set)
    test_dataset = VesselDataset(domain_role='source_test')
    if args.image_idx >= len(test_dataset):
        print(f"Error: image_idx {args.image_idx} is out of bounds (dataset size: {len(test_dataset)})")
        return
        
    # Get the data
    img_tensor, mask_tensor = test_dataset[args.image_idx]
    
    # Original paths for displaying name
    row = test_dataset.df.iloc[args.image_idx]
    image_name = os.path.basename(row['image_path'])

    # Run inference
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).float().squeeze().cpu().numpy()
        
    # Prepare images for plotting
    # Convert image tensor (C, H, W) back to numpy (H, W, C) [0, 255]
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gt_mask_np = mask_tensor.squeeze().numpy()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title(f"Input Image ({image_name})")
    axes[0].axis("off")
    
    axes[1].imshow(gt_mask_np, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")
    
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title(f"Predicted Mask ({args.model})")
    axes[2].axis("off")
    
    plt.tight_layout()
    out_path = f"results/inference_{args.model}_{image_name}.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved inference plot to {out_path}")

if __name__ == "__main__":
    main()
