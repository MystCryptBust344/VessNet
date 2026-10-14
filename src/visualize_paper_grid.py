import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from src.train import get_model
from src.preprocessing.preprocess import preprocess_pair

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    models = {
        'U-Net': 'unet',
        'Swin-UNet': 'swin_unet',
        'U-Net++': 'unetpp'
    }
    
    loaded_models = {}
    for display_name, arch in models.items():
        ckpt_path = f"checkpoints/best_{arch}.pth"
        if not os.path.exists(ckpt_path):
            print(f"Warning: {ckpt_path} missing. Skipping {display_name}.")
            continue
        m = get_model(arch, device)
        m.load_state_dict(torch.load(ckpt_path, map_location=device))
        m.eval()
        loaded_models[display_name] = m
        
    df = pd.read_csv("data/manifest.csv")
    
    # Let's try to find 4 diverse images (2 from DRIVE, 2 from STARE)
    # The paper uses images with exudates (STARE) and clean images (DRIVE)
    target_names = ['01_test.tif', '02_test.tif', 'im0001.ppm', 'im0002.ppm']
    
    selected_rows = []
    for name in target_names:
        matches = df[df['image_path'].str.contains(name, regex=False)]
        if len(matches) > 0:
            selected_rows.append(matches.iloc[0])
            
    if len(selected_rows) < 4:
        # Fallback to just grabbing the first 4 test images
        print("Could not find specific target images, falling back to random test images.")
        test_df = df[df['domain_role'].str.contains('test')]
        selected_rows = [test_df.iloc[i] for i in range(min(4, len(test_df)))]

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    
    col_titles = ['Original Image', 'Ground Truth'] + list(loaded_models.keys())
    
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=14, fontweight='bold', pad=10)

    for i, row in enumerate(selected_rows):
        img_path = row['image_path']
        mask_path = row['mask_path']
        
        img_np = cv2.imread(img_path)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        
        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        img_proc, mask_proc = preprocess_pair(img_np, mask_np)
        
        img_tensor = torch.from_numpy(img_proc.transpose((2, 0, 1))).float() / 255.0
        mask_tensor = (torch.from_numpy(mask_proc).float() > 0.5).float()
        
        # 1. Plot Input
        axes[i, 0].imshow(img_proc)
        axes[i, 0].axis('off')
        
        # 2. Plot GT
        axes[i, 1].imshow(mask_proc, cmap='gray')
        axes[i, 1].axis('off')
        
        # Inference
        with torch.no_grad():
            x = img_tensor.unsqueeze(0).to(device)
            
            for j, (name, m) in enumerate(loaded_models.items()):
                logits = m(x)
                probs = torch.sigmoid(logits)
                pred_mask = (probs > 0.5).float().squeeze().cpu().numpy()
                
                axes[i, 2 + j].imshow(pred_mask, cmap='gray')
                axes[i, 2 + j].axis('off')

    plt.tight_layout()
    out_path = "results/paper_grid_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved grid to {out_path}")

if __name__ == "__main__":
    main()
