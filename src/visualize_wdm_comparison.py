import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from src.train import get_model
from src.preprocessing.preprocess import preprocess_pair
from wdm_unet.model import WDMUNet
from wdm_unet.dataset import preprocess_fundus

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load UNet++
    unetpp = get_model('unetpp', device)
    unetpp_ckpt = "checkpoints/best_unetpp.pth"
    if os.path.exists(unetpp_ckpt):
        unetpp.load_state_dict(torch.load(unetpp_ckpt, map_location=device))
        unetpp.eval()
    else:
        print(f"Warning: {unetpp_ckpt} not found!")

    # 2. Load WDM-UNet
    wdmunet = WDMUNet(in_channels=1, base=32).to(device)
    wdmunet_ckpt = "wdm_unet/runs/drive_wdm/best.pt"
    if os.path.exists(wdmunet_ckpt):
        ckpt = torch.load(wdmunet_ckpt, map_location=device, weights_only=False)
        wdmunet.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        wdmunet.eval()
    else:
        print(f"Warning: {wdmunet_ckpt} not found!")
        
    df = pd.read_csv("data/manifest.csv")
    
    # Pick the same 4 images
    target_names = ['01_test.tif', '02_test.tif', 'im0001.ppm', 'im0002.ppm']
    selected_rows = []
    for name in target_names:
        matches = df[df['image_path'].str.contains(name, regex=False)]
        if len(matches) > 0:
            selected_rows.append(matches.iloc[0])

    if len(selected_rows) < 4:
        test_df = df[df['domain_role'].str.contains('test')]
        selected_rows = [test_df.iloc[i] for i in range(min(4, len(test_df)))]

    # We will create one plot with: Input | GT | UNet++ | WDM-UNet
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    col_titles = ['Original Image', 'Ground Truth', 'Our UNet++', 'WDM-UNet']
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=16, fontweight='bold', pad=10)

    for i, row in enumerate(selected_rows):
        img_path = row['image_path']
        mask_path = row['mask_path']
        
        # Load raw
        img_np = cv2.imread(img_path)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # --- Preprocessing for UNet++ ---
        img_unetpp, mask_unetpp = preprocess_pair(img_np, mask_np)
        tensor_unetpp = torch.from_numpy(img_unetpp.transpose((2, 0, 1))).float() / 255.0
        
        # --- Preprocessing for WDM-UNet ---
        img_wdm = cv2.resize(img_np, (565, 565), interpolation=cv2.INTER_AREA)
        x_wdm = preprocess_fundus(img_wdm, input_mode="gray")
        tensor_wdm = torch.from_numpy(x_wdm[None]).float()
        
        # Plot Input & GT
        axes[i, 0].imshow(img_unetpp)
        axes[i, 0].axis('off')
        axes[i, 1].imshow(mask_unetpp, cmap='gray')
        axes[i, 1].axis('off')
        
        with torch.no_grad():
            # UNet++ Inference
            out_unetpp = unetpp(tensor_unetpp.unsqueeze(0).to(device))
            pred_unetpp = (torch.sigmoid(out_unetpp) > 0.5).float().squeeze().cpu().numpy()
            
            # WDM-UNet Inference
            out_wdm = wdmunet(tensor_wdm.unsqueeze(0).to(device))
            pred_wdm = (torch.sigmoid(out_wdm) > 0.5).float().squeeze().cpu().numpy()
            
        axes[i, 2].imshow(pred_unetpp, cmap='gray')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(pred_wdm, cmap='gray')
        axes[i, 3].axis('off')

    plt.tight_layout()
    out_path = "results/wdm_unetpp_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved comparison grid to {out_path}")

if __name__ == "__main__":
    main()
