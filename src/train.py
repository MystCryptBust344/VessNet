import os
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import deque

from src.datasets.vessel_dataset import VesselDataset
from src.models.unet import UNet
from src.models.unetpp import UNetPlusPlus
from src.models.swin_unet import SwinUNet
from src.losses.bce_dice import BCEDiceLoss
from src.metrics.metrics import compute_metrics

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model(model_name, device):
    if model_name == 'unet':
        return UNet(in_channels=3, out_channels=1).to(device)
    elif model_name == 'unetpp':
        return UNetPlusPlus(in_channels=3, out_channels=1).to(device)
    elif model_name == 'swin_unet':
        return SwinUNet(in_channels=3, out_channels=1).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'unetpp', 'swin_unet'])
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, Model: {args.model}")
    
    # Load splits
    split_path = "configs/baseline_split.json"
    with open(split_path, 'r') as f:
        splits = json.load(f)
        
    train_ids = splits['train']
    val_ids = splits['val']
    
    print(f"Loaded splits: {len(train_ids)} train, {len(val_ids)} val")
    
    # Datasets and Loaders
    train_dataset = VesselDataset(image_ids=train_ids)
    val_dataset = VesselDataset(image_ids=val_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    # Model, Optimizer, Loss
    model = get_model(args.model, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = BCEDiceLoss(bce_weight=0.4, dice_weight=0.6)
    
    epochs = 40
    best_smoothed_dice = 0.0
    val_dice_history = deque(maxlen=3) # Smoothing over last 3 epochs
    
    log_file = f"checkpoints/{args.model}_training_log.csv"
    os.makedirs("checkpoints", exist_ok=True)
    
    logs = []
    
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item() * imgs.size(0)
                
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu())
                all_targets.append(masks.cpu())
                
        val_loss /= len(val_loader.dataset)
        
        # Compute metrics
        all_probs = torch.cat(all_probs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        val_metrics = compute_metrics(all_probs, all_targets)
        val_dice = val_metrics['Dice']
        
        # Smoothing
        val_dice_history.append(val_dice)
        smoothed_dice = sum(val_dice_history) / len(val_dice_history)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}, Smoothed Dice={smoothed_dice:.4f}")
        
        # Logging
        logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "smoothed_val_dice": smoothed_dice
        })
        pd.DataFrame(logs).to_csv(log_file, index=False)
        
        # Checkpoint saving
        if smoothed_dice > best_smoothed_dice:
            best_smoothed_dice = smoothed_dice
            torch.save(model.state_dict(), f"checkpoints/best_{args.model}.pth")
            print(f"--> Saved new best checkpoint! (Smoothed Dice: {best_smoothed_dice:.4f})")
            
    print("Training complete!")

if __name__ == "__main__":
    main()
