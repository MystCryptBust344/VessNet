import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from src.preprocessing.preprocess import preprocess_pair

import cv2

class VesselDataset(Dataset):
    def __init__(self, manifest_path="data/manifest.csv", domain_role=None, image_ids=None):
        """
        Args:
            manifest_path: Path to the main manifest CSV file.
            domain_role: If specified, filter by domain_role (e.g., 'source_test').
            image_ids: If specified, filter to only include these image_ids (e.g., for train/val split).
        """
        self.df = pd.read_csv(manifest_path)
        
        if domain_role:
            self.df = self.df[self.df['domain_role'] == domain_role]
            
        if image_ids is not None:
            self.df = self.df[self.df['image_id'].isin(image_ids)]
            
        if len(self.df) == 0:
            raise ValueError(f"Dataset is empty after filtering! role={domain_role}, ids={len(image_ids) if image_ids else 'all'}")
            
        # Reset index so we can use iloc safely
        self.df = self.df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        mask_path = row['mask_path']
        
        # Load images
        img_np = cv2.imread(image_path)
        if img_np is not None:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        else:
            raise RuntimeError(f"Could not load image: {image_path}")
            
        if pd.isna(mask_path):
            # For tests that don't have masks, create a dummy mask
            mask_np = np.zeros(img_np.shape[:2], dtype=np.uint8)
        else:
            mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_np is None:
                raise RuntimeError(f"Could not load mask: {mask_path}")
        
        # 1. Apply canonical preprocessing (Phase 1)
        # Returns: image shape (512, 512, 3), mask shape (512, 512)
        # Both are numpy arrays. Image is uint8 [0, 255], Mask is float32/uint8 [0, 1]
        img_np, mask_np = preprocess_pair(img_np, mask_np)
        
        # 2. Convert to tensors
        # Image: [H, W, C] -> [C, H, W] and normalize to [0, 1] FloatTensor
        img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).float() / 255.0
        
        # Mask: [H, W] -> [1, H, W] and ensure float32 in [0, 1]
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
        
        # Just to be absolutely safe on binarization
        mask_tensor = (mask_tensor > 0.5).float()
        
        return img_tensor, mask_tensor
