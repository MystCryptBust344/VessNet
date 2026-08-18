import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

def create_baseline_split():
    manifest_path = "data/manifest.csv"
    output_path = "configs/baseline_split.json"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = pd.read_csv(manifest_path)
    
    # Filter only source_train (DRIVE training set)
    source_train_df = df[df['domain_role'] == 'source_train']
    
    if len(source_train_df) != 20:
        raise ValueError(f"Expected 20 source_train images, found {len(source_train_df)}")
        
    image_ids = source_train_df['image_id'].tolist()
    
    # Create a 16/4 split using random_state=42
    train_ids, val_ids = train_test_split(image_ids, test_size=4, random_state=42)
    
    split_dict = {
        "train": sorted(train_ids),
        "val": sorted(val_ids)
    }
    
    with open(output_path, 'w') as f:
        json.dump(split_dict, f, indent=4)
        
    print(f"Saved split to {output_path}")
    print(f"Train (16): {split_dict['train']}")
    print(f"Val (4): {split_dict['val']}")

if __name__ == "__main__":
    create_baseline_split()
