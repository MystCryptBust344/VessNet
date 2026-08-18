import os
import shutil
import pandas as pd

manifest_path = "data/manifest.csv"
df = pd.read_csv(manifest_path)

src_dir = "drive_av_dataset/DRIVE_AV/test/1st_manual"
dst_dir = "data/DRIVE/test/masks"
os.makedirs(dst_dir, exist_ok=True)

# Copy files and update manifest
for i, row in df.iterrows():
    if row['dataset'] == 'DRIVE' and row['split'] == 'test':
        # "DRIVE_01_test" -> subject_id "01"
        subject_id = str(row['subject_id']).zfill(2)
        src_mask = os.path.join(src_dir, f"{subject_id}_test.png")
        if os.path.exists(src_mask):
            dst_mask = os.path.join(dst_dir, f"{subject_id}_test.png")
            shutil.copy2(src_mask, dst_mask)
            # Use forward slashes for cross-platform compatibility
            df.at[i, 'mask_path'] = f"data/DRIVE/test/masks/{subject_id}_test.png"
        else:
            print(f"Warning: {src_mask} not found")

df.to_csv(manifest_path, index=False)
print("Manifest updated successfully.")
