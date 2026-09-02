import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import RetinalVesselDataset, pair_by_stem
from model import WDMUNet
from train import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--size", type=int, default=565)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--input-mode", choices=["gray", "green"], default="gray")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    imgs, masks = pair_by_stem(args.images, args.masks)
    ds = RetinalVesselDataset(imgs, masks, args.size, train=False, input_mode=args.input_mode)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WDMUNet(in_channels=1, base=args.base).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    m = evaluate(model, dl, device, args.threshold)
    for k, v in m.items():
        print(f"{k:12s}: {v*100:.3f}%")


if __name__ == "__main__":
    main()
