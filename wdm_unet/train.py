import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from dataset import RetinalVesselDataset, pair_by_stem
from model import WDMUNet


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bw = bce_weight
        self.dw = dice_weight
        self.smooth = smooth

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (probs * target).sum(dims)
        den = probs.sum(dims) + target.sum(dims)
        dice = ((2 * inter + self.smooth) / (den + self.smooth)).mean()
        return self.bw * bce + self.dw * (1 - dice)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    tp = tn = fp = fn = 0.0
    auc_y, auc_p = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = torch.sigmoid(model(x))
        pred = p >= threshold
        yy = y >= 0.5

        tp += (pred & yy).sum().item()
        tn += ((~pred) & (~yy)).sum().item()
        fp += (pred & (~yy)).sum().item()
        fn += ((~pred) & yy).sum().item()

        auc_y.append(y.flatten().cpu().numpy())
        auc_p.append(p.flatten().cpu().numpy())

    eps = 1e-8
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    se = tp / (tp + fn + eps)
    sp = tn / (tn + fp + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2 * precision * se / (precision + se + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)

    y_true = np.concatenate(auc_y)
    y_prob = np.concatenate(auc_p)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return {"acc": acc, "sensitivity": se, "specificity": sp, "f1": f1, "dice": dice, "auc": auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=565)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--input-mode", choices=["gray", "green"], default="gray")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--output", default="runs/wdm_unet")
    ap.add_argument("--bce-weight", type=float, default=0.5)
    ap.add_argument("--dice-weight", type=float, default=0.5)
    args = ap.parse_args()

    seed_all(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    image_paths, mask_paths = pair_by_stem(args.images, args.masks)
    n = len(image_paths)
    indices = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    n_val = max(1, int(round(n * args.val_frac)))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_ds = RetinalVesselDataset(image_paths, mask_paths, args.size, train=True, input_mode=args.input_mode)
    val_ds = RetinalVesselDataset(image_paths, mask_paths, args.size, train=False, input_mode=args.input_mode)
    train_loader = DataLoader(Subset(train_ds, train_idx.tolist()), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(Subset(val_ds, val_idx.tolist()), batch_size=1, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WDMUNet(in_channels=1, base=args.base).to(device)
    criterion = BCEDiceLoss(args.bce_weight, args.dice_weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Paper: LR x0.1 every 50 epochs.
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = -1.0
    history = []
    print(f"device={device} train={len(train_idx)} val={len(val_idx)}")
    print(f"parameters={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)

        sched.step()
        metrics = evaluate(model, val_loader, device)
        train_loss = running / max(1, len(train_idx))
        score = metrics["f1"]
        row = {"epoch": epoch, "train_loss": train_loss, "lr": opt.param_groups[0]["lr"], **metrics}
        history.append(row)
        print(
            f"E{epoch:03d} loss={train_loss:.4f} "
            f"Acc={metrics['acc']*100:.2f} Se={metrics['sensitivity']*100:.2f} "
            f"Sp={metrics['specificity']*100:.2f} F1={metrics['f1']*100:.2f} "
            f"Dice={metrics['dice']*100:.2f} AUC={metrics['auc']*100:.2f}"
        )

        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "args": vars(args), "metrics": metrics}, out / "best.pt")

        with open(out / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"Best validation F1={best*100:.2f}% | checkpoint={out/'best.pt'}")


if __name__ == "__main__":
    main()
