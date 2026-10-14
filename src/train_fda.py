import argparse
import csv
import itertools
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.vessel_dataset import VesselDataset
from src.domain_adaptation.fda import fda_transfer
from src.losses.bce_dice import BCEDiceLoss
from src.metrics.metrics import compute_metrics
from src.train import get_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unetpp")
    p.add_argument("--beta", type=float, default=0.05,
                   help="FDA low-freq window fraction (0.01=tiny, 0.05=paper default, 0.09=aggressive)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrain", action="store_true",
                   help="If set, fine-tune from checkpoints/best_{model}.pth. Default: train from scratch.")
    return p.parse_args()


def evaluate_domain(model, domain_role, device, criterion):
    """Run inference on an entire domain_role split. Returns avg_loss, dice."""
    try:
        ds = VesselDataset(manifest_path="data/manifest.csv", domain_role=domain_role)
    except ValueError:
        return None, None
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    model.eval()
    losses, all_probs, all_masks = [], [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            losses.append(criterion(logits, masks).item())
            all_probs.append(torch.sigmoid(logits).cpu())
            all_masks.append(masks.cpu())
    loss = float(np.mean(losses))
    metrics = compute_metrics(torch.cat(all_probs), torch.cat(all_masks))
    return loss, metrics["Dice"]


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | beta={args.beta} | epochs={args.epochs} | lr={args.lr} | pretrain={args.pretrain}")

    with open("configs/baseline_split.json") as f:
        split = json.load(f)
    train_ids = split["train"]

    source_train_ds = VesselDataset(manifest_path="data/manifest.csv", domain_role="source_train", image_ids=train_ids)
    target_ds = VesselDataset(manifest_path="data/manifest.csv", domain_role="target_train")

    source_loader = DataLoader(source_train_ds, batch_size=2, shuffle=True, num_workers=0)
    target_loader = DataLoader(target_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True)

    model = get_model(args.model, device)

    if args.pretrain:
        ckpt_path = f"checkpoints/best_{args.model}.pth"
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            print(f"Fine-tuning from {ckpt_path}")
        else:
            print(f"Warning: {ckpt_path} not found. Training from scratch.")
    else:
        print("Training from scratch (no pretrained weights loaded).")

    criterion = BCEDiceLoss(bce_weight=0.4, dice_weight=0.6)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Cosine annealing: smoothly reduces LR to 0 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs("checkpoints", exist_ok=True)
    log_path = f"checkpoints/{args.model}_fda_training_log.csv"
    best_ckpt = f"checkpoints/best_{args.model}_fda.pth"
    best_target_dice = 0.0

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "src_val_dice", "tgt_val_dice", "lr"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        target_iter = itertools.cycle(target_loader)
        total_loss, steps = 0.0, 0

        for src_imgs, src_masks in source_loader:
            tgt_imgs, _ = next(target_iter)
            src_imgs, src_masks = src_imgs.to(device), src_masks.to(device)

            # Ensure batch sizes match
            b = src_imgs.shape[0]
            if tgt_imgs.shape[0] < b:
                tgt_imgs = tgt_imgs.repeat((b // tgt_imgs.shape[0]) + 1, 1, 1, 1)[:b]
            else:
                tgt_imgs = tgt_imgs[:b]
            tgt_imgs = tgt_imgs.to(device)

            adapted_imgs = fda_transfer(src_imgs, tgt_imgs, beta=args.beta)

            optimizer.zero_grad()
            # FDA path: source content + target style
            fda_loss = criterion(model(adapted_imgs), src_masks)
            # Consistency path: keep source domain performance
            src_loss = criterion(model(src_imgs), src_masks)
            loss = 0.5 * fda_loss + 0.5 * src_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        scheduler.step()
        train_loss = total_loss / max(steps, 1)
        current_lr = scheduler.get_last_lr()[0]

        # Evaluate on SOURCE val and TARGET val every epoch
        _, src_dice = evaluate_domain(model, "source_test", device, criterion)
        _, tgt_dice = evaluate_domain(model, "target_test", device, criterion)

        src_dice = src_dice or 0.0
        tgt_dice = tgt_dice or 0.0

        print(f"Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f} | src_dice={src_dice:.4f} | tgt_dice={tgt_dice:.4f} | lr={current_lr:.2e}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, src_dice, tgt_dice, current_lr])

        # *** Checkpoint based on TARGET domain Dice — this is the key fix ***
        if tgt_dice > best_target_dice:
            best_target_dice = tgt_dice
            torch.save(model.state_dict(), best_ckpt)
            print(f"  --> New best TARGET Dice={best_target_dice:.4f} | saved {best_ckpt}")


if __name__ == "__main__":
    main()
