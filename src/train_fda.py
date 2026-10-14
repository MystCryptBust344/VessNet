import argparse
import csv
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unetpp")
    p.add_argument("--beta", type=float, default=0.05,
                   help="FDA low-freq window fraction (0.01=tiny, 0.05=paper default, 0.09=aggressive)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3, help="peak (initial) learning rate (e.g. 1e-3 or 0.0005)")
    p.add_argument("--min_lr", type=float, default=1e-6, help="final learning rate")
    p.add_argument("--high_epochs", type=int, default=5,
                   help="number of initial epochs to hold at peak high LR before cosine decay starts")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--eval_every", type=int, default=1,
                   help="evaluate source/target every N epochs (the last epoch is always evaluated)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    p.add_argument("--pretrain", action="store_true",
                   help="If set, fine-tune from checkpoints/best_{model}.pth. Default: train from scratch.")
    return p.parse_args()


def build_scheduler(optimizer, epochs, high_epochs, min_lr):
    """Holds LR high at peak for initial epochs, then smoothly decays via Cosine Annealing to min_lr."""
    if high_epochs > 0 and epochs > high_epochs:
        hold = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=high_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - high_epochs, eta_min=min_lr
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[hold, cosine], milestones=[high_epochs]
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)


def infinite_loader(loader):
    """Yield batches forever, reshuffling every pass (itertools.cycle would replay the first pass)."""
    while True:
        for batch in loader:
            yield batch


def make_eval_loader(domain_role, batch_size, loader_kwargs):
    """Build the eval dataset/loader ONCE (the old code rebuilt it every epoch)."""
    try:
        ds = VesselDataset(manifest_path="data/manifest.csv", domain_role=domain_role)
    except ValueError:
        return None
    if len(ds) == 0:
        return None
    return DataLoader(ds, batch_size=batch_size, shuffle=False, **loader_kwargs)


@torch.no_grad()
def evaluate_loader(model, loader, device, use_amp, channels_last):
    """Run inference on a prebuilt loader. Returns Dice (or None if loader missing)."""
    if loader is None:
        return None
    model.eval()
    all_probs, all_masks = [], []
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(imgs)
        all_probs.append(torch.sigmoid(logits.float()).cpu())
        all_masks.append(masks.cpu())
    metrics = compute_metrics(torch.cat(all_probs), torch.cat(all_masks))
    return metrics["Dice"]


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    use_amp = use_cuda and not args.no_amp
    channels_last = use_cuda and args.model in ("unet", "unetpp")
    print(f"Device: {device} | AMP={use_amp} | beta={args.beta} | epochs={args.epochs} | "
          f"lr={args.lr}->{args.min_lr} | bs={args.batch_size} | pretrain={args.pretrain}")

    if use_cuda:
        torch.backends.cudnn.benchmark = True

    with open("configs/baseline_split.json") as f:
        split = json.load(f)
    train_ids = split["train"]

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
    )

    source_train_ds = VesselDataset(manifest_path="data/manifest.csv", domain_role="source_train", image_ids=train_ids)
    target_ds = VesselDataset(manifest_path="data/manifest.csv", domain_role="target_train")

    source_loader = DataLoader(source_train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    target_loader = DataLoader(target_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    target_iter = infinite_loader(target_loader)

    # Eval loaders built once
    src_eval_loader = make_eval_loader("source_test", args.batch_size, loader_kwargs)
    tgt_eval_loader = make_eval_loader("target_test", args.batch_size, loader_kwargs)

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

    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = BCEDiceLoss(bce_weight=0.4, dice_weight=0.6)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = build_scheduler(optimizer, args.epochs, args.high_epochs, args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs("checkpoints", exist_ok=True)
    log_path = f"checkpoints/{args.model}_fda_training_log.csv"
    best_ckpt = f"checkpoints/best_{args.model}_fda.pth"
    best_target_dice = 0.0

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "src_val_dice", "tgt_val_dice", "lr"])

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]  # LR used during this epoch
        model.train()
        total_loss, steps = 0.0, 0

        for src_imgs, src_masks in source_loader:
            tgt_imgs, _ = next(target_iter)
            src_imgs = src_imgs.to(device, non_blocking=True)
            src_masks = src_masks.to(device, non_blocking=True)

            # Ensure batch sizes match
            b = src_imgs.shape[0]
            if tgt_imgs.shape[0] < b:
                tgt_imgs = tgt_imgs.repeat((b // tgt_imgs.shape[0]) + 1, 1, 1, 1)[:b]
            else:
                tgt_imgs = tgt_imgs[:b]
            tgt_imgs = tgt_imgs.to(device, non_blocking=True)

            # FDA uses FFT: keep it in float32, outside autocast
            adapted_imgs = fda_transfer(src_imgs, tgt_imgs, beta=args.beta)

            if channels_last:
                adapted_imgs = adapted_imgs.contiguous(memory_format=torch.channels_last)
                src_imgs = src_imgs.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                adapted_logits = model(adapted_imgs)   # FDA path: source content + target style
                src_logits = model(src_imgs)           # Consistency path: keep source performance
            # Losses in float32 for numerical safety
            fda_loss = criterion(adapted_logits.float(), src_masks)
            src_loss = criterion(src_logits.float(), src_masks)
            loss = 0.5 * fda_loss + 0.5 * src_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            steps += 1

        scheduler.step()
        train_loss = total_loss / max(steps, 1)

        # Evaluate on SOURCE and TARGET (skipped on off-epochs to save time)
        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if do_eval:
            src_dice = evaluate_loader(model, src_eval_loader, device, use_amp, channels_last) or 0.0
            tgt_dice = evaluate_loader(model, tgt_eval_loader, device, use_amp, channels_last) or 0.0
            print(f"Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f} | src_dice={src_dice:.4f} | "
                  f"tgt_dice={tgt_dice:.4f} | lr={current_lr:.2e}")
        else:
            src_dice = tgt_dice = float("nan")
            print(f"Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f} | lr={current_lr:.2e}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, src_dice, tgt_dice, current_lr])

        # Checkpoint based on TARGET domain Dice
        if do_eval and tgt_dice > best_target_dice:
            best_target_dice = tgt_dice
            torch.save(model.state_dict(), best_ckpt)
            print(f"  --> New best TARGET Dice={best_target_dice:.4f} | saved {best_ckpt}")


if __name__ == "__main__":
    main()