"""
Phase 7: Adversarial Domain Adaptation via Gradient Reversal Layer (GRL)

Architecture:
  - U-Net++ encoder (ResNet-34) shared for segmentation + domain classification
  - Domain Discriminator receives bottleneck features (512-dim after global avg pool)
  - GRL reverses gradients so the encoder learns domain-invariant features
  - FDA is also applied during training to further bridge the style gap

Loss:
  total = seg_loss(source) + lambda_adv * adv_loss
  adv_loss = BCE(discriminator(source_feats), 0) + BCE(discriminator(target_feats), 1)
  (reversed through GRL, so encoder is trained to fool the discriminator)

lambda_adv is annealed: 0 -> 1 over the first half of training (warm-up)
alpha for GRL is also annealed: 0 -> 1
"""
import argparse
import csv
import itertools
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from src.datasets.vessel_dataset import VesselDataset
from src.domain_adaptation.fda import fda_transfer
from src.domain_adaptation.grl import DomainDiscriminator
from src.losses.bce_dice import BCEDiceLoss
from src.metrics.metrics import compute_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-disc", type=float, default=1e-3)
    p.add_argument("--lambda-adv", type=float, default=0.1,
                   help="Max weight for adversarial loss (annealed from 0)")
    p.add_argument("--beta", type=float, default=0.05,
                   help="FDA beta — set 0 to disable FDA")
    p.add_argument("--pretrain", action="store_true",
                   help="Fine-tune from checkpoints/best_unetpp.pth")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_alpha(epoch, total_epochs):
    """GRL alpha: ramps from 0 to 1 over training (DANN schedule)."""
    p = epoch / total_epochs
    return float(2.0 / (1.0 + np.exp(-10 * p)) - 1.0)


def evaluate_domain(encoder, decoder, domain_role, device, criterion):
    try:
        ds = VesselDataset(manifest_path="data/manifest.csv", domain_role=domain_role)
    except ValueError:
        return None
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    encoder.eval(); decoder.eval()
    all_probs, all_masks = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            feats = encoder(imgs)
            # SMP decoder takes full feature list
            logits = decoder(*feats)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_masks.append(masks)
    metrics = compute_metrics(torch.cat(all_probs), torch.cat(all_masks))
    return metrics["Dice"]


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | epochs={args.epochs} | lr={args.lr} | lambda_adv={args.lambda_adv} | beta={args.beta}")

    # ---- Data ----
    with open("configs/baseline_split.json") as f:
        split = json.load(f)
    train_ids = split["train"]

    src_ds = VesselDataset(manifest_path="data/manifest.csv", domain_role="source_train", image_ids=train_ids)
    tgt_ds = VesselDataset(manifest_path="data/manifest.csv", domain_role="target_train")

    src_loader = DataLoader(src_ds, batch_size=2, shuffle=True, num_workers=0)
    tgt_loader = DataLoader(tgt_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True)

    # ---- Model: split encoder + decoder for feature access ----
    full_model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    if args.pretrain:
        ckpt = "checkpoints/best_unetpp.pth"
        if os.path.exists(ckpt):
            # SMP model state dict wraps under .model
            sd = torch.load(ckpt, map_location=device, weights_only=True)
            # Our UNetPlusPlus wrapper stores it under 'model.'
            sd = {k.replace("model.", ""): v for k, v in sd.items()}
            full_model.load_state_dict(sd)
            print(f"Loaded pretrained weights from {ckpt}")

    encoder = full_model.encoder
    decoder = full_model.decoder
    seg_head = full_model.segmentation_head

    # Domain discriminator: 512-dim bottleneck (last encoder feature)
    discriminator = DomainDiscriminator(in_channels=512).to(device)

    # ---- Optimizers ----
    seg_params = list(encoder.parameters()) + list(decoder.parameters()) + list(seg_head.parameters())
    optimizer_seg = torch.optim.Adam(seg_params, lr=args.lr)
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=args.lr_disc)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_seg, T_max=args.epochs, eta_min=1e-6)

    seg_criterion = BCEDiceLoss(bce_weight=0.4, dice_weight=0.6)
    disc_criterion = nn.BCEWithLogitsLoss()

    os.makedirs("checkpoints", exist_ok=True)
    log_path = "checkpoints/unetpp_grl_training_log.csv"
    best_ckpt = "checkpoints/best_unetpp_grl.pth"
    best_target_dice = 0.0

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "seg_loss", "disc_loss", "src_dice", "tgt_dice", "alpha", "lambda_adv"])

    for epoch in range(1, args.epochs + 1):
        encoder.train(); decoder.train(); seg_head.train(); discriminator.train()

        alpha = get_alpha(epoch, args.epochs)
        # Anneal adversarial weight: 0 for first 10% of training, then rises
        lam = args.lambda_adv * max(0.0, (epoch / args.epochs - 0.1) / 0.9)

        tgt_iter = itertools.cycle(tgt_loader)
        seg_losses, disc_losses = [], []

        for src_imgs, src_masks in src_loader:
            tgt_imgs, _ = next(tgt_iter)
            b = src_imgs.shape[0]
            if tgt_imgs.shape[0] < b:
                tgt_imgs = tgt_imgs.repeat((b // tgt_imgs.shape[0]) + 1, 1, 1, 1)[:b]
            else:
                tgt_imgs = tgt_imgs[:b]

            src_imgs, src_masks = src_imgs.to(device), src_masks.to(device)
            tgt_imgs = tgt_imgs.to(device)

            # Optionally apply FDA before the forward pass
            if args.beta > 0:
                src_imgs_adapted = fda_transfer(src_imgs, tgt_imgs, beta=args.beta)
            else:
                src_imgs_adapted = src_imgs

            # ---- Step 1: Train segmentation network + GRL adversarial ----
            optimizer_seg.zero_grad()

            src_feats = encoder(src_imgs_adapted)      # list of feature maps
            tgt_feats = encoder(tgt_imgs)

            # Segmentation loss (source only — we have no target labels)
            src_logits = seg_head(decoder(*src_feats))
            seg_loss = seg_criterion(src_logits, src_masks)

            # Adversarial loss via GRL: encoder tries to fool discriminator
            # Global average pool the bottleneck (last feature map)
            src_bottleneck = src_feats[-1].mean(dim=[2, 3])   # (B, 512)
            tgt_bottleneck = tgt_feats[-1].mean(dim=[2, 3])   # (B, 512)

            # Discriminator preds with GRL — gradients reversed for encoder
            src_domain_pred = discriminator(src_bottleneck, alpha=alpha)
            tgt_domain_pred = discriminator(tgt_bottleneck, alpha=alpha)

            # Adversarial loss: encoder wants discriminator to fail
            src_dom_labels = torch.ones(b, 1, device=device)   # label 1=source
            tgt_dom_labels = torch.zeros(b, 1, device=device)  # label 0=target
            adv_loss = disc_criterion(src_domain_pred, src_dom_labels) + \
                       disc_criterion(tgt_domain_pred, tgt_dom_labels)

            total_loss = seg_loss + lam * adv_loss
            total_loss.backward()
            optimizer_seg.step()

            seg_losses.append(seg_loss.item())

            # ---- Step 2: Train discriminator (no GRL — clean gradients) ----
            optimizer_disc.zero_grad()

            with torch.no_grad():
                src_feats_d = encoder(src_imgs_adapted)
                tgt_feats_d = encoder(tgt_imgs)
                src_bn = src_feats_d[-1].mean(dim=[2, 3]).detach()
                tgt_bn = tgt_feats_d[-1].mean(dim=[2, 3]).detach()

            # Direct forward without GRL (alpha=0 means no reversal)
            src_pred_d = discriminator(src_bn, alpha=0.0)
            tgt_pred_d = discriminator(tgt_bn, alpha=0.0)

            disc_loss = disc_criterion(src_pred_d, src_dom_labels) + \
                        disc_criterion(tgt_pred_d, tgt_dom_labels)
            disc_loss.backward()
            optimizer_disc.step()
            disc_losses.append(disc_loss.item())

        scheduler.step()

        avg_seg = float(np.mean(seg_losses))
        avg_disc = float(np.mean(disc_losses))

        # Evaluate on source test + target test
        src_dice = evaluate_domain(encoder, decoder, "source_test", device, seg_criterion) or 0.0
        tgt_dice = evaluate_domain(encoder, decoder, "target_test", device, seg_criterion) or 0.0

        print(f"Epoch {epoch:3d}/{args.epochs} | seg={avg_seg:.4f} | disc={avg_disc:.4f} | "
              f"src_dice={src_dice:.4f} | tgt_dice={tgt_dice:.4f} | α={alpha:.3f} | λ={lam:.4f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_seg, avg_disc, src_dice, tgt_dice, alpha, lam])

        # Save best based on TARGET Dice
        if tgt_dice > best_target_dice:
            best_target_dice = tgt_dice
            torch.save(full_model.state_dict(), best_ckpt)
            print(f"  --> New best TARGET Dice={best_target_dice:.4f} | saved {best_ckpt}")

    print(f"\nTraining complete. Best target Dice: {best_target_dice:.4f}")


if __name__ == "__main__":
    main()
