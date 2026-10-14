import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # save to files, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.vessel_dataset import VesselDataset
from src.train import get_model


def parse_args():
    p = argparse.ArgumentParser(description="Visualise input / ground truth / predicted mask from a saved model.")
    p.add_argument("--model", default="unetpp", choices=["unet", "unetpp", "swin_unet"])
    p.add_argument("--ckpt", default=None,
                   help="Checkpoint path. Default: checkpoints/best_{model}.pth (or best_{model}_fda.pth with --fda)")
    p.add_argument("--fda", action="store_true", help="Use the FDA-trained checkpoint")
    p.add_argument("--split", default="val", choices=["train", "val", "domain"],
                   help="train/val: ids from configs/baseline_split.json. domain: use --domain_role from the manifest")
    p.add_argument("--domain_role", default="target_test",
                   help="Used when --split domain (e.g. source_test, target_test, external_test)")
    p.add_argument("--num", type=int, default=10, help="How many images to visualise (0 = all)")
    p.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for the binary mask")
    p.add_argument("--out_dir", default="results/predictions")
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()


def to_display_image(img_tensor):
    """CHW tensor -> HWC float array in [0, 1] for plotting (min-max, so it works even if normalised)."""
    img = img_tensor.detach().cpu().float().numpy()
    img = np.transpose(img, (1, 2, 0)) if img.ndim == 3 else img
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-8)


def dice_score(pred, gt, eps=1e-8):
    inter = (pred * gt).sum()
    return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load the best checkpoint ----
    ckpt = args.ckpt or (f"checkpoints/best_{args.model}_fda.pth" if args.fda
                         else f"checkpoints/best_{args.model}.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model = get_model(args.model, device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state and not any(k.startswith(("enc", "dec")) for k in state):
        state = state["model"]  # in case a full-training checkpoint was saved
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {ckpt} on {device}")

    # ---- Dataset ----
    if args.split == "domain":
        ds = VesselDataset(manifest_path="data/manifest.csv", domain_role=args.domain_role)
        tag = args.domain_role
    else:
        with open("configs/baseline_split.json") as f:
            ids = json.load(f)[args.split]
        ds = VesselDataset(image_ids=ids)
        tag = args.split
    n_total = len(ds) if args.num == 0 else min(args.num, len(ds))
    print(f"Visualising {n_total} of {len(ds)} images from '{tag}'")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    save_dir = os.path.join(args.out_dir, f"{args.model}{'_fda' if args.fda else ''}_{tag}")
    os.makedirs(save_dir, exist_ok=True)

    done = 0
    dices = []
    with torch.no_grad():
        for imgs, masks in loader:
            probs = torch.sigmoid(model(imgs.to(device))).cpu()
            preds = (probs > args.threshold).float()

            for i in range(imgs.size(0)):
                if done >= n_total:
                    break
                gt = masks[i, 0].numpy()
                pr = preds[i, 0].numpy()
                d = dice_score(pr, (gt > 0.5).astype(np.float32))
                dices.append(d)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(to_display_image(imgs[i]))
                axes[0].set_title("Input image")
                axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
                axes[1].set_title("Ground truth")
                axes[2].imshow(pr, cmap="gray", vmin=0, vmax=1)
                axes[2].set_title(f"Predicted mask (Dice {d:.3f})")
                for ax in axes:
                    ax.axis("off")
                plt.tight_layout()
                out_path = os.path.join(save_dir, f"sample_{done:03d}.png")
                plt.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                done += 1
            if done >= n_total:
                break

    print(f"Saved {done} figures to {save_dir}")
    print(f"Mean Dice over these {done} images: {np.mean(dices):.4f}")


if __name__ == "__main__":
    main()