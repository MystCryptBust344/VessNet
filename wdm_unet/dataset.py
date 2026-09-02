from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".ppm"}


def read_image(path: str):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: str):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (m > 127).astype(np.float32)


def preprocess_fundus(img_rgb, input_mode="gray", clahe=True):
    if input_mode == "green":
        x = img_rgb[..., 1]
    else:
        x = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        x = c.apply(x.astype(np.uint8))

    x = x.astype(np.float32)
    # Paper: zero mean/unit variance, then linear rescale. For neural training,
    # final standardized values are more useful; this preserves their normalization intent.
    x = (x - x.mean()) / (x.std() + 1e-6)
    return x


class RetinalVesselDataset(Dataset):
    def __init__(self, image_paths, mask_paths, size=565, train=False, input_mode="gray"):
        self.image_paths = list(map(str, image_paths))
        self.mask_paths = list(map(str, mask_paths))
        assert len(self.image_paths) == len(self.mask_paths)
        self.size = int(size)
        self.train = train
        self.input_mode = input_mode

    def __len__(self):
        return len(self.image_paths)

    def _augment(self, img, mask):
        # Paper's eightfold geometric family: rotations {0,90,180,270} plus flip variants.
        k = np.random.randint(0, 4)
        img = np.rot90(img, k).copy()
        mask = np.rot90(mask, k).copy()
        if np.random.rand() < 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        if np.random.rand() < 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
        return img, mask

    def __getitem__(self, idx):
        img = read_image(self.image_paths[idx])
        mask = read_mask(self.mask_paths[idx])

        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        x = preprocess_fundus(img, self.input_mode)

        if self.train:
            x, mask = self._augment(x, mask)

        x = torch.from_numpy(x[None]).float()
        y = torch.from_numpy(mask[None]).float()
        return x, y


def pair_by_stem(image_dir, mask_dir):
    images = [p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMG_EXTS]
    masks = [p for p in Path(mask_dir).iterdir() if p.suffix.lower() in IMG_EXTS]

    def key(p):
        s = p.stem.lower()
        for token in ["_manual1", "_manual", "_mask", "_label", "_gt", "_1stho", "_2ndho", "_training", "_test"]:
            s = s.replace(token, "")
        return s

    mask_map = {key(p): p for p in masks}
    pairs = []
    for p in sorted(images):
        k = key(p)
        if k in mask_map:
            pairs.append((str(p), str(mask_map[k])))
    if not pairs:
        raise RuntimeError(
            "No image/mask pairs found automatically. Rename matching pairs to the same stem, "
            "e.g. 01.png and 01_mask.png."
        )
    return list(zip(*pairs))
