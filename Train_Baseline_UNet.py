#!/usr/bin/env python3

import os, glob, json
from PIL import Image
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor

CLEAN_ROOT = "data/clean_png"
NOISY_ROOT = "data/noisy_png"
SPLIT_JSON = "data/splits/split_by_patient.json"
OUTDIR = "outputs/unet_baseline"

class PairDataset(Dataset):
    def __init__(self, split):
        with open(SPLIT_JSON) as f:
            splits = json.load(f)
        self.patients = splits[split]
        self.pairs = []

        for pid in self.patients:
            clean_dir = os.path.join(CLEAN_ROOT, pid)
            noisy_dir = os.path.join(NOISY_ROOT, pid)
            if not (os.path.isdir(clean_dir) and os.path.isdir(noisy_dir)):
                continue

            for cpath in sorted(glob.glob(os.path.join(clean_dir, "slice_*.png"))):
                fname = os.path.basename(cpath)
                npath = os.path.join(noisy_dir, fname)
                if os.path.exists(npath):
                    self.pairs.append((npath, cpath))

        if not self.pairs:
            raise RuntimeError(
                f"No paired images found. "
                f"Check CLEAN_ROOT={CLEAN_ROOT}, NOISY_ROOT={NOISY_ROOT} and your splits."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy = to_tensor(Image.open(noisy_path).convert("L"))   # [1,H,W], 0..1
        clean = to_tensor(Image.open(clean_path).convert("L"))
        return noisy, clean

