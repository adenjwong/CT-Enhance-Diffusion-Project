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

def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True),
    )

class TinyUNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.enc1 = conv_block(1, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bott = conv_block(base * 2, base * 4)

        self.up2  = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1  = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bott(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out(d1))

from torchvision.utils import save_image

def psnr(x, y):
    mse = torch.mean((x - y) ** 2)
    if mse <= 1e-12:
        return torch.tensor(99.0, device=x.device)
    return 10.0 * torch.log10(1.0 / mse)

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device("cpu")

    train_ds = PairDataset("train")
    val_ds   = PairDataset("val")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False)

    net = TinyUNet(base=32).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    best_psnr = 0.0
    epochs = 5

    for ep in range(1, epochs + 1):
        net.train()
        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            den = net(noisy)
            loss = loss_fn(den, clean)

            opt.zero_grad()
            loss.backward()
            opt.step()

        # validation
        net.eval()
        psnrs = []
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                den = net(noisy).clamp(0, 1)
                psnrs.append(psnr(den, clean).item())
        mpsnr = float(np.mean(psnrs)) if psnrs else 0.0
        print(f"Epoch {ep}: val PSNR = {mpsnr:.2f} dB")

        # save a small qualitative grid
        noisy, clean = next(iter(val_loader))
        noisy = noisy.to(device)
        clean = clean.to(device)
        den = net(noisy).clamp(0, 1)

        grid = torch.cat([noisy[:4], den[:4], clean[:4]], dim=0)
        save_image(grid, os.path.join(OUTDIR, f"val_panels_ep{ep}.png"), nrow=4)

        # checkpoint
        if mpsnr > best_psnr:
            best_psnr = mpsnr
            torch.save(net.state_dict(), os.path.join(OUTDIR, "unet_best.pth"))

    print("Training done. Best val PSNR:", best_psnr)

if __name__ == "__main__":
    main()
