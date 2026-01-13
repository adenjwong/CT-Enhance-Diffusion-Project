#!/usr/bin/env python3

import os, glob, json
from datetime import datetime
import logging
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from tqdm import tqdm

CLEAN_ROOT = "data/clean_png"
NOISY_ROOT = "data/noisy_png"
SPLIT_JSON = "data/splits/split_by_patient.json"

RESULTS_ROOT = "results"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTDIR = os.path.join(RESULTS_ROOT, timestamp)

HU_MIN, HU_MAX = -1024, 3071
WINDOW_CENTER = 40
WINDOW_WIDTH  = 400

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

        arr16 = np.array(Image.open(clean_path), dtype=np.uint16)
        x01   = arr16.astype(np.float32) / 65535.0
        hu    = x01 * (HU_MAX - HU_MIN) + HU_MIN

        low  = WINDOW_CENTER - WINDOW_WIDTH / 2.0
        high = WINDOW_CENTER + WINDOW_WIDTH / 2.0
        clean01 = np.clip((hu - low) / (high - low), 0.0, 1.0)

        clean = torch.from_numpy(clean01).unsqueeze(0)

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

        return self.out(d1)

def psnr(x, y):
    mse = torch.mean((x - y) ** 2)
    if mse <= 1e-12:
        return torch.tensor(99.0, device=x.device)
    return 10.0 * torch.log10(1.0 / mse)

def _gaussian_window(window_size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.view(1, 1, 1, -1)
    window_2d = window_1d.transpose(2, 3) @ window_1d
    return window_2d

def ssim(x, y, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
    device = x.device
    window = _gaussian_window(window_size, sigma, device=device)

    mu_x = F.conv2d(x, window, padding=window_size//2, groups=1)
    mu_y = F.conv2d(y, window, padding=window_size//2, groups=1)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=window_size//2, groups=1) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size//2, groups=1) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size//2, groups=1) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + 1e-12)
    return ssim_map.mean()

def main():
    os.makedirs(OUTDIR, exist_ok=False)

    log_path = os.path.join(OUTDIR, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
    )
    log = logging.getLogger()

    device = torch.device("cpu")

    train_ds = PairDataset("train")
    val_ds   = PairDataset("val")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False)

    net = TinyUNet(base=64).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    best_psnr = 0.0
    epochs = 5

    for ep in range(1, epochs + 1):
        net.train()
        running_loss = 0.0
        seen = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [train]")
        for noisy, clean in train_bar:
            noisy = noisy.to(device)
            clean = clean.to(device)

            den = net(noisy)
            loss = loss_fn(den, clean)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            seen += 1
            avg_loss = running_loss / seen
            train_bar.set_postfix(loss=f"{avg_loss:.4f}")
        log.info(f"Example weight mean: {net.out.weight.data.mean().item()}")

        # validation
        net.eval()
        psnrs = []
        ssims = []
        val_bar = tqdm(val_loader, desc=f"Epoch {ep}/{epochs} [val]")
        with torch.no_grad():
            for noisy, clean in val_bar:
                noisy = noisy.to(device)
                clean = clean.to(device)

                den = net(noisy)
                den01 = den.clamp(0, 1)

                psnr_in  = psnr(noisy, clean).item()
                psnr_out = psnr(den01, clean).item()

                psnrs.append(psnr_out)
                
                ssim_in  = ssim(noisy, clean).item()
                ssim_out = ssim(den01, clean).item()
                ssims.append(ssim_out)
                
                val_bar.set_postfix(
                    in_psnr=f"{psnr_in:.2f}",
                    out_psnr=f"{psnr_out:.2f}",
                    in_ssim=f"{ssim_in:.3f}",
                    out_ssim=f"{ssim_out:.3f}",
                )

        mpsnr = float(np.mean(psnrs)) if psnrs else 0.0
        log.info(f"Epoch {ep}: val PSNR = {mpsnr:.2f} dB")
        mssim = float(np.mean(ssims)) if ssims else 0.0
        log.info(f"Epoch {ep}: val PSNR = {mpsnr:.2f} dB | val SSIM = {mssim:.3f}")

        # save a small qualitative grid
        noisy, clean = next(iter(val_loader))
        noisy = noisy.to(device)
        clean = clean.to(device)

        den = net(noisy)
        den01 = den.clamp(0, 1)

        grid = torch.cat([noisy[:4], den01[:4], clean[:4]], dim=0)
        save_image(grid, os.path.join(OUTDIR, f"val_panels_ep{ep}.png"), nrow=4)

        # checkpoint
        if mpsnr > best_psnr:
            best_psnr = mpsnr
            torch.save(net.state_dict(), os.path.join(OUTDIR, "unet_best.pth"))

    log.info(f"Training done. Best val PSNR: {best_psnr}")
    log.info(f"Saved results to: {OUTDIR}")

if __name__ == "__main__":
    main()
