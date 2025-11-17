#!/usr/bin/env python3
"""
Compute PSNR between a noisy PNG and its clean PNG counterpart.

Usage:
    python Compute_PSNR_Pair.py /path/to/noisy.png /path/to/clean.png
"""

import sys
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor

def psnr(x, y):
    """Compute PSNR between two tensors scaled in [0,1]."""
    mse = torch.mean((x - y) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(10 * torch.log10(1.0 / mse))

def main():
    if len(sys.argv) != 3:
        print("Usage: python Compute_PSNR_Pair.py noisy.png clean.png")
        return

    noisy_path = sys.argv[1]
    clean_path = sys.argv[2]

    # Load both images as grayscale tensors in [0,1]
    noisy = to_tensor(Image.open(noisy_path).convert("L"))
    clean = to_tensor(Image.open(clean_path).convert("L"))

    value = psnr(noisy, clean)
    print(f"PSNR(noisy → clean): {value:.3f} dB")

if __name__ == "__main__":
    main()
