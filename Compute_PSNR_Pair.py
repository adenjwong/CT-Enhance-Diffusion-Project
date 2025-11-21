#!/usr/bin/env python3
"""
Compute PSNR between a noisy PNG and its clean PNG counterpart.

Now automatic:

    - Looks in:
        data/noisy_png/<patient>/slice_XXXX.png
        data/clean_png/<patient>/slice_XXXX.png
    - Picks the first patient and first slice it finds.
"""

import os
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor

CLEAN_ROOT = "data/clean_png"
NOISY_ROOT = "data/noisy_png"

def psnr(x, y):
    """Compute PSNR between two tensors scaled in [0,1]."""
    mse = torch.mean((x - y) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(10 * torch.log10(1.0 / mse))

def main():
    patient = sorted(os.listdir(NOISY_ROOT))[0]
    noisy_patient_dir = os.path.join(NOISY_ROOT, patient)
    clean_patient_dir = os.path.join(CLEAN_ROOT, patient)

    slice_name = sorted(
        f for f in os.listdir(noisy_patient_dir)
        if f.startswith("slice_") and f.endswith(".png")
    )[0]

    noisy_path = os.path.join(noisy_patient_dir, slice_name)
    clean_path = os.path.join(clean_patient_dir, slice_name)

    # Load both images as grayscale tensors in [0,1]
    noisy = to_tensor(Image.open(noisy_path).convert("L"))
    clean = to_tensor(Image.open(clean_path).convert("L"))

    value = psnr(noisy, clean)
    print(f"Patient: {patient}")
    print(f"Slice:   {slice_name}")
    print(f"PSNR(noisy → clean): {value:.3f} dB")

if __name__ == "__main__":
    main()
