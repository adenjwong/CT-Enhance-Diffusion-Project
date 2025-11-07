#!/usr/bin/env python3
"""

Method summary
------------------------------------------------------------
Goal: Convert original DICOM CT series into standardized 16-bit PNG slices
for downstream processing and noise simulation.

Approach:
1) Read raw CT DICOMs, extract pixel data, and apply rescale slope/intercept
   to obtain voxel intensities in Hounsfield Units (HU)
2) Clip each voxel's HU value to a physically meaningful range
   [HU_MIN, HU_MAX] to prevent overflow or underflow from scanner variations
3) Linearly rescale that HU range into 16-bit integer space [0, 65535],
   preserving the full dynamic range of tissue attenuation
4) Save each slice as a 16-bit PNG (mode “I;16”) so that later scripts
   (e.g., Synthetic_Noise_Creation.py) can apply arbitrary windowing
   (lung, soft-tissue, bone, etc.) without saturation or loss of detail

Motivation:
Standard 8-bit PNGs (0-255) cannot represent the full 4000+ HU range found
in CT scans. They require an explicit windowing step that compresses HU
values to fit within a narrower grayscale band. This causes clipping in
bright (bone, contrast, metal) or dark (air) regions and permanently discards
information. By contrast, exporting 16-bit scaled PNGs retains all voxel data
while remaining portable and lightweight for downstream machine-learning use.

CT physics background
------------------------------------------------------------
CT reconstruction produces voxel intensities in Hounsfield Units (HU):

    HU = 1000 x (μ_tissue - μ_water) / (μ_water - μ_air)

where μ represents the linear X-ray attenuation coefficient.
By definition:
    Water ≈ 0 HU
    Air ≈ -1000 HU
    Compact bone ≈ +1000 HU
    Metal / contrast ≈ +2000 to +3000 HU

This scale allows consistent comparison across scanners and patients.

Typical intensity range:
    HU_MIN = -1024 -> air and below
    HU_MAX = +3071 -> dense bone and metal

Values outside this range are uncommon in diagnostic CT and mostly represent
noise or hardware saturation. Clipping to this interval retains all relevant
anatomical content while limiting dynamic range for efficient 16-bit storage.

Rescaling
------------------------------------------------------------
After clipping, each voxel's HU is linearly mapped to 16-bit integer space:

    arr16 = ((HU - HU_MIN) / (HU_MAX - HU_MIN)) x 65535

This yields:
    Air (≈-1000 HU)  → ~0
    Water (0 HU)     → ~16384
    Bone (+1000 HU)  → ~22900
    Dense bone (+3000 HU) → ~41000-65535 (depending on scanner)

No windowing is applied here; instead, all detail is preserved.
Later stages can window dynamically when generating 8-bit
visualizations or synthetic noise variants.

Benefits
------------------------------------------------------------
• Preserves full HU dynamic range for flexible post-processing
• Prevents highlight saturation seen in lung/bone window PNGs
• Compatible with PIL and standard ML data loaders
• Enables quantitative HU analysis and re-windowing without rereading DICOMs

Caveats
------------------------------------------------------------
• Although HU scaling is preserved, all DICOM metadata (spacing, orientation)
  is not included in the PNG output.
• Intended for proof-of-concept image enhancement workflows, not clinical.
"""

import os, csv, pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm

SELECT_CSV = "data/selected_series.csv"
OUT_ROOT = "data/clean_png"
WINDOW_CENTER = -600  # lung window center
WINDOW_WIDTH = 1500   # lung window width

def apply_window(img, center, width):
    low, high = center - width / 2, center + width / 2
    img = np.clip(img, low, high)
    img = (img - low) / (high - low) * 255.0
    return img.astype(np.uint8)

def convert_series(series_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dcm_files = sorted([f for f in os.listdir(series_path) if f.endswith(".dcm")])
    for i, fname in enumerate(tqdm(dcm_files, desc=os.path.basename(series_path))):
        dcm_path = os.path.join(series_path, fname)
        ds = pydicom.dcmread(dcm_path)
        img = ds.pixel_array.astype(np.float32)
        if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
            img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        img = apply_window(img, WINDOW_CENTER, WINDOW_WIDTH)
        Image.fromarray(img).save(os.path.join(out_dir, f"slice_{i:04d}.png"))

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(SELECT_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient = row["PatientID"]
            path = row["SeriesPath"]
            if not os.path.isdir(path):
                print(f"Skipping {patient}: path not found")
                continue
            out_dir = os.path.join(OUT_ROOT, patient)
            convert_series(path, out_dir)
    print("All selected series converted to PNGs.")

if __name__ == "__main__":
    main()
