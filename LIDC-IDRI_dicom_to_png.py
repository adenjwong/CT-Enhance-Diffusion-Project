#!/usr/bin/env python3
"""
Convert selected LIDC-IDRI DICOM series into PNG slices.
Outputs per-patient folders in data/clean_png/.
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
