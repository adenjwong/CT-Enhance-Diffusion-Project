# CT-Enhance-Diffusion-Project

A proof-of-concept pipeline for **CT denoising** using paired data: a *clean* CT slice and a synthetically corrupted *low-dose-like* noisy PNG.  
The project is designed to be **step-by-step, hackable, and reproducible**, with a clear path toward replacing synthetic noise with **real low-dose CT data**.

This repository currently implements:

- Data verification & indexing (LIDC-IDRI DICOM series discovery)
- DICOM → **16-bit PNG** conversion (Hounsfield Unit preserving)
- Synthetic low-dose-like noise generation (Poisson + Gaussian)
- Patient-level train/val/test splits
- Baseline **U-Net** training and evaluation
- Metrics: **PSNR** and **SSIM**
- Timestamped experiment runs with logs, checkpoints, and visual outputs

---

## Why this project exists

CT images—especially low-dose CT—are inherently noisy. That noise can obscure subtle anatomical structures and edges critical for diagnosis.

This project explores whether modern machine learning (starting with a simple **baseline U-Net**) can learn to:

- reduce noise
- preserve anatomy
- improve perceptual image quality

The current pipeline uses **standard-dose CT (LIDC-IDRI)** and applies *synthetic low-dose-like noise* to obtain **paired supervision**.  
This is an intentional **proof-of-concept stage** before moving to *true low-dose CT pairs* or sinogram-space modeling.

---

## Core ideas

### 1) HU-preserving clean images (16-bit PNG)

Instead of saving early 8-bit windowed images (which permanently clip information), the pipeline:

- converts DICOM pixels → Hounsfield Units (HU)
- clips to a clinically meaningful HU range
- linearly rescales to **16-bit PNG (0–65535)**

This ensures a stable *clean reference* that can be re-windowed later **without information loss**.

### 2) Synthetic low-dose-like noise (POC)

Noise is simulated in **image space** (not sinogram space):

- windowed intensity ≈ proxy for photon counts
- Poisson noise models photon shot noise
- small Gaussian noise approximates electronic/readout noise

This is **not a full CT physics model**, but sufficient to validate the learning pipeline end-to-end.

### 3) Patient-level splits

All slices from the same patient are kept in the same split.  
This prevents data leakage and ensures validation metrics reflect **generalization to unseen patients**.

### 4) Reproducible experiment runs

Each training run writes to a unique, timestamped directory:

- results/YYYY-MM-DD_HH-MM-SS/
  - model checkpoints
  - validation image grids
  - full training logs

---

## Project structure

    CT-Enhance-Diffusion-Project/
    ├── data/
    │   ├── raw_dicom/                      # (local only) downloaded DICOMs — DO NOT COMMIT
    │   ├── metadata.csv                    # TCIA metadata (safe to commit)
    │   ├── manifest.tcia                   # TCIA manifest (safe to commit)
    │   ├── index_summary.csv               # generated series index (ignore in git)
    │   ├── selected_series.csv             # selected CT series per patient (ignore in git)
    │   ├── clean_png/                      # generated 16-bit clean PNGs (ignore in git)
    │   ├── noisy_png/                      # generated noisy PNGs (ignore in git)
    │   └── splits/
    │       └── split_by_patient.json       # patient-level splits (safe to commit)
    │
    ├── results/
    │   └── YYYY-MM-DD_HH-MM-SS/            # one folder per training run (ignore in git)
    │       ├── train.log                   # full training log
    │       ├── val_panels_ep*.png          # noisy | denoised | clean grids
    │       └── unet_best.pth               # best checkpoint (by val PSNR)
    │
    ├── Data_Verification.py
    ├── Data_Conversion.py
    ├── Synthetic_Noise_Creation.py
    ├── Create_Split.py
    ├── Train_Baseline_UNet.py
    ├── Compute_PSNR_Pair.py
    ├── .gitignore
    └── README.md

---

## Script-by-script overview

**Data_Verification.py**  
Discovers valid CT DICOM series and builds CSV indexes.  
Outputs:
- data/index_summary.csv
- data/selected_series.csv

**Data_Conversion.py**  
Converts selected DICOM series into HU-preserving 16-bit PNG slices.  
Outputs:
- data/clean_png/<PatientID>/slice_XXXX.png

**Synthetic_Noise_Creation.py**  
Generates synthetic low-dose-like noisy images from clean PNGs.  
Outputs:
- data/noisy_png/<PatientID>/slice_XXXX.png

**Create_Split.py**  
Creates patient-level train/val/test splits to prevent leakage.  
Outputs:
- data/splits/split_by_patient.json

**Train_Baseline_UNet.py**  
Trains a baseline U-Net denoiser and evaluates PSNR + SSIM.  
Outputs (per run):
- train.log
- val_panels_ep*.png
- unet_best.pth

**Compute_PSNR_Pair.py**  
Lightweight utility to compute PSNR for a single noisy/clean image pair.

---

## Prerequisites

- Conda (or Miniconda)
- Python 3.10
- CPU-only PyTorch is sufficient (GPU optional)

---

## Setup

### 1) Create and activate environment

    conda create -n ct-enhance python=3.10 -y
    conda activate ct-enhance

### 2) Install dependencies

    pip install numpy pillow pydicom tqdm
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

---

## End-to-end run order

From the repository root:

    python Data_Verification.py
    python Data_Conversion.py
    python Synthetic_Noise_Creation.py
    python Create_Split.py
    python Train_Baseline_UNet.py

Each training run creates a new timestamped folder under results/.

---

*This project is intended for research and educational purposes only and is not a clinical system.*
