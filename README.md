# CT-Enhance-Diffusion-Project

A proof-of-concept pipeline for **CT denoising** using paired data: a “clean” CT slice and a synthetically corrupted “low-dose-like” noisy PNG. The project is designed to be **step-by-step, hackable, and reproducible**, with a clear path to later swapping synthetic noise for **real low-dose CT**.

This repository currently implements:
- Data verification + indexing (LIDC-IDRI DICOM series discovery)
- DICOM → **16-bit PNG** conversion (HU-preserving scaling)
- Synthetic low-dose-like noise generation (Poisson + Gaussian)
- Patient-level train/val/test splits
- Baseline **U-Net** training and evaluation
- Metrics: **PSNR** and **SSIM**
- Timestamped runs with logs + outputs

---

## Why this project exists

CT images can be noisy, especially in low-dose settings. Noise can obscure subtle structures and edges. This project explores whether modern ML (starting with a baseline U-Net) can learn to:
- reduce noise
- preserve anatomy
- improve perceptual clarity

The current pipeline starts with **standard-dose CT data (LIDC-IDRI)** and creates a synthetic noisy version to obtain **paired supervision**. This is an intentional **proof-of-concept** step before moving toward **true low-dose CT pairs**.

---

## Key ideas

### 1) HU-preserving clean images (16-bit PNG)
Instead of saving windowed 8-bit images early (which clip/saturate detail), we:
- convert DICOM pixels to HU using slope/intercept
- clip HU to a typical diagnostic range
- scale to 16-bit PNG (0–65535)

This keeps a stable, “clean” reference that can be re-windowed later without losing information.

### 2) Synthetic low-dose-like noise (POC)
Noise is simulated in **image space** (not sinogram space):
- treat windowed intensity as a proxy for photon counts
- apply Poisson (shot) noise
- add small Gaussian readout noise

This is **not a full CT physics model**, but sufficient for validating a denoising learning pipeline.

### 3) Patient-level splits
All slices from a patient remain in the same split to prevent data leakage.

### 4) Reproducible experiment runs
Each training run writes to:
- `results/<timestamp>/` for checkpoints and visual outputs
- `results/<timestamp>/train.log` for logs

---

## Project breakdown

CT-Enhance-Diffusion-Project/

---

## Dataset and compliance

### Dataset used (current POC)
This project is structured around the **LIDC-IDRI** dataset (CT scans available via TCIA).

---

## Script-by-script overview

### `Data_Verification.py`
**Purpose:** verifies your dataset layout and builds CSV indexes from `data/raw_dicom/`.  
**Outputs:**
- `data/index_summary.csv` (all discovered series)
- `data/selected_series.csv` (one best CT series per patient)

### `Data_Conversion.py`
**Purpose:** converts the chosen CT DICOM series into **HU-preserving 16-bit PNG slices**.  
**Inputs:**
- `data/selected_series.csv`
- DICOMs under `data/raw_dicom/`
**Outputs:**
- `data/clean_png/<PatientID>/slice_XXXX.png` (16-bit PNG)

### `Synthetic_Noise_Creation.py`
**Purpose:** generates **synthetic low-dose-like** noisy images from the 16-bit clean PNGs.  
**Inputs:**
- `data/clean_png/...` (16-bit PNGs)
**Outputs:**
- `data/noisy_png/<PatientID>/slice_XXXX.png` (8-bit PNG)

### `Create_Split.py`
**Purpose:** creates **patient-level** splits (train/val/test) to prevent leakage.  
**Inputs:**
- typically uses the patient IDs found in `data/clean_png/` (or from selected series)
**Outputs:**
- `data/splits/split_by_patient.json`

### `Train_Baseline_UNet.py`
**Purpose:** trains a baseline U-Net denoiser using `(noisy, clean)` slice pairs.  
**Inputs:**
- `data/noisy_png/...` (8-bit noisy)
- `data/clean_png/...` (16-bit clean)
- `data/splits/split_by_patient.json`
**Outputs (per run):**
- `results/<timestamp>/train.log`
- `results/<timestamp>/val_panels_ep*.png`
- `results/<timestamp>/unet_best.pth`

### `Compute_PSNR_Pair.py`
**Purpose:** quick sanity-check metric script to compute PSNR on a pair.  
**Typical use:** validate that noisy→clean PSNR is reasonable before training, or spot-check a few slices.

---

## Setup

### 1) Create and activate environment
Conda example:

```bash
conda create -n ct-enhance python=3.10 -y
conda activate ct-enhance
pip install numpy pillow pydicom tqdm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
