#!/usr/bin/env python3
"""

Method summary
------------------------------------------------------------
Goal: create synthetic paired (noisy, clean) CT images

Approach:
1) Assume the windowed PNG (from DICOM -> PNG) is a proxy for relative photon
   intensity in [0,1] (brighter ≈ more photons)
2) Add Poisson (shot) noise: photon counting is Poisson; lowering dose
   reduces expected counts and increases relative variance
3) Add small Gaussian readout noise for detector/electronics effects
4) Clip back to [0,1] and save as 8-bit PNG


Math and Physics
------------------------------------------------------------
CT imaging is a photon-counting process: each detector element measures
how many X-ray photons arrive after passing through the patient.
Photon arrivals are random and follow a Poisson distribution:

    N ~ Poisson(λ)

where λ is the expected photon count for that pixel, which depends on
the X-ray tube current (mA), exposure time, and attenuation by tissue.

Fundamental Poisson properties:
    E[N] = λ
    Var[N] = λ
So the signal-to-noise ratio (SNR) for a pixel is:

    SNR = E[N] / sqrt(Var[N]) = sqrt(λ)

This means if you reduce the dose (therefore λ) by a factor of k,
the SNR only improves by √k when increasing dose or equivalently,
it drops by √k when reducing dose.

In our code:
  lam = photons * dose_scale * N0
  noisy = Poisson(lam) / (dose_scale * N0)

Here, 'photons' is the normalized intensity in [0,1] (proxy for expected
photon flux), 'dose_scale' reduces the expected counts (λ), and N0 sets
the baseline photon scale. The Poisson draw introduces variance equal to
the mean, reproducing the fundamental “fuzziness” of low-dose CT.

Afterward, a small Gaussian term is added to simulate readout/electronic
noise, which is approximately signal-independent:

  noisy += Normal(0, σ_read)

This results in a synthetic low-dose CT slice where:
- Bright areas (more photons) show smaller relative noise
- Darker areas (fewer photons) show larger fluctuations
- The overall noise strength scales with dose_scale ∝ 1/SNR²


Key parameters and tuning
------------------------------------------------------------
DOSE_SCALE (default 0.30)
• Scales the expected photon counts globally. Lower values simulate fewer detected photons = higher relative shot noise.
• Practical ranges:
  - 0.10–0.20: aggressive low-dose scans (challenging denoising)
  - 0.25–0.40: moderate low-dose (typical of many low-dose computed tomography studies
  - 0.50–1.00: mild noise (closer to NDCT appearance, standard level of radiation exposure)
• Interaction: SNR ∝ sqrt(DOSE_SCALE · N0). Halving DOSE_SCALE drops SNR by ~√2
• Tuning: select a value that yields a visibly low-dose look but still retains anatomy (start at 0.30 and adjust ±0.05)

READ_NOISE (default 0.002)
• Additive white Gaussian term approximating electronics/ADC noise after reconstruction.
• Unlike Poisson, it does not depend on brightness.
• Practical ranges:
  - 0.001–0.003: perceptible but secondary to shot noise.
  - >0.005: can start to dominate very dark regions and flatten contrast.
• Interaction: adds variance independent of intensity, keeps backgrounds from being unnaturally black after Poisson sampling.
• Tuning: keep small increase only if images look “too clean” in flat regions after applying Poisson.

N0 (implicit 1000 in code)
• Nominal photon count scale used to construct the Poisson rate λ = (img01_norm) × DOSE_SCALE × N0.
• Global SNR baseline. Larger N0 = higher absolute counts at the same DOSE_SCALE = milder noise.
• Practical ranges:
  - 500–1500 is typical for 8-bit.
  - If you increase N0 by k×, SNR rises by √k (at fixed DOSE_SCALE).
• Interaction with DOSE_SCALE: only the product (DOSE_SCALE × N0) really matters for SNR. Consider N0 as the “calibration" and DOSE_SCALE as the “dose”
• Tuning:
  - Set N0 so that DOSE_SCALE=0.30 produces a "realistic" noise level
  - Vary DOSE_SCALE to simulate different dose settings without touching N0

Additional notes on normalization & reproducibility
• Per-slice mean normalization: before Poisson sampling, the image is normalized by its mean to keep dose comparable across slices with different global brightness, this reduces slice-to-slice SNR drift unrelated to anatomy
• Random seeds: fixed seeds make the noisy realizations deterministic for exact reproducibility. Remove or vary seeds to generate multiple independent noisy pairs per slice for augmentation

Caveats:
- Image-space surrogate (not sinogram/log space), so it ignores view-dependent and reconstruction-kernel correlations, good for POC , not for exact physics fidelity
"""

import os, glob, random
import numpy as np
from PIL import Image
from tqdm import tqdm

CLEAN_ROOT = "data/clean_png"
NOISY_ROOT = "data/noisy_png"

# Reproducibility
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)

# Noise control
DOSE_SCALE = 0.30
READ_NOISE  = 0.002

def load01(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0

def save01(path, arr01):
    arr8 = (np.clip(arr01, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr8).save(path)

def add_ldct_like_noise(img01, dose_scale=DOSE_SCALE, read_noise=READ_NOISE):
    # Poisson (shot) noise on normalized "photon" field + small Gaussian readout noise
    photons = np.clip(img01, 1e-6, 1.0)
    photons = photons / (photons.mean() + 1e-8)
    lam = photons * dose_scale * 1000.0
    noisy = np.random.poisson(lam).astype(np.float32) / (dose_scale * 1000.0)
    noisy += np.random.normal(0, read_noise, size=noisy.shape).astype(np.float32)
    return np.clip(noisy, 0, 1)

def main():
    patients = [d for d in sorted(os.listdir(CLEAN_ROOT))
                if os.path.isdir(os.path.join(CLEAN_ROOT, d))]
    if not patients:
        print(f"No patient folders found under {CLEAN_ROOT}. Did you run the DICOM→PNG step?")
        return

    for pid in patients:
        clean_dir = os.path.join(CLEAN_ROOT, pid)
        out_dir   = os.path.join(NOISY_ROOT, pid)
        os.makedirs(out_dir, exist_ok=True)

        imgs = sorted(glob.glob(os.path.join(clean_dir, "slice_*.png")))
        if not imgs:
            print(f"Warning: no PNGs found for {pid} in {clean_dir}")
            continue

        for fp in tqdm(imgs, desc=pid):
            base = os.path.basename(fp)
            clean01 = load01(fp)
            noisy01 = add_ldct_like_noise(clean01)
            save01(os.path.join(out_dir, base), noisy01)

    print(f"Done. Created noisy PNGs in {NOISY_ROOT}")

if __name__ == "__main__":
    main()
