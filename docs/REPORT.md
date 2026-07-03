# Generative-AI Cloud Removal & Reconstruction for LISS-IV Imagery

**Problem Statement 2 — Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery**

---

## 1. Problem & Objective

Persistent cloud cover over the North-Eastern Region (NER) of India severely limits the
usability of high-resolution **LISS-IV** optical imagery (Green/Red/NIR at 5.8 m) for
land-use mapping, disaster monitoring, and environmental assessment. Traditional cloud
masking discards contaminated pixels, leaving gaps in the scene.

We develop a **Generative-AI framework that automatically removes clouds and reconstructs
the underlying surface** in LISS-IV-style imagery, preserving both **spatial structure**
and **spectral consistency**, and producing analysis-ready cloud-free products.

## 2. Approach & Novelty

Standard cloud-removal models are trained on Sentinel-2, which carries SWIR/cirrus bands
that LISS-IV lacks. Our framework is designed around three contributions:

1. **SWIR-less sensor adaptation** — the pipeline operates on the LISS-IV band set
   (Green, Red, NIR only), matching Sentinel-2's B3/B4/B8 for cross-sensor transfer.
2. **SAR–optical multimodal fusion** — cloud-penetrating **Sentinel-1 SAR (VV, VH)** is
   fused as a structural prior, so reconstruction is guided by *real* sub-cloud
   information rather than pure hallucination.
3. **Architecture comparison** — a SAR-fusion conditional **GAN** baseline plus a
   conditional **diffusion (DDPM)** model share the same data interface, enabling a
   direct GAN-vs-diffusion assessment.

## 3. Data & Preprocessing

All data is free and reproducible.

| Source | Role | Access |
|---|---|---|
| Sentinel-2 SR (B3,B4,B8) cloud-free composite | Clear-optical **target** | Google Earth Engine |
| Sentinel-1 GRD (VV,VH) | Cloud-penetrating **prior** | Google Earth Engine |
| Sentinel-2 SCL / s2cloudless | Cloud masks | Google Earth Engine |
| LISS-IV (Bhoonidhi) | Target sensor for fine-tuning | ISRO Bhoonidhi |

**Pipeline:**
1. **Acquisition** (`scripts/gee_export.py`) — export co-registered S2/S1/mask GeoTIFFs
   over a bounding box, downloaded directly into the compute environment.
2. **Tiling** (`tiling.py`) — reproject SAR onto the optical grid, percentile-scale to
   reflectance, cut into aligned 256×256 patches.
3. **Synthetic cloud injection** (`synthetic_clouds.py`) — paste physically-plausible
   fractal clouds + cast shadows onto clear patches, generating unlimited **paired**
   (cloudy, clear) training data without scarce natural pairs.

**Dataset used:** 18 regions spanning all NE India states and varied terrain (urban,
hills, floodplain, tea gardens, river valleys), dry season (Nov 2023–Feb 2024) for
cloud-free targets → **459 clear patches → 2,295 training pairs**.

## 4. Method / Architecture

**Generator input** = `[cloudy optical (3) ⊕ SAR (2) ⊕ cloud mask (1)]` = 6 channels →
**output** = reconstructed clear optical (3).

- **Baseline:** SAR-fusion **pix2pix** — 8-block U-Net generator with skip connections +
  70×70 PatchGAN discriminator. Objective = adversarial + **mask-weighted L1** (clouded
  pixels weighted ×10, focusing capacity on the regions actually being reconstructed).
- **Novelty model:** conditional **DDPM** (`diffusion.py`) — a compact time-conditioned
  U-Net predicting noise on the clear image, conditioned on the same 6-channel stack;
  DDIM sampling (50 steps) at inference. Designed for warm-start transfer from the
  SEN12MS-CR benchmark.

Training is mixed-precision and **fully resumable** (atomic checkpointing), enabling
robust operation on free, session-limited compute (Kaggle T4).

## 5. Evaluation Metrics

- **PSNR** — pixel-level reconstruction fidelity.
- **SSIM** — structural similarity (spatial-structure preservation).
- **SAM (Spectral Angle Mapper, radians)** — spectral consistency; the key metric for
  "analysis-ready" spectral fidelity.
- **Cloud-region variants** (`PSNR_cloud`, `SAM_cloud`) — the same metrics computed over
  clouded pixels only, isolating true reconstruction quality from unaffected areas.

## 6. Quantitative Results

SAR-fusion GAN, validation set. The dataset-scaling study shows the effect of expanding
from 6 to 18 regions:

| Configuration | PSNR ↑ | SSIM ↑ | SAM ↓ | PSNR_cloud ↑ | SAM_cloud ↓ |
|---|---|---|---|---|---|
| GAN, 636 pairs (6 regions), 40 ep | 22.14 | 0.938 | 0.093 | 20.57 | 0.108 |
| **GAN, 2,295 pairs (18 regions), 60 ep** | **23.52** | **0.944** | **0.083** | **22.11** | **0.099** |

Expanding geographic/terrain diversity improved every metric, most notably **+1.5 dB
PSNR in the reconstructed (clouded) regions** — direct evidence of better surface
recovery, not just background copying.

## 7. Qualitative Results

Reconstructions are rendered in three standard styles (`--render vivid|natural|fcc`) to
show the result is robust across visualizations:
- **Vivid / Natural** — white-balanced true-ish color (vegetation green, land natural).
- **FCC (False Color Composite, NIR-R-G)** — the standard remote-sensing view;
  vegetation appears red, aiding visual assessment of reconstructed land cover.

Each figure is a triptych: **cloudy input → reconstructed → clear ground truth**. Clouds
and shadows are removed while road networks, field boundaries, water bodies, and
vegetation texture are preserved and spectrally consistent with the target.

## 8. Limitations & Future Work

- **GAN softness:** fine texture inside large clouded regions is slightly smoothed — the
  known limitation of adversarial inpainting at this data/compute scale.
- **Next step — diffusion:** train the conditional DDPM (already implemented) with
  SEN12MS-CR warm-start for sharper, higher-fidelity reconstruction, and report the
  GAN-vs-diffusion comparison.
- **LISS-IV fine-tuning:** the model is trained on Sentinel band-analogues; fine-tuning
  on native LISS-IV scenes (Bhoonidhi) closes the sensor gap for operational deployment.
- **Scale-up:** larger footprints and multi-season acquisition for a bigger, more varied
  training corpus.

## 9. Reproducibility

- **Repository:** github.com/Kayd-06/cloud-removal-lissiv
- **Compute:** Kaggle (free T4 GPU); no paid services.
- **Data:** Google Earth Engine (free, non-commercial tier); Bhoonidhi for LISS-IV.
- End-to-end runnable: acquisition → preprocessing → training → evaluation → demo, with a
  synthetic-data smoke test for instant verification.

---

*Deliverables: automated cloud-free reconstruction, spatially & spectrally consistent
analysis-ready products, a comparative GenAI framework, and a prototype workflow for
operational deployment.*
