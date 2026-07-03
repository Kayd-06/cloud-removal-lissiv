# Cloud Removal & Reconstruction for LISS-IV Imagery

Generative-AI framework for automated cloud removal in **LISS-IV** satellite imagery,
using **Sentinel-1 SAR** as a cloud-penetrating structural prior and cross-sensor
transfer learning from the **SEN12MS-CR** benchmark.

> **Novelty angle:** LISS-IV has *no SWIR bands* (only Green / Red / NIR at 5.8 m),
> so standard Sentinel-2 cloud-removal models don't transfer directly. We fuse
> cloud-penetrating SAR, reconstruct with both a GAN and a conditional diffusion
> model, and adapt across sensors. Three defensible contributions:
> (1) SWIR-less sensor adaptation, (2) SAR-optical multimodal fusion,
> (3) diffusion-vs-GAN comparison.

## Zero-cost stack

| Need | Free tool |
|------|-----------|
| Compute | **Kaggle** (GPU **T4 x2**, 30 GPU-h/week) → Colab free as overflow |
| LISS-IV | Bhoonidhi (ISRO) |
| Sentinel-1 SAR / Sentinel-2 | Google Earth Engine / Copernicus Browser |
| Cloud masks | `s2cloudless` / Sentinel-2 SCL |
| Pretraining data | SEN12MS-CR / SEN12MS-CR-TS |
| Frameworks | PyTorch, rasterio, GDAL, Gradio |

## Data convention

Every training sample is a co-registered patch stored as a single `.npz`:

| Key | Channels | Meaning |
|-----|----------|---------|
| `cloudy` | 3 | Cloud-corrupted optical (G, R, NIR), reflectance in [0,1] |
| `sar`    | 2 | Sentinel-1 (VV, VH), normalized dB |
| `mask`   | 1 | Cloud mask, 1 = cloud/shadow, 0 = clear |
| `clear`  | 3 | Cloud-free target optical (G, R, NIR) |

Generator input = `cloudy(3) + sar(2) + mask(1) = 6` channels → output `clear(3)`.

## Pipeline

```
0. scripts/gee_export.py       pull co-registered S1 SAR + S2 clear + cloud mask (free)
1. preprocess/tiling.py        raw GeoTIFFs        -> aligned 256x256 patches
2. preprocess/cloud_mask.py    transfer S2 cloud mask onto LISS-IV grid
3. preprocess/synthetic_clouds.py  clear patch + real mask -> paired (cloudy, clear)
4. train/train_gan.py          SAR-fusion pix2pix baseline (resumable)
5. train/train_diffusion.py    conditional DDPM, fine-tune from checkpoint
6. eval/compare.py             PSNR / SSIM / SAM table + ablation
7. demo/app.py                 Gradio before/after slider
```

## Data acquisition (the real bottleneck — start day 1)

Two-track strategy so you're never blocked waiting on LISS-IV orders:

- **Track A — build the training set now (free, fast):** `scripts/gee_export.py` pulls
  a Sentinel-2 cloud-free composite (matching G/R/NIR bands) + Sentinel-1 SAR + cloud
  mask over any footprint, co-registered by GEE. Train the whole model on this today.
- **Track B — LISS-IV fine-tune (slow):** order LISS-IV scenes on Bhoonidhi in parallel;
  when they arrive, run them through `tiling.py` and fine-tune. Cross-sensor transfer
  (S2 → LISS-IV) is itself a novelty point.

```bash
python scripts/gee_export.py --project <ee-proj> \
    --bbox 91.70 26.10 91.85 26.25 --start 2023-11-01 --end 2024-02-28
```

## Quick start (Kaggle)

```bash
pip install -r requirements.txt

# 1. Build synthetic paired data from clear patches (no natural pairs needed)
python -m src.preprocess.synthetic_clouds --clear_dir data/clear --out_dir data/patches

# 2. Train the SAR-fusion GAN baseline (resumes automatically from checkpoint)
python -m src.train.train_gan --data data/patches --ckpt_dir /kaggle/working/ckpts

# 3. Evaluate
python -m src.eval.compare --data data/patches --ckpt /kaggle/working/ckpts/gan_last.pt

# 4. Demo
python demo/app.py --ckpt /kaggle/working/ckpts/gan_last.pt
```

See `notebooks/kaggle_entrypoint.md` for the copy-paste Kaggle setup.

## Fallback

If Sentinel-1 co-registration proves too painful in the hackathon window, swap the
`sar` channels for a **temporal prior** (an earlier clear LISS-IV scene). The
`--cond_mode temporal` flag in the dataset supports this with no model changes.
