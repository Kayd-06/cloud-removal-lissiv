# Presentation Guide — LISS-IV Cloud Removal

A slide-by-slide script + talking points + anticipated Q&A for the demo.
Aim: ~6–8 slides, 5–7 minutes.

---

## Slide 1 — Title
**Generative-AI Cloud Removal & Reconstruction for LISS-IV Imagery**
Your name / team · Problem Statement 2

> *Say:* "Clouds make high-resolution optical satellite imagery unusable over regions
> like North-East India. We built a generative-AI system that removes clouds and
> reconstructs the surface underneath."

## Slide 2 — The problem
- LISS-IV = 5.8 m resolution, but **frequent cloud cover** over NER (tropical, mountainous).
- Traditional cloud masking = **discards data**, leaves holes.
- Need: fill the clouded regions *plausibly and consistently*.

> *Say:* "The hard part isn't detecting clouds — it's reconstructing what's underneath
> without just making it up."

## Slide 3 — Our key idea (the novelty)
Three contributions:
1. **SWIR-less adaptation** — LISS-IV lacks the SWIR bands most cloud-removal models rely
   on; ours works on Green/Red/NIR only.
2. **SAR fusion** — Sentinel-1 radar **sees through clouds**, so we condition on *real*
   sub-cloud structure instead of hallucinating.
3. **Comparative GenAI framework** — GAN and diffusion on one shared interface.

> *Say:* "SAR is the trick. Radar penetrates clouds, so our reconstruction is guided by
> real ground information, not a guess."

## Slide 4 — Data & pipeline (show the diagram)
- Sentinel-2 cloud-free composite = target; Sentinel-1 SAR = prior; all free via Google
  Earth Engine, co-registered automatically.
- **Synthetic cloud injection** solves the paired-data problem: paste realistic clouds +
  shadows onto clear scenes → unlimited (cloudy, clear) training pairs.
- **18 regions** across all NE states, varied terrain → 2,295 training patches.

> *Say:* "We never needed hard-to-find matched cloudy/clear photos — we synthesize the
> clouds, which also lets us control difficulty."

## Slide 5 — Model
- Generator input: `cloudy(3) + SAR(2) + cloud-mask(1)`.
- SAR-fusion **pix2pix**: U-Net generator + PatchGAN, **mask-weighted L1** (clouded pixels
  weighted 10× so the model focuses where it matters).
- Fully reproducible & resumable on **free** Kaggle GPU.

## Slide 6 — Results (show before/after triptychs + table)
- Show 2–3 **cloudy → reconstructed → clear** images (use the FCC or vivid render).
- Metrics table: **PSNR 23.5 · SSIM 0.944 · SAM 0.083**; **+1.5 dB** inside clouded areas
  when scaling from 6→18 regions.

> *Say:* "SSIM 0.94 means structure — roads, field boundaries, rivers — is preserved. Low
> SAM means the colours/spectra stay faithful, so the output is analysis-ready."

## Slide 7 — Impact & future work
- **Impact:** turns cloud-blocked scenes into usable data for LULC, disaster monitoring,
  environmental assessment.
- **Future:** conditional diffusion (implemented, ready to train) for sharper detail;
  fine-tune on native LISS-IV from Bhoonidhi; larger multi-season corpus.

## Slide 8 — Reproducibility / close
- 100% free stack (Kaggle + Google Earth Engine), open on GitHub, end-to-end runnable.

> *Say:* "Everything you saw runs on free compute and is fully reproducible from our repo."

---

## Anticipated Q&A

**Q: Isn't the model just hallucinating the clouded pixels?**
A: The SAR prior gives real sub-cloud structure, and we weight the loss 10× on clouded
pixels. Metrics are also reported *cloud-region-only* (PSNR_cloud/SAM_cloud), so we
measure exactly the reconstructed area, not the easy background.

**Q: Why Sentinel-2 as the target instead of LISS-IV?**
A: LISS-IV clear scenes are scarce/slow to order; Sentinel-2 has matching G/R/NIR bands
and is abundant + free. We train cross-sensor now and fine-tune on LISS-IV as the last
step — a deliberate transfer-learning design.

**Q: Why synthetic clouds — is that realistic?**
A: It solves the paired-data problem (no natural cloudy/clear pairs exist for the same
instant) and is a standard technique. Masks can also be sourced from real Sentinel-2
cloud maps for extra realism.

**Q: GAN vs diffusion — which is better?**
A: We implemented both on a shared interface. The GAN is trained and evaluated here;
diffusion is our next step. At this data/compute scale a well-tuned GAN is competitive;
diffusion typically needs more data/compute to pull ahead — which is exactly our future
work.

**Q: How would this deploy operationally?**
A: The workflow is scalable and automated: export → tile → infer → evaluate. Resumable
checkpointing and a batchable pipeline make it deployable on modest hardware.
