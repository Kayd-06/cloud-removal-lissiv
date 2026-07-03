"""Comparative assessment: run GAN and/or diffusion over the val set, print a metric
table, and dump before/after images. Directly satisfies the problem statement's
'comparative assessment of different Generative AI architectures'.

Run:
  python -m src.eval.compare --data data/patches --gan ckpts/gan_last.pt --diff ckpts/diff_last.pt
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import CFG
from src.datasets.cloud_dataset import CloudPatchDataset
from src.models.pix2pix import UNetGenerator
from src.models.diffusion import CondUNet, GaussianDiffusion
from src.eval.metrics import evaluate


def _mean(dicts: list[dict]) -> dict:
    keys = dicts[0].keys()
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}


def to_display(img):
    """Turn a (3,H,W)=[Green,Red,NIR] patch into a natural-looking RGB for viewing.

    LISS-IV has no Blue band, so we synthesize a plausible blue from green and render
    Red/Green/synth-Blue, then apply a per-image 2-98 percentile stretch + mild gamma.
    This makes vegetation green and land natural instead of the raw NIR-in-blue cast.
    """
    g, r, nir = img[0], img[1], img[2]
    b = np.clip(0.6 * g, 0, 1)                       # synthesized blue-ish channel
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    lo, hi = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1) ** 0.8
    return rgb


def save_triptych(cloudy, pred, target, path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    for a, img, t in zip(ax, [cloudy, pred, target], ["cloudy", "reconstructed", "clear"]):
        a.imshow(to_display(img[:3])); a.set_title(t); a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gan", default=None)
    ap.add_argument("--diff", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n_images", type=int, default=8)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    va = CloudPatchDataset(args.data, "val", augment=False)
    vl = DataLoader(va, batch_size=4, shuffle=False)

    models = {}
    if args.gan:
        G = UNetGenerator(CFG.in_channels, CFG.out_channels).to(dev).eval()
        G.load_state_dict(torch.load(args.gan, map_location=dev)["G"])
        models["GAN (SAR-fusion)"] = ("gan", G)
    if args.diff:
        M = CondUNet(CFG.out_channels, CFG.in_channels, CFG.base_ch).to(dev).eval()
        M.load_state_dict(torch.load(args.diff, map_location=dev)["model"])
        models["Diffusion (cond DDPM)"] = ("diff", (M, GaussianDiffusion(CFG.timesteps, dev)))

    if not models:
        raise SystemExit("Provide --gan and/or --diff checkpoints.")

    results = {name: [] for name in models}
    saved = 0
    with torch.no_grad():
        for batch in vl:
            x = batch["input"].to(dev); y = batch["target"].to(dev); m = batch["mask"].to(dev)
            for name, (kind, obj) in models.items():
                if kind == "gan":
                    pred = obj(x)
                else:
                    net, diff = obj
                    pred = diff.ddim_sample(net, x, y.shape, steps=CFG.sample_steps)
                for i in range(y.size(0)):
                    results[name].append(evaluate(pred[i], y[i], m[i]))
                    if saved < args.n_images and name == list(models)[0]:
                        save_triptych(batch["cloudy"][i].numpy(), pred[i].cpu().numpy(),
                                      y[i].cpu().numpy(),
                                      os.path.join(args.out, f"cmp_{saved}.png"))
                        saved += 1

    # print table
    print("\n=== Comparative assessment (val set) ===")
    header = ["model", "PSNR", "SSIM", "SAM", "PSNR_cloud", "SAM_cloud"]
    print("  ".join(f"{h:>16}" for h in header))
    for name, lst in results.items():
        mm = _mean(lst)
        row = [name, mm["psnr"], mm["ssim"], mm["sam"],
               mm.get("psnr_cloud", float("nan")), mm.get("sam_cloud", float("nan"))]
        print("  ".join(f"{row[0]:>16}" if i == 0 else f"{row[i]:>16.3f}"
                        for i in range(len(row))))
    print(f"\nSaved {saved} before/after images -> {args.out}/")


if __name__ == "__main__":
    main()
