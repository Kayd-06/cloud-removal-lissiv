"""Train the SAR-fusion pix2pix baseline. Resumable, mixed-precision, mask-weighted L1.

Run:
  python -m src.train.train_gan --data data/patches --ckpt_dir /kaggle/working/ckpts

Resumes automatically from <ckpt_dir>/gan_last.pt.
"""
from __future__ import annotations
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import CFG
from src.datasets.cloud_dataset import CloudPatchDataset
from src.models.pix2pix import UNetGenerator, PatchGAN
from src.eval.metrics import evaluate
from src.train import checkpoint as ckpt


def build(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = UNetGenerator(CFG.in_channels, CFG.out_channels).to(dev)
    D = PatchGAN(CFG.in_channels, CFG.out_channels).to(dev)
    optG = torch.optim.Adam(G.parameters(), lr=CFG.lr, betas=CFG.betas)
    optD = torch.optim.Adam(D.parameters(), lr=CFG.lr, betas=CFG.betas)
    return dev, G, D, optG, optD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt_dir", default="ckpts")
    ap.add_argument("--epochs", type=int, default=CFG.epochs)
    ap.add_argument("--batch", type=int, default=CFG.batch_size)
    args = ap.parse_args()

    torch.manual_seed(CFG.seed)
    dev, G, D, optG, optD = build(args)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")

    tr = CloudPatchDataset(args.data, "train")
    va = CloudPatchDataset(args.data, "val", augment=False)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True,
                    num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=CFG.num_workers)
    print(f"[data] train={len(tr)} val={len(va)} device={dev} in_ch={CFG.in_channels}")

    last = os.path.join(args.ckpt_dir, "gan_last.pt")
    state = ckpt.load(last, map_location=dev)
    start = ckpt.resume_epoch(state)
    if state:
        G.load_state_dict(state["G"]); D.load_state_dict(state["D"])
        optG.load_state_dict(state["optG"]); optD.load_state_dict(state["optD"])
        print(f"[resume] from epoch {start}")

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss(reduction="none")

    for epoch in range(start, args.epochs):
        G.train(); D.train(); t0 = time.time()
        for step, batch in enumerate(tl):
            x = batch["input"].to(dev, non_blocking=True)
            y = batch["target"].to(dev, non_blocking=True)
            m = batch["mask"].to(dev, non_blocking=True)

            # --- D step ---
            with torch.cuda.amp.autocast(enabled=dev == "cuda"):
                fake = G(x)
                d_real = D(x, y)
                d_fake = D(x, fake.detach())
                loss_D = 0.5 * (bce(d_real, torch.ones_like(d_real)) +
                                bce(d_fake, torch.zeros_like(d_fake)))
            optD.zero_grad(set_to_none=True)
            scaler.scale(loss_D).backward(); scaler.step(optD)

            # --- G step ---
            with torch.cuda.amp.autocast(enabled=dev == "cuda"):
                d_fake = D(x, fake)
                adv = bce(d_fake, torch.ones_like(d_fake))
                # mask-weighted L1: clouded pixels count extra
                w = 1.0 + CFG.lambda_mask * m
                rec = (l1(fake, y) * w).mean()
                loss_G = adv + CFG.lambda_l1 * rec
            optG.zero_grad(set_to_none=True)
            scaler.scale(loss_G).backward(); scaler.step(optG); scaler.update()

            if step % CFG.log_every == 0:
                print(f"e{epoch} s{step}/{len(tl)} "
                      f"D={loss_D.item():.3f} G={loss_G.item():.3f} rec={rec.item():.3f}")

        # --- validation ---
        G.eval()
        agg = {}
        with torch.no_grad():
            for batch in vl:
                x = batch["input"].to(dev); y = batch["target"].to(dev); m = batch["mask"].to(dev)
                fake = G(x)
                mets = evaluate(fake, y, m)
                for k, v in mets.items():
                    agg[k] = agg.get(k, 0.0) + v
        agg = {k: v / max(len(vl), 1) for k, v in agg.items()}
        print(f"[val] epoch {epoch} " + " ".join(f"{k}={v:.3f}" for k, v in agg.items())
              + f"  ({time.time()-t0:.0f}s)")

        if epoch % CFG.ckpt_every == 0:
            ckpt.save(last, epoch=epoch, G=G.state_dict(), D=D.state_dict(),
                      optG=optG.state_dict(), optD=optD.state_dict(), val=agg)
            print(f"[ckpt] saved {last}")

    print("[done] training complete")


if __name__ == "__main__":
    main()
