"""Fine-tune the conditional DDPM (the novelty model). Resumable, mixed-precision.

Run:
  python -m src.train.train_diffusion --data data/patches --ckpt_dir /kaggle/working/ckpts \
      --init sen12mscr_pretrained.pt   # optional warm-start for cross-sensor transfer

Conditioning = [cloudy, sar, mask] = (in_channels - 0) ... target = clear(3).
Because the model predicts the target directly, cond_ch = CFG.in_channels - 0 here we
split: target is the 3 optical channels, conditioning is everything the dataset packed
into `input` EXCEPT it already includes cloudy; we pass the full input as conditioning.
"""
from __future__ import annotations
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from src.config import CFG
from src.datasets.cloud_dataset import CloudPatchDataset
from src.models.diffusion import CondUNet, GaussianDiffusion
from src.eval.metrics import evaluate
from src.train import checkpoint as ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt_dir", default="ckpts")
    ap.add_argument("--init", default=None, help="optional pretrained weights (transfer learning)")
    ap.add_argument("--epochs", type=int, default=CFG.epochs)
    ap.add_argument("--batch", type=int, default=CFG.batch_size)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(CFG.seed)

    # conditioning channels = full generator input (cloudy + cond + mask)
    cond_ch = CFG.in_channels
    model = CondUNet(target_ch=CFG.out_channels, cond_ch=cond_ch, base=CFG.base_ch).to(dev)
    diff = GaussianDiffusion(CFG.timesteps, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")

    tr = CloudPatchDataset(args.data, "train")
    va = CloudPatchDataset(args.data, "val", augment=False)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True,
                    num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=CFG.num_workers)
    print(f"[data] train={len(tr)} val={len(va)} device={dev} cond_ch={cond_ch}")

    # optional cross-sensor warm start (SEN12MS-CR -> LISS-IV)
    if args.init and os.path.exists(args.init):
        w = torch.load(args.init, map_location=dev)
        sd = w.get("model", w)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[transfer] loaded {args.init} (missing={len(missing)} unexpected={len(unexpected)})")

    last = os.path.join(args.ckpt_dir, "diff_last.pt")
    state = ckpt.load(last, map_location=dev)
    start = ckpt.resume_epoch(state)
    if state:
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
        print(f"[resume] from epoch {start}")

    for epoch in range(start, args.epochs):
        model.train(); t0 = time.time()
        for step, batch in enumerate(tl):
            cond = batch["input"].to(dev, non_blocking=True)   # (B, cond_ch, H, W)
            x0 = batch["target"].to(dev, non_blocking=True)    # (B, 3, H, W)
            with torch.cuda.amp.autocast(enabled=dev == "cuda"):
                loss = diff.loss(model, x0, cond)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if step % CFG.log_every == 0:
                print(f"e{epoch} s{step}/{len(tl)} loss={loss.item():.4f}")

        # --- validation via DDIM sampling on a few batches ---
        model.eval(); agg = {}; nb = 0
        with torch.no_grad():
            for batch in vl:
                cond = batch["input"].to(dev); y = batch["target"].to(dev); m = batch["mask"].to(dev)
                pred = diff.ddim_sample(model, cond, y.shape, steps=CFG.sample_steps)
                for k, v in evaluate(pred, y, m).items():
                    agg[k] = agg.get(k, 0.0) + v
                nb += 1
                if nb >= 4:  # keep val fast on free GPU
                    break
        agg = {k: v / max(nb, 1) for k, v in agg.items()}
        print(f"[val] epoch {epoch} " + " ".join(f"{k}={v:.3f}" for k, v in agg.items())
              + f"  ({time.time()-t0:.0f}s)")

        if epoch % CFG.ckpt_every == 0:
            ckpt.save(last, epoch=epoch, model=model.state_dict(), opt=opt.state_dict(), val=agg)
            print(f"[ckpt] saved {last}")

    print("[done] diffusion training complete")


if __name__ == "__main__":
    main()
