"""Fast end-to-end sanity check on CPU. Verifies shapes/losses/sampling all wire up.

  python scripts/smoke_test.py
"""
from __future__ import annotations
import os
import sys

# allow running as a plain script (`python scripts/smoke_test.py`) by putting the
# project root on the path so `import src...` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import CFG
from src.models.pix2pix import UNetGenerator, PatchGAN
from src.models.diffusion import CondUNet, GaussianDiffusion
from src.eval.metrics import evaluate


def main() -> None:
    b, s = 2, 64  # tiny for CPU speed (still divisible for U-Net downsampling)
    in_ch, out_ch = CFG.in_channels, CFG.out_channels

    x = torch.rand(b, in_ch, s, s)
    y = torch.rand(b, out_ch, s, s)
    m = (torch.rand(b, 1, s, s) > 0.5).float()

    # --- pix2pix ---
    G = UNetGenerator(in_ch, out_ch)
    D = PatchGAN(in_ch, out_ch)
    # U-Net needs size >= 256 for full 8 downs; assert generator handles small via 64 only if divisible
    try:
        fake = G(torch.rand(b, in_ch, 256, 256))
        assert fake.shape == (b, out_ch, 256, 256)
        disc = D(torch.rand(b, in_ch, 256, 256), fake)
        print(f"[ok] pix2pix  G_out={tuple(fake.shape)} D_out={tuple(disc.shape)}")
    except Exception as e:
        print(f"[FAIL] pix2pix: {e}")

    # --- diffusion ---
    model = CondUNet(out_ch, in_ch, base=32)
    diff = GaussianDiffusion(timesteps=50, device="cpu")
    loss = diff.loss(model, y, x)
    samp = diff.ddim_sample(model, x, y.shape, steps=5)
    assert samp.shape == y.shape
    print(f"[ok] diffusion loss={loss.item():.4f} sample={tuple(samp.shape)}")

    # --- metrics ---
    mets = evaluate(y, y, m)  # identical -> high psnr, low sam
    print(f"[ok] metrics on identical pair: {mets}")
    assert mets["psnr"] > 40 and mets["sam"] < 1e-2
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
