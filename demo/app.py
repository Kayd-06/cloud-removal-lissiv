"""Gradio before/after demo. Upload a cloudy patch (.npz with cloudy/sar/mask) or a
PNG, and see the reconstruction. Great for the judge-facing demo.

Run:  python demo/app.py --ckpt ckpts/gan_last.pt
"""
from __future__ import annotations
import argparse

import numpy as np
import torch

from src.config import CFG
from src.models.pix2pix import UNetGenerator


def load_gan(ckpt_path, dev):
    G = UNetGenerator(CFG.in_channels, CFG.out_channels).to(dev).eval()
    G.load_state_dict(torch.load(ckpt_path, map_location=dev)["G"])
    return G


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    import gradio as gr
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = load_gan(args.ckpt, dev)

    def infer(npz_file):
        d = np.load(npz_file.name)
        cloudy = d["cloudy"].astype(np.float32)
        mask = d["mask"].astype(np.float32)
        sar = d["sar"].astype(np.float32) if "sar" in d else \
            np.zeros((CFG.sar_bands, *cloudy.shape[1:]), np.float32)
        cond = sar if CFG.cond_mode == "sar" else np.zeros((0, *cloudy.shape[1:]), np.float32)
        inp = np.concatenate([cloudy, cond, mask], 0)[None]
        with torch.no_grad():
            pred = G(torch.from_numpy(inp).to(dev))[0].cpu().numpy()
        to_img = lambda a: np.transpose(a[:3], (1, 2, 0)).clip(0, 1)
        return to_img(cloudy), to_img(pred)

    demo = gr.Interface(
        fn=infer,
        inputs=gr.File(label="Cloudy patch (.npz)"),
        outputs=[gr.Image(label="Cloudy input"), gr.Image(label="Reconstructed (cloud-free)")],
        title="LISS-IV Cloud Removal — SAR-fusion GAN",
        description="Upload a co-registered .npz patch (keys: cloudy, sar, mask).",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
