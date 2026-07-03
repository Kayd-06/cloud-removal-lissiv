"""Generate fake 'clear' patches so you can run the WHOLE pipeline end-to-end today,
before any real LISS-IV data arrives. Produces structured noise (not real imagery) --
purely to validate that training/eval/demo run without errors.

  python scripts/make_dummy_data.py --out data/clear --n 40
  python -m src.preprocess.synthetic_clouds --clear_dir data/clear --out_dir data/patches
"""
from __future__ import annotations
import argparse
import os

import numpy as np


def fake_scene(size, rng):
    """Smooth low-freq fields per band -> vaguely land-cover-like, plus 2 SAR pols."""
    def field():
        g = rng.random((size // 16, size // 16)).astype(np.float32)
        ys = np.linspace(0, g.shape[0] - 1, size).astype(int)
        xs = np.linspace(0, g.shape[1] - 1, size).astype(int)
        return np.clip(g[ys][:, xs] + 0.05 * rng.random((size, size)), 0, 1)
    clear = np.stack([field() for _ in range(3)]).astype(np.float32)
    sar = np.stack([field() for _ in range(2)]).astype(np.float32)
    return clear, sar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.n):
        clear, sar = fake_scene(args.size, rng)
        np.savez_compressed(os.path.join(args.out, f"dummy_{i:03d}.npz"), clear=clear, sar=sar)
    print(f"[dummy] wrote {args.n} fake clear patches -> {args.out}")


if __name__ == "__main__":
    main()
