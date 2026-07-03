"""Synthetic cloud injection.

The single biggest data problem in cloud removal is the lack of *paired*
cloudy / cloud-free imagery of the same place at the same time. We sidestep it:
take CLEAR patches and paste physically-plausible clouds + shadows onto them,
giving unlimited supervised pairs (cloudy input, clear target).

Two cloud sources are supported:
  1. Real cloud masks harvested from Sentinel-2 SCL (best realism) -- pass --mask_dir.
  2. Procedural fractal (Perlin-like) clouds -- used when no mask bank exists.

Output: one .npz per patch with keys cloudy / sar / mask / clear (see README).
"""
from __future__ import annotations
import argparse
import glob
import os

import numpy as np


# --------------------------------------------------------------------------- #
# Procedural cloud field (diamond-square fractal noise -> soft cloud alpha)
# --------------------------------------------------------------------------- #
def _fractal_noise(size: int, rng: np.random.Generator, octaves: int = 5) -> np.ndarray:
    """Value-noise summed over octaves, normalized to [0,1]."""
    noise = np.zeros((size, size), np.float32)
    amp, freq, total_amp = 1.0, 1, 0.0
    for _ in range(octaves):
        g = max(2, size // (2 ** (octaves - freq)))
        coarse = rng.random((g, g)).astype(np.float32)
        # bilinear upsample to full size
        up = _resize_bilinear(coarse, size)
        noise += amp * up
        total_amp += amp
        amp *= 0.5
        freq += 1
    noise /= max(total_amp, 1e-6)
    return noise


def _resize_bilinear(a: np.ndarray, size: int) -> np.ndarray:
    ys = np.linspace(0, a.shape[0] - 1, size)
    xs = np.linspace(0, a.shape[1] - 1, size)
    y0 = np.floor(ys).astype(int); y1 = np.clip(y0 + 1, 0, a.shape[0] - 1)
    x0 = np.floor(xs).astype(int); x1 = np.clip(x0 + 1, 0, a.shape[1] - 1)
    wy = (ys - y0)[:, None]; wx = (xs - x0)[None, :]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def procedural_cloud(size: int, rng: np.random.Generator, coverage: float) -> np.ndarray:
    """Return a soft cloud alpha map in [0,1] with roughly `coverage` fraction clouded."""
    n = _fractal_noise(size, rng, octaves=5)
    # threshold so that ~coverage of pixels exceed it, then soften
    thr = np.quantile(n, 1.0 - coverage)
    alpha = np.clip((n - thr) / (1.0 - thr + 1e-6), 0, 1)
    alpha = alpha ** 0.8  # fatten cloud cores
    return alpha.astype(np.float32)


def add_shadow(clear: np.ndarray, alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Darken a shifted copy of the cloud footprint to fake cast shadows."""
    shift = rng.integers(4, 24)
    shadow = np.zeros_like(alpha)
    shadow[shift:, shift:] = alpha[:-shift, :-shift]
    shadow = (shadow > 0.3).astype(np.float32) * 0.35
    out = clear * (1.0 - shadow[None])   # clear is (C,H,W)
    return out.astype(np.float32)


def composite(clear: np.ndarray, alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Blend bright cloud radiance over the (shadowed) clear image."""
    shadowed = add_shadow(clear, alpha, rng)
    # clouds are bright & slightly desaturated; add mild per-cloud brightness jitter
    cloud_val = np.clip(0.85 + 0.15 * rng.random(), 0, 1)
    cloudy = shadowed * (1.0 - alpha[None]) + cloud_val * alpha[None]
    return np.clip(cloudy, 0, 1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_clear_patch(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a clear patch .npz (keys: clear, optional sar) -> (clear CxHxW, sar or None)."""
    d = np.load(path)
    clear = d["clear"].astype(np.float32)
    sar = d["sar"].astype(np.float32) if "sar" in d else None
    return clear, sar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear_dir", required=True, help="dir of clear patch .npz files")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mask_dir", default=None, help="optional bank of real cloud-mask .npy files")
    ap.add_argument("--per_patch", type=int, default=3, help="synthetic variants per clear patch")
    ap.add_argument("--min_cov", type=float, default=0.1)
    ap.add_argument("--max_cov", type=float, default=0.6)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    mask_bank = sorted(glob.glob(os.path.join(args.mask_dir, "*.npy"))) if args.mask_dir else []
    clears = sorted(glob.glob(os.path.join(args.clear_dir, "*.npz")))
    if not clears:
        raise SystemExit(f"No .npz clear patches found in {args.clear_dir}")

    n = 0
    for cp in clears:
        clear, sar = load_clear_patch(cp)
        if sar is None:
            sar = np.zeros((2, clear.shape[1], clear.shape[2]), np.float32)
        base = os.path.splitext(os.path.basename(cp))[0]
        for k in range(args.per_patch):
            cov = float(rng.uniform(args.min_cov, args.max_cov))
            if mask_bank:
                m = np.load(mask_bank[rng.integers(len(mask_bank))]).astype(np.float32)
                m = _resize_bilinear(m, args.size)
                alpha = np.clip(m, 0, 1)
            else:
                alpha = procedural_cloud(args.size, rng, cov)
            cloudy = composite(clear, alpha, rng)
            mask = (alpha > 0.2).astype(np.float32)[None]  # binary cloud mask (1,H,W)
            out = os.path.join(args.out_dir, f"{base}_syn{k}.npz")
            np.savez_compressed(out, cloudy=cloudy, sar=sar, mask=mask, clear=clear)
            n += 1
    print(f"[synthetic_clouds] wrote {n} paired patches -> {args.out_dir}")


if __name__ == "__main__":
    main()
