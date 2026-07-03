"""Co-register + tile raw GeoTIFFs into aligned patches.

Given a cloud-free LISS-IV scene (and optionally a Sentinel-1 SAR scene covering the
same footprint), this reprojects/resamples SAR onto the LISS-IV grid and cuts both
into non-overlapping patches saved as .npz with keys `clear` (and `sar`).

These `clear` patches are then fed to synthetic_clouds.py to build training pairs.

Requires rasterio. On Kaggle rasterio is preinstalled; locally use conda-forge.
"""
from __future__ import annotations
import argparse
import os

import numpy as np

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.enums import Resampling as RS
except ImportError:  # allow importing this module without rasterio for tests
    rasterio = None


def read_scaled(path: str, bands: list[int]) -> tuple[np.ndarray, "rasterio.Affine", str]:
    """Read selected bands, scale to reflectance [0,1] via per-band 2-98 percentile."""
    with rasterio.open(path) as src:
        arr = src.read(bands).astype(np.float32)  # (C,H,W)
        transform, crs = src.transform, src.crs
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        lo, hi = np.percentile(arr[i], (2, 98))
        out[i] = np.clip((arr[i] - lo) / (hi - lo + 1e-6), 0, 1)
    return out, transform, crs


def warp_to_ref(sar_path: str, ref_path: str) -> np.ndarray:
    """Reproject a Sentinel-1 scene onto the LISS-IV grid; returns (2,H,W) in dB-ish."""
    with rasterio.open(ref_path) as ref:
        dst_shape = (ref.count, ref.height, ref.width)
        ref_transform, ref_crs = ref.transform, ref.crs
        H, W = ref.height, ref.width
    with rasterio.open(sar_path) as s:
        n = min(2, s.count)
        dst = np.zeros((2, H, W), np.float32)
        for b in range(n):
            reproject(
                source=rasterio.band(s, b + 1),
                destination=dst[b],
                src_transform=s.transform, src_crs=s.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
    # normalize each SAR pol to [0,1] via percentile (keeps it comparable to optical)
    for b in range(dst.shape[0]):
        lo, hi = np.percentile(dst[b], (2, 98))
        dst[b] = np.clip((dst[b] - lo) / (hi - lo + 1e-6), 0, 1)
    return dst


def tile(clear: np.ndarray, sar: np.ndarray | None, size: int, stride: int,
         out_dir: str, prefix: str, min_valid: float = 0.6) -> int:
    C, H, W = clear.shape
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for y in range(0, H - size + 1, stride):
        for x in range(0, W - size + 1, stride):
            c = clear[:, y:y + size, x:x + size]
            # skip mostly-empty (nodata) tiles
            if (c > 1e-4).mean() < min_valid:
                continue
            kw = {"clear": c.astype(np.float32)}
            if sar is not None:
                kw["sar"] = sar[:, y:y + size, x:x + size].astype(np.float32)
            np.savez_compressed(os.path.join(out_dir, f"{prefix}_{y}_{x}.npz"), **kw)
            n += 1
    return n


def main() -> None:
    if rasterio is None:
        raise SystemExit("rasterio is required for tiling. `pip install rasterio` (conda-forge locally).")
    ap = argparse.ArgumentParser()
    ap.add_argument("--liss", required=True, help="cloud-free LISS-IV GeoTIFF")
    ap.add_argument("--bands", default="1,2,3", help="LISS-IV band indices for G,R,NIR")
    ap.add_argument("--sar", default=None, help="optional Sentinel-1 GeoTIFF (VV,VH)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--prefix", default="tile")
    args = ap.parse_args()

    bands = [int(b) for b in args.bands.split(",")]
    clear, _, _ = read_scaled(args.liss, bands)
    sar = warp_to_ref(args.sar, args.liss) if args.sar else None
    n = tile(clear, sar, args.size, args.stride, args.out_dir, args.prefix)
    print(f"[tiling] wrote {n} clear patches -> {args.out_dir}")


if __name__ == "__main__":
    main()
