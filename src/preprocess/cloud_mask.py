"""Cloud-mask utilities for LISS-IV.

LISS-IV lacks SWIR, so classic cloud tests (e.g. cirrus band) are unavailable.
Two practical, zero-cost strategies:

  1. transfer_s2_mask(): reproject a Sentinel-2 SCL / s2cloudless mask (which DOES
     have the right bands) onto the LISS-IV grid. Best accuracy -- use when a
     near-coincident S2 acquisition exists.

  2. spectral_mask(): a fast fallback for natural cloudy LISS-IV scenes using
     brightness + NDVI heuristics. Rough, but needs no auxiliary data.

For TRAINING we mostly rely on synthetic masks (see synthetic_clouds.py); this
module is for building masks on REAL cloudy scenes at inference/eval time.
"""
from __future__ import annotations
import numpy as np

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
except ImportError:
    rasterio = None


def spectral_mask(optical: np.ndarray, bright_thr: float = 0.6,
                  ndvi_thr: float = 0.15) -> np.ndarray:
    """Heuristic cloud mask from (G,R,NIR) reflectance in [0,1].

    Clouds are bright across all bands and have low NDVI. Returns (1,H,W) in {0,1}.
    """
    g, r, nir = optical[0], optical[1], optical[2]
    brightness = (g + r + nir) / 3.0
    ndvi = (nir - r) / (nir + r + 1e-6)
    cloud = (brightness > bright_thr) & (ndvi < ndvi_thr)
    return cloud.astype(np.float32)[None]


def transfer_s2_mask(s2_mask_path: str, ref_liss_path: str) -> np.ndarray:
    """Reproject a Sentinel-2 cloud mask GeoTIFF onto the LISS-IV grid.

    s2_mask can be SCL (values 3,8,9,10 = cloud/shadow) or a binary s2cloudless map.
    Returns (1,H,W) binary mask.
    """
    if rasterio is None:
        raise RuntimeError("rasterio required for transfer_s2_mask")
    with rasterio.open(ref_liss_path) as ref:
        H, W = ref.height, ref.width
        ref_transform, ref_crs = ref.transform, ref.crs
    with rasterio.open(s2_mask_path) as s:
        dst = np.zeros((H, W), np.float32)
        reproject(
            source=rasterio.band(s, 1), destination=dst,
            src_transform=s.transform, src_crs=s.crs,
            dst_transform=ref_transform, dst_crs=ref_crs,
            resampling=Resampling.nearest,
        )
    # SCL cloud/shadow classes -> 1; if already binary this is a no-op-ish threshold
    scl_cloud = np.isin(dst.astype(int), [3, 8, 9, 10])
    binary = scl_cloud | (dst > 0.5)
    return binary.astype(np.float32)[None]


def dilate(mask: np.ndarray, iters: int = 2) -> np.ndarray:
    """Grow the mask slightly to cover soft cloud edges (uses OpenCV if available)."""
    try:
        import cv2
        k = np.ones((3, 3), np.uint8)
        m = cv2.dilate((mask[0] > 0.5).astype(np.uint8), k, iterations=iters)
        return m.astype(np.float32)[None]
    except ImportError:
        return mask
