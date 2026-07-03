"""Reconstruction metrics. All operate on tensors/arrays in [0,1], shape (C,H,W) or (B,C,H,W).

- PSNR : pixel fidelity
- SSIM : structural similarity
- SAM  : Spectral Angle Mapper (radians) -- proves *spectral consistency*, the metric
         most teams forget and exactly what the problem statement asks for.

For cloud removal we also report metrics *restricted to clouded pixels* (via the mask),
since that is the region the model actually had to reconstruct.
"""
from __future__ import annotations
import numpy as np
import torch


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, np.float32)


def psnr(pred, target, max_val: float = 1.0) -> float:
    p, t = _to_np(pred), _to_np(target)
    mse = np.mean((p - t) ** 2)
    if mse < 1e-12:
        return 99.0
    return float(20 * np.log10(max_val) - 10 * np.log10(mse))


def ssim(pred, target, C1=0.01 ** 2, C2=0.03 ** 2) -> float:
    """Global (single-window) SSIM averaged over channels. Lightweight, no deps."""
    p, t = _to_np(pred), _to_np(target)
    if p.ndim == 4:  # batch -> mean
        return float(np.mean([ssim(p[i], t[i]) for i in range(p.shape[0])]))
    vals = []
    for c in range(p.shape[0]):
        a, b = p[c], t[c]
        mu_a, mu_b = a.mean(), b.mean()
        va, vb = a.var(), b.var()
        cov = ((a - mu_a) * (b - mu_b)).mean()
        s = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / \
            ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
        vals.append(s)
    return float(np.mean(vals))


def sam(pred, target, eps: float = 1e-8) -> float:
    """Mean Spectral Angle Mapper in radians (lower = more spectrally faithful)."""
    p, t = _to_np(pred), _to_np(target)
    if p.ndim == 3:
        p, t = p[None], t[None]
    # (B,C,H,W) -> per-pixel spectral vectors
    B, C, H, W = p.shape
    pv = p.reshape(B, C, -1)
    tv = t.reshape(B, C, -1)
    dot = (pv * tv).sum(1)
    np_ = np.linalg.norm(pv, axis=1)
    nt = np.linalg.norm(tv, axis=1)
    cos = np.clip(dot / (np_ * nt + eps), -1, 1)
    return float(np.mean(np.arccos(cos)))


def _masked_spectra(pred, target, mask):
    """Select spectral vectors only at clouded pixels. Returns (pv, tv) each (C, Nmask).

    Handles (C,H,W) and (B,C,H,W). Selecting real pixels (rather than zero-filling
    the background) is what makes the masked metrics meaningful -- zero-filled
    pixels create zero spectral vectors that corrupt PSNR/SAM.
    """
    p, t, m = _to_np(pred), _to_np(target), _to_np(mask)
    if p.ndim == 3:
        p, t, m = p[None], t[None], m[None]
    B, C = p.shape[:2]
    pv = p.transpose(1, 0, 2, 3).reshape(C, -1)     # (C, B*H*W)
    tv = t.transpose(1, 0, 2, 3).reshape(C, -1)
    mm = m.reshape(B, m.shape[1], -1)[:, 0, :].reshape(-1) > 0.5   # (B*H*W,)
    return pv[:, mm], tv[:, mm]


def masked_psnr(pred, target, mask, max_val: float = 1.0) -> float:
    """PSNR computed over clouded pixels only."""
    pv, tv = _masked_spectra(pred, target, mask)
    if pv.shape[1] < 1:
        return psnr(pred, target, max_val)
    mse = np.mean((pv - tv) ** 2)
    if mse < 1e-12:
        return 99.0
    return float(20 * np.log10(max_val) - 10 * np.log10(mse))


def masked_sam(pred, target, mask, eps: float = 1e-8) -> float:
    """Mean SAM (radians) over clouded pixels only."""
    pv, tv = _masked_spectra(pred, target, mask)
    if pv.shape[1] < 1:
        return sam(pred, target)
    dot = (pv * tv).sum(0)
    cos = np.clip(dot / (np.linalg.norm(pv, axis=0) * np.linalg.norm(tv, axis=0) + eps), -1, 1)
    return float(np.mean(np.arccos(cos)))


def evaluate(pred, target, mask=None) -> dict:
    out = {"psnr": psnr(pred, target), "ssim": ssim(pred, target), "sam": sam(pred, target)}
    if mask is not None:
        out.update({
            "psnr_cloud": masked_psnr(pred, target, mask),
            "sam_cloud": masked_sam(pred, target, mask),
        })
    return out
