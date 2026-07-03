"""Lazy, tiled dataset. Reads one .npz patch per item -> stays inside Colab/Kaggle RAM.

Each .npz has keys: cloudy(3,H,W) sar(2,H,W) mask(1,H,W) clear(3,H,W), all float32 [0,1].

The generator input tensor is assembled here according to CFG.cond_mode:
  sar      -> [cloudy, sar, mask]        (6 ch)
  temporal -> [cloudy, prior_clear, mask](7 ch)  # prior scene passed in via `sar` slot
  none     -> [cloudy, mask]             (4 ch)
"""
from __future__ import annotations
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import CFG


class CloudPatchDataset(Dataset):
    def __init__(self, root: str, split: str = "train", val_frac: float = 0.1,
                 augment: bool = True, cond_mode: str | None = None):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz patches in {root}")
        # deterministic split
        n_val = max(1, int(len(self.files) * val_frac))
        if split == "train":
            self.files = self.files[n_val:]
        elif split == "val":
            self.files = self.files[:n_val]
        self.augment = augment and split == "train"
        self.cond_mode = cond_mode or CFG.cond_mode

    def __len__(self) -> int:
        return len(self.files)

    def _aug(self, *arrs: np.ndarray) -> list[np.ndarray]:
        # shared random flips/rotations across all channels
        if np.random.rand() < 0.5:
            arrs = [a[:, :, ::-1] for a in arrs]
        if np.random.rand() < 0.5:
            arrs = [a[:, ::-1, :] for a in arrs]
        k = np.random.randint(4)
        if k:
            arrs = [np.rot90(a, k, axes=(1, 2)) for a in arrs]
        return [np.ascontiguousarray(a) for a in arrs]

    def __getitem__(self, i: int) -> dict:
        d = np.load(self.files[i])
        cloudy = d["cloudy"].astype(np.float32)
        clear = d["clear"].astype(np.float32)
        mask = d["mask"].astype(np.float32)
        sar = d["sar"].astype(np.float32) if "sar" in d else \
            np.zeros((CFG.sar_bands, *cloudy.shape[1:]), np.float32)

        if self.augment:
            cloudy, clear, mask, sar = self._aug(cloudy, clear, mask, sar)

        if self.cond_mode == "sar":
            cond = sar
        elif self.cond_mode == "temporal":
            cond = sar  # caller stores the prior clear scene in the sar slot
        else:
            cond = np.zeros((0, *cloudy.shape[1:]), np.float32)

        inp = np.concatenate([cloudy, cond, mask], axis=0)  # (in_ch,H,W)
        return {
            "input": torch.from_numpy(inp),
            "target": torch.from_numpy(clear),
            "cloudy": torch.from_numpy(cloudy),
            "mask": torch.from_numpy(mask),
        }
