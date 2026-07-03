"""Resumable checkpointing -- the thing that makes free Colab/Kaggle actually usable.

Save to a persistent dir (Google Drive mount or a Kaggle /kaggle/working that you
version as an output dataset). Training always resumes from `<name>_last.pt` if present,
so a killed 9-hour session costs you at most `ckpt_every` epochs.
"""
from __future__ import annotations
import os

import torch


def save(path: str, **state) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic -> never leaves a half-written checkpoint


def load(path: str, map_location="cpu") -> dict | None:
    if os.path.exists(path):
        return torch.load(path, map_location=map_location)
    return None


def resume_epoch(ckpt: dict | None) -> int:
    return 0 if ckpt is None else int(ckpt.get("epoch", 0)) + 1
