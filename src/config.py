"""Central configuration. Import `CFG` everywhere so training/eval/demo stay in sync."""
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- data ---
    patch_size: int = 256          # drop to 128 on tight memory (Colab free)
    optical_bands: int = 3         # LISS-IV: Green, Red, NIR
    sar_bands: int = 2             # Sentinel-1: VV, VH
    cond_mode: str = "sar"         # "sar" | "temporal" | "none"

    @property
    def in_channels(self) -> int:
        """Generator input = cloudy optical + conditioning + cloud mask."""
        cond = {"sar": self.sar_bands, "temporal": self.optical_bands, "none": 0}[self.cond_mode]
        return self.optical_bands + cond + 1  # +1 for the cloud mask

    @property
    def out_channels(self) -> int:
        return self.optical_bands

    # --- training ---
    batch_size: int = 8            # 8 fits P100 @256; use 16 @128
    lr: float = 2e-4
    betas: tuple = (0.5, 0.999)
    epochs: int = 100
    lambda_l1: float = 100.0       # pix2pix L1 weight
    lambda_mask: float = 10.0      # extra weight on clouded pixels
    num_workers: int = 2
    seed: int = 42

    # --- diffusion ---
    timesteps: int = 1000
    sample_steps: int = 50         # DDIM steps at inference
    base_ch: int = 64

    # --- checkpointing (resumable across Kaggle/Colab sessions) ---
    ckpt_every: int = 1            # epochs
    log_every: int = 50            # steps


CFG = Config()
