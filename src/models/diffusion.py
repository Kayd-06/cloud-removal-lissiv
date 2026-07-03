"""Conditional DDPM for cloud reconstruction (the 'novelty' model).

We predict the clear image conditioned on [cloudy, sar, mask]. Conditioning is by
channel concatenation into a compact U-Net noise predictor. Training uses the
standard epsilon-prediction objective; sampling uses DDIM for fast (50-step) inference.

Deliberately small (base=64, 3 resolutions) so fine-tuning fits a free P100. Intended
to be *fine-tuned* from a SEN12MS-CR checkpoint, not trained from scratch.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------- building blocks ------------------------- #
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.temb = nn.Linear(tdim, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class CondUNet(nn.Module):
    """Noise predictor. in_ch = target(3) + conditioning(cond_ch)."""

    def __init__(self, target_ch: int, cond_ch: int, base: int = 64):
        super().__init__()
        tdim = base * 4
        self.temb = nn.Sequential(nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.in_conv = nn.Conv2d(target_ch + cond_ch, base, 3, 1, 1)

        self.d1 = ResBlock(base, base, tdim)
        self.d2 = ResBlock(base, base * 2, tdim)
        self.d3 = ResBlock(base * 2, base * 4, tdim)
        self.down = nn.AvgPool2d(2)

        self.mid = ResBlock(base * 4, base * 4, tdim)

        self.u3 = ResBlock(base * 4 + base * 4, base * 2, tdim)
        self.u2 = ResBlock(base * 2 + base * 2, base, tdim)
        self.u1 = ResBlock(base + base, base, tdim)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(),
                                 nn.Conv2d(base, target_ch, 3, 1, 1))
        self.base = base

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        temb = self.temb(timestep_embedding(t, self.base))
        h = self.in_conv(torch.cat([x_t, cond], 1))
        h1 = self.d1(h, temb)
        h2 = self.d2(self.down(h1), temb)
        h3 = self.d3(self.down(h2), temb)
        m = self.mid(h3, temb)
        u = self.u3(torch.cat([m, h3], 1), temb)
        u = self.u2(torch.cat([self.up(u), h2], 1), temb)
        u = self.u1(torch.cat([self.up(u), h1], 1), temb)
        return self.out(u)


# ------------------------- diffusion process ------------------------- #
class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, device: str = "cpu"):
        self.T = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.acp = torch.cumprod(self.alphas, 0)          # alpha-bar
        self.device = device

    def q_sample(self, x0, t, noise):
        acp = self.acp[t][:, None, None, None]
        return acp.sqrt() * x0 + (1 - acp).sqrt() * noise

    def loss(self, model, x0, cond):
        b = x0.size(0)
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = model(x_t, t, cond)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def ddim_sample(self, model, cond, shape, steps: int = 50, eta: float = 0.0):
        """Fast deterministic sampling. Returns reconstructed clear image in [0,1]-ish."""
        x = torch.randn(shape, device=self.device)
        ts = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=self.device)
        for i in range(steps):
            t = ts[i].expand(shape[0])
            acp_t = self.acp[t][:, None, None, None]
            eps = model(x, t, cond)
            x0 = (x - (1 - acp_t).sqrt() * eps) / acp_t.sqrt()
            x0 = x0.clamp(0, 1)
            if i < steps - 1:
                t_next = ts[i + 1].expand(shape[0])
                acp_n = self.acp[t_next][:, None, None, None]
                sigma = eta * ((1 - acp_n) / (1 - acp_t)).sqrt() * (1 - acp_t / acp_n).sqrt()
                x = acp_n.sqrt() * x0 + (1 - acp_n - sigma ** 2).clamp(min=0).sqrt() * eps
                if eta > 0:
                    x = x + sigma * torch.randn_like(x)
            else:
                x = x0
        return x.clamp(0, 1)
