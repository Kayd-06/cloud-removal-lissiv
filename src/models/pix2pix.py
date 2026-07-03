"""SAR-fusion pix2pix baseline: U-Net generator + PatchGAN discriminator.

Input  : [cloudy(3), sar(2), mask(1)] = 6 ch   (see CFG.in_channels)
Output : reconstructed clear optical (3 ch)

Kept intentionally compact so it trains in a few hours on a free P100 at 256x256.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _down(cin, cout, norm=True):
    layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(cout))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _up(cin, cout, dropout=False):
    layers = [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False),
              nn.InstanceNorm2d(cout), nn.ReLU(inplace=True)]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    """8-block U-Net (256->1->256) with skip connections."""

    def __init__(self, in_ch: int, out_ch: int, base: int = 64):
        super().__init__()
        self.d1 = _down(in_ch, base, norm=False)   # 128
        self.d2 = _down(base, base * 2)            # 64
        self.d3 = _down(base * 2, base * 4)        # 32
        self.d4 = _down(base * 4, base * 8)        # 16
        self.d5 = _down(base * 8, base * 8)        # 8
        self.d6 = _down(base * 8, base * 8)        # 4
        self.d7 = _down(base * 8, base * 8)        # 2
        self.d8 = _down(base * 8, base * 8, norm=False)  # 1

        self.u1 = _up(base * 8, base * 8, dropout=True)
        self.u2 = _up(base * 16, base * 8, dropout=True)
        self.u3 = _up(base * 16, base * 8, dropout=True)
        self.u4 = _up(base * 16, base * 8)
        self.u5 = _up(base * 16, base * 4)
        self.u6 = _up(base * 8, base * 2)
        self.u7 = _up(base * 4, base)
        self.u8 = nn.Sequential(
            nn.ConvTranspose2d(base * 2, out_ch, 4, 2, 1),
            nn.Sigmoid(),  # reflectance in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3)
        d5 = self.d5(d4); d6 = self.d6(d5); d7 = self.d7(d6); d8 = self.d8(d7)
        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7], 1))
        u3 = self.u3(torch.cat([u2, d6], 1))
        u4 = self.u4(torch.cat([u3, d5], 1))
        u5 = self.u5(torch.cat([u4, d4], 1))
        u6 = self.u6(torch.cat([u5, d3], 1))
        u7 = self.u7(torch.cat([u6, d2], 1))
        return self.u8(torch.cat([u7, d1], 1))


class PatchGAN(nn.Module):
    """70x70 PatchGAN. Sees the conditioning input concatenated with the target/fake."""

    def __init__(self, in_ch: int, out_ch: int, base: int = 64):
        super().__init__()
        c = in_ch + out_ch
        self.net = nn.Sequential(
            _down(c, base, norm=False),
            _down(base, base * 2),
            _down(base * 2, base * 4),
            nn.Conv2d(base * 4, base * 8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(base * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 8, 1, 4, 1, 1),
        )

    def forward(self, cond: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([cond, img], 1))
