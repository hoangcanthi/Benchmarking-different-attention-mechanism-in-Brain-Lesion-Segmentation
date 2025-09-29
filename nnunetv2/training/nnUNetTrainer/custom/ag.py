from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class AttentionGate2D(nn.Module):

    def __init__(
        self,
        in_channels_skip: int,
        in_channels_gating: int,
        inter_channels: Optional[int] = None,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if inter_channels is None:
            inter_channels = max(1, in_channels_skip // 2)

        self.theta_x = nn.Conv2d(in_channels_skip, inter_channels, kernel_size=1, bias=False)
        self.phi_g = nn.Conv2d(in_channels_gating, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)

        # gn for more stable normalization at small batch sizes
        self.norm = nn.GroupNorm(32, inter_channels)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        # gate strength (scalar). 1.0 = full gating, 0.0 = pass-through
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_l: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # skip and gating to inter_channels
        theta_x = self.theta_x(x_l)
        phi_g = self.phi_g(g)

        #resize gating to match skip spatial size
        if phi_g.shape[-2:] != theta_x.shape[-2:]:
            phi_g = F.interpolate(phi_g, size=theta_x.shape[-2:], mode="bilinear", align_corners=False)

        f = self.relu(self.norm(theta_x + phi_g))
        psi = self.sigmoid(self.psi(f))  # (N, 1, H, W)
        return x_l * ((1 - self.gamma) + self.gamma * psi)


