from __future__ import annotations

from torch import nn


class SqueezeExcitation2D(nn.Module):
    """
    Classic Squeeze-and-Excitation (SE) block for 2D feature maps.

    Applies channel-wise attention via global average pooling followed by
    a small MLP (implemented with 1x1 convolutions).
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x)
        s = self.fc1(s)
        s = self.relu(s)
        s = self.fc2(s)
        s = self.sigmoid(s)
        return x * s


