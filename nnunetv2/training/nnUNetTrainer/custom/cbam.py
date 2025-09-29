import torch
from torch import nn
import torch.nn.functional as F


class ChannelAttention2D(nn.Module):
    """
    Channel attention: shared MLP on global avg/max pooled descriptors.
    Paper spec: reduction ratio r (default 16), 2-layer MLP, bias=False.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        assert channels > 0 and reduction > 0
        hidden = max(1, channels // reduction)
        # shared MLP (Conv1x1 is equivalent to Linear on 1x1 descriptors)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class SpatialAttention2D(nn.Module):
    """
    Spatial attention: concat of channel-wise avg & max -> 7x7 conv -> sigmoid.
    Paper uses kernel_size=7 (bias=False). kernel_size=3 is the light variant.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)      # channel-wise avg
        max_map, _ = torch.max(x, dim=1, keepdim=True)    # channel-wise max
        x_cat = torch.cat([avg_map, max_map], dim=1)      # (B, 2, H, W)
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn


class CBAM2D(nn.Module):
    """
    CBAM block (2D): Channel Attention -> Spatial Attention.
    Matches the original CBAM ordering and ops.
    """
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention2D(channels, reduction=reduction)
        self.sa = SpatialAttention2D(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x
