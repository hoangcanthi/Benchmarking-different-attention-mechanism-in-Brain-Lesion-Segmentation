from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class AxialAttention2D(nn.Module):
    """Axial (row/column) self-attention block for 2D feature maps.

    This approximates criss-cross attention by sequentially applying attention
    along height then width. It is lightweight and VRAM-friendly compared to
    full 2D non-local attention.
    """

    def __init__(
        self,
        in_channels: int,
        attn_channels: int | None = None,
        heads: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        c_attn = attn_channels if attn_channels is not None else max(8, in_channels // 4)
        assert c_attn % heads == 0, "attn_channels must be divisible by heads"
        self.c_per_head = c_attn // heads

        # shared projections for H and W attentions
        self.to_q = nn.Conv2d(in_channels, c_attn, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(in_channels, c_attn, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(in_channels, c_attn, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(c_attn, in_channels, kernel_size=1, bias=False)

        # Swap BN -> GN per request
        self.norm = nn.GroupNorm(32, in_channels)
        # Residual scaling (LayerScale-lite)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _attend_height(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> treat each column independently
        B, C, H, W = q.shape
        # reshape to (B*W, heads, H, c_head)
        def prep(t):
            t = t.permute(0, 3, 2, 1).contiguous()  # (B, W, H, C)
            t = t.view(B * W, H, self.heads, self.c_per_head).permute(0, 2, 1, 3)  # (BW, heads, H, c)
            return t
        qh, kh, vh = map(prep, (q, k, v))
        scale = 1.0 / math.sqrt(self.c_per_head)
        attn = torch.matmul(qh, kh.transpose(-2, -1)) * scale  # (BW, heads, H, H)
        attn = attn - attn.max(dim=-1, keepdim=True).values  # numerical stability
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, vh)  # (BW, heads, H, c)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, W, H, self.heads * self.c_per_head)
        out = out.permute(0, 3, 2, 1).contiguous()  # (B, Cattn, H, W)
        return out

    def _attend_width(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> treat each row independently
        B, C, H, W = q.shape
        def prep(t):
            t = t.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
            t = t.view(B * H, W, self.heads, self.c_per_head).permute(0, 2, 1, 3)  # (BH, heads, W, c)
            return t
        qw, kw, vw = map(prep, (q, k, v))
        scale = 1.0 / math.sqrt(self.c_per_head)
        attn = torch.matmul(qw, kw.transpose(-2, -1)) * scale  # (BH, heads, W, W)
        attn = attn - attn.max(dim=-1, keepdim=True).values  # numerical stability
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, vw)  # (BH, heads, W, c)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, H, W, self.heads * self.c_per_head)
        out = out.permute(0, 3, 1, 2).contiguous()  # (B, Cattn, H, W)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # projections
        dtype_in = x.dtype
        q = self.to_q(x).float()
        k = self.to_k(x).float()
        v = self.to_v(x).float()

        # height attention then width attention (criss-cross sequential)
        h_out = self._attend_height(q, k, v)
        w_out = self._attend_width(q, k, v)
        out = h_out + w_out
        out = self.dropout(out)
        out = self.proj(out)
        out = self.norm(out)
        out = out.to(dtype_in)
        return x + self.gamma * out


class CrissCrossAttentionBlock2D(nn.Module):
    """Wrapper block: axial attention + optional conv refinement."""

    def __init__(self, channels: int, attn_channels: int | None = None, heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.attn = AxialAttention2D(channels, attn_channels=attn_channels, heads=heads, dropout=dropout)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.refine(x)
        return x


