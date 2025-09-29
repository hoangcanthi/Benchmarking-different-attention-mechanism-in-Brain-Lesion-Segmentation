from __future__ import annotations

from typing import Any

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from .crisscross_attention import CrissCrossAttentionBlock2D


class nnUNetTrainer_CrissCross(nnUNetTrainer):
    """Trainer that mirrors criss-cross (axial) attention in encoder and decoder (2D).

    Usage:
      nnUNetv2_train 201 2d 0 -tr nnUNetTrainer_CrissCross
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Safer learning rate for attention-augmented nets
        self.initial_lr = 1e-2

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        net = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )

        # Only for 2D conv UNet
        is_2d = any(m.__class__.__name__ == 'Conv2d' for m in net.modules())
        if is_2d:
            nnUNetTrainer_CrissCross._inject_crisscross_2d(net)
        return net

    @staticmethod
    def _inject_crisscross_2d(net: nn.Module) -> None:
        try:
            # Encoder: add attention after each stage (match CBAM placement)
            enc = net.encoder
            enc_stages = enc.stages
            enc_channels = []
            for s in enc_stages:
                first_conv = None
                for m in s.modules():
                    if isinstance(m, nn.Conv2d):
                        first_conv = m
                        break
                enc_channels.append(first_conv.out_channels if first_conv is not None else None)

            for i, ch in enumerate(enc_channels):
                if ch is None:
                    continue
                attn = CrissCrossAttentionBlock2D(channels=ch, attn_channels=max(8, ch // 4), heads=1, dropout=0.0)
                original = enc_stages[i]
                class StageWithAttn(nn.Module):
                    def __init__(self, stage: nn.Module, attn_block: nn.Module):
                        super().__init__()
                        self.stage = stage
                        self.attn = attn_block
                    def forward(self, x):
                        x = self.stage(x)
                        return self.attn(x)
                enc_stages[i] = StageWithAttn(original, attn)

            # Bottleneck: insert attention after bottleneck block if present
            if hasattr(net, "bottleneck"):
                first_conv = None
                for m in net.bottleneck.modules():
                    if isinstance(m, nn.Conv2d):
                        first_conv = m
                        break
                ch_bn = first_conv.out_channels if first_conv is not None else None
                if ch_bn is not None:
                    attn_bn = CrissCrossAttentionBlock2D(
                        channels=ch_bn,
                        attn_channels=max(8, ch_bn // 4),
                        heads=1,
                        dropout=0.0,
                    )
                    net.bottleneck = nn.Sequential(net.bottleneck, attn_bn)

            # Decoder: insert attention after each decoder stage stack (match CBAM placement)
            dec = net.decoder
            dec_stages = dec.stages  # list of StackedConvBlocks
            dec_channels = []
            for s in dec_stages:
                first_conv = None
                for m in s.modules():
                    if isinstance(m, nn.Conv2d):
                        first_conv = m
                        break
                dec_channels.append(first_conv.out_channels if first_conv is not None else None)

            for i, ch in enumerate(dec_channels):
                if ch is None:
                    continue
                attn = CrissCrossAttentionBlock2D(channels=ch, attn_channels=max(8, ch // 4), heads=1, dropout=0.0)
                original = dec_stages[i]
                class DecStageWithAttn(nn.Module):
                    def __init__(self, stage: nn.Module, attn_block: nn.Module):
                        super().__init__()
                        self.stage = stage
                        self.attn = attn_block
                    def forward(self, x):
                        x = self.stage(x)
                        return self.attn(x)
                dec_stages[i] = DecStageWithAttn(original, attn)
        except Exception as e:
            print(f"[CrissCross] Injection failed: {e}")


