from __future__ import annotations

from typing import Any, Union, List, Tuple

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context

from .cbam import CBAM2D
from .contrastive_pretrain import apply_pretrained_backbone_weights


class nnUNetTrainer_CBAM(nnUNetTrainer):
    """Custom trainer that injects CBAM modules into the encoder and decoder and bottleneck stages.

    Usage:
      nnUNetv2_train 201 2d 0 -tr nnUNetTrainer_CBAM
    """

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
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

        # Only add CBAM for 2D UNet (Conv2d)
        is_2d = any(m.__class__.__name__ == 'Conv2d' for m in net.modules())
        if is_2d:
            nnUNetTrainer_CBAM_Contrastive._inject_cbam_2d(net)


    @staticmethod
    def _get_env(key: str, default: Any = None) -> Any:
        import os
        return os.environ.get(key, default)

    @staticmethod
    def _inject_cbam_2d(net: nn.Module) -> None:
        """
        Insert CBAM after each encoder and decoder stage output in PlainConvUNet (2D).
        Keeps shapes unchanged and is safe with deep supervision heads.
        """
        def first_conv_out_channels(mod: nn.Module):
            for m in mod.modules():
                if isinstance(m, nn.Conv2d):
                    return m.out_channels
            return None

        class StageWithCBAM(nn.Module):
            def __init__(self, stage: nn.Module, attn: nn.Module):
                super().__init__()
                self.stage = stage
                self.attn = attn
            def forward(self, x):
                x = self.stage(x)
                return self.attn(x)

        try:
            # ----- Encoder -----
            enc = net.encoder
            enc_stages = enc.stages  # ModuleList
            for i, s in enumerate(enc_stages):
                ch = first_conv_out_channels(s)
                if ch is None:
                    continue
                cbam = CBAM2D(channels=ch, reduction=16, spatial_kernel=7)
                enc_stages[i] = StageWithCBAM(s, cbam)

            # ----- Bottleneck -----
            if hasattr(net, "bottleneck"):
                ch = first_conv_out_channels(net.bottleneck)
                if ch is not None:
                    net.bottleneck = nn.Sequential(net.bottleneck, CBAM2D(ch, 16, 7))

            # ----- Decoder -----
            if hasattr(net, "decoder"):
                dec = net.decoder
                if hasattr(dec, "stages"):
                    dec_stages = dec.stages
                    for i, s in enumerate(dec_stages):
                        ch = first_conv_out_channels(s)
                        if ch is None:
                            continue
                        cbam = CBAM2D(channels=ch, reduction=16, spatial_kernel=7)
                        dec_stages[i] = StageWithCBAM(s, cbam)
        except Exception as e:
            print(f"[CBAM_Contrastive] Could not inject CBAM (encoder/decoder): {e}")


