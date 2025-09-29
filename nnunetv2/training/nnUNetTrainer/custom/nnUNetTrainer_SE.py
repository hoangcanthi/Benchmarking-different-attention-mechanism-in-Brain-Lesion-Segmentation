from __future__ import annotations

from typing import Union, List, Tuple

from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from .se import SqueezeExcitation2D


class nnUNetTrainer_SE(nnUNetTrainer):


    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        net = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )

        is_2d = any(m.__class__.__name__ == 'Conv2d' for m in net.modules())
        if not is_2d:
            return net

        def last_conv_out_channels(mod: nn.Module):
            last = None
            for m in mod.modules():
                if isinstance(m, nn.Conv2d):
                    last = m
            return last.out_channels if last is not None else None

        class StageWithSE(nn.Module):
            def __init__(self, stage: nn.Module, se: nn.Module):
                super().__init__()
                self.stage = stage
                self.se = se
            def forward(self, x):
                x = self.stage(x)
                return self.se(x)

        try:
            enc_stages = net.encoder.stages
            for i, s in enumerate(enc_stages):
                ch = last_conv_out_channels(s)
                if ch is None:
                    continue
                enc_stages[i] = StageWithSE(s, SqueezeExcitation2D(ch, reduction=16))


            if hasattr(net, 'bottleneck'):
                ch_bn = last_conv_out_channels(net.bottleneck)
                if ch_bn is not None:
                    net.bottleneck = nn.Sequential(net.bottleneck, SqueezeExcitation2D(ch_bn, reduction=16))


            if hasattr(net, 'decoder') and hasattr(net.decoder, 'stages'):
                dec_stages = net.decoder.stages
                for i, s in enumerate(dec_stages):
                    ch = last_conv_out_channels(s)
                    if ch is None:
                        continue
                    dec_stages[i] = StageWithSE(s, SqueezeExcitation2D(ch, reduction=16))
        except Exception as e:
            print(f"[SE] Injection failed: {e}")

        return net


