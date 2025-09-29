from __future__ import annotations

from typing import Any, Union, List, Tuple

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from .ag import AttentionGate2D


class nnUNetTrainer_AttentionGate(nnUNetTrainer):
    """
    Trainer that injects Attention Gates on skip connections of the 2D PlainConvUNet decoder.

    Usage:
      nnUNetv2_train 201 2d 0 -tr nnUNetTrainer_AttentionGate
    """

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

        # Only target 2D UNet
        is_2d = any(m.__class__.__name__ == 'Conv2d' for m in net.modules())
        if not is_2d:
            return net

        try:
            # encoder: net.encoder with stages and output channels
            # decoder: net.decoder with transpconvs and stages, and forward that cat([up, skip])

            decoder = net.decoder
            encoder = net.encoder

            # Build Attention Gates per decoder stage; match channels
            gates = []
            n_stages = len(decoder.stages)
            for s in range(n_stages):
                ch_skip = encoder.output_channels[-(s + 2)]
                ch_gating = decoder.transpconvs[s].out_channels
                gates.append(AttentionGate2D(ch_skip, ch_gating, inter_channels=max(1, ch_skip // 2)))

            decoder.attention_gates = nn.ModuleList(gates)

            # Wrap decoder forward to insert gating before concatenation
            original_forward = decoder.forward

            def gated_forward(skips):
                lres_input = skips[-1]
                seg_outputs = []
                for s in range(len(decoder.stages)):
                    x = decoder.transpconvs[s](lres_input)
                    skip = skips[-(s + 2)]
                    # apply gate on skip using x as gating signal
                    gated_skip = decoder.attention_gates[s](skip, x)
                    x = torch.cat((x, gated_skip), 1)
                    x = decoder.stages[s](x)
                    if decoder.deep_supervision:
                        seg_outputs.append(decoder.seg_layers[s](x))
                    elif s == (len(decoder.stages) - 1):
                        seg_outputs.append(decoder.seg_layers[-1](x))
                    lres_input = x

                seg_outputs = seg_outputs[::-1]
                if not decoder.deep_supervision:
                    return seg_outputs[0]
                return seg_outputs

            decoder.forward = gated_forward  # type: ignore
        except Exception as e:
            print(f"[AttentionGate] Injection failed: {e}")

        return net


