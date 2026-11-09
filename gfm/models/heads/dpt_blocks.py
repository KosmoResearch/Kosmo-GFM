# DPT-family scratch and fusion blocks: see NOTICE and licenses/DPT-LICENSE.txt.
# Direct-source provenance and intermediate license terms require confirmation;
# see licenses/README.md. This notice does not relicense upstream adaptations.
"""Shared DPT building blocks (fusion / scratch / Resnet refine)."""

import torch
import torch.nn as nn

from ..layers.conv_blocks import ResidualConvUnit, Resnet


class Exp(nn.Module):
    def forward(self, x):
        return torch.exp(x)


def _make_fusion_block(
    features,
    use_bn,
    size=None,
    out_conv_cfg=None,
):
    return FeatureFusionDepthBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
        out_conv_cfg=out_conv_cfg,
    )

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3],
            out_shape4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )

    return scratch

class FeatureFusionDepthBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        out_conv_cfg=None,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionDepthBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1
        self.out_conv_cfg = out_conv_cfg

        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features // 2

        if self.out_conv_cfg is None:
            self.out_conv = nn.Conv2d(
                features,
                out_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                groups=1,
            )
        else:
            self.out_conv_cfg["input_channel"] = features
            self.out_conv = nn.Sequential(
                Resnet(**{k: v for k, v in self.out_conv_cfg.items() if k != "type"}),
                nn.Conv2d(
                    features,
                    out_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                    groups=1,
                ),
            )

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output
