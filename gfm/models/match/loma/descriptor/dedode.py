# Adapted from DeDoDe for Kosmo-GFM shared-feature inference.
# Copyright (c) 2023 Johan Edstedt
# MIT License: licenses/DeDoDe-LICENSE.txt.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from ..device import amp_dtype

class Decoder(nn.Module):
    def __init__(
        self, layers, *args, super_resolution=False, descriptor_dim=1, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.scales = sorted(list(layers.keys()), key=lambda x: int(x), reverse=True)
        self.super_resolution = super_resolution
        self.descriptor_dim = descriptor_dim

    def forward(self, features, context=None, scale=None):
        if context is not None:
            features = torch.cat((features, context), dim=1)
        stuff = self.layers[scale](features)
        logits, context = (
            stuff[:, : self.descriptor_dim],
            stuff[:, self.descriptor_dim :],
        )
        return logits, context


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        hidden_blocks: int,
        amp: bool,
        amp_dtype: torch.dtype,
        dw: bool = True,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.block1 = self.create_block(
            in_dim,
            hidden_dim,
            dw=False,
            kernel_size=1,
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.hidden_blocks = self.hidden_blocks
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        self.amp = amp
        self.amp_dtype = amp_dtype

    def create_block(
        self,
        in_dim,
        out_dim,
        dw=True,
        kernel_size=5,
        bias=True,
        norm_type=nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert out_dim % in_dim == 0, (
                "outdim must be divisible by indim for depthwise"
            )
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = (
            norm_type(out_dim)
            if norm_type is nn.BatchNorm2d
            else norm_type(num_channels=out_dim)
        )
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(self, feats):
        b, c, hs, ws = feats.shape
        with torch.autocast(
            device_type=feats.device.type, enabled=self.amp, dtype=self.amp_dtype
        ):
            x0 = self.block1(feats)
            x = self.hidden_blocks(x0)
            x = (x + x0) / 1.4
            x = self.out_conv(x)
            return x


class VGG(nn.Module):
    def __init__(self, size="19", amp=False, amp_dtype=amp_dtype) -> None:
        super().__init__()
        if size == "11":
            self.layers = nn.ModuleList(tvm.vgg11_bn().features[:22])
        elif size == "13":
            self.layers = nn.ModuleList(tvm.vgg13_bn().features[:28])
        elif size == "19":
            self.layers = nn.ModuleList(tvm.vgg19_bn().features[:40])
        # Maxpool layers: 6, 13, 26, 39
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        with torch.autocast(
            device_type=x.device.type, enabled=self.amp, dtype=self.amp_dtype
        ):
            feats = []
            sizes = []
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats.append(x)
                    sizes.append(x.shape[-2:])
                x = layer(x)
            return feats, sizes


def dedode_descriptor_G(descriptor_dim: int, hidden_blocks: int = 5, compute_dtype=None):
    amp = True
    compute_dtype = amp_dtype if compute_dtype is None else compute_dtype

    conv_refiner = nn.ModuleDict(
        {
            "14": ConvRefiner(
                1024,
                768,
                512 + descriptor_dim,
                hidden_blocks=hidden_blocks,
                amp=amp,
                amp_dtype=compute_dtype,
            ),
            "8": ConvRefiner(
                512 + 512,
                512,
                256 + descriptor_dim,
                hidden_blocks=hidden_blocks,
                amp=amp,
                amp_dtype=compute_dtype,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + descriptor_dim,
                hidden_blocks=hidden_blocks,
                amp=amp,
                amp_dtype=compute_dtype,
            ),
            "2": ConvRefiner(
                128 + 128,
                64,
                32 + descriptor_dim,
                hidden_blocks=hidden_blocks,
                amp=amp,
                amp_dtype=compute_dtype,
            ),
            "1": ConvRefiner(
                64 + 32,
                32,
                1 + descriptor_dim,
                hidden_blocks=hidden_blocks,
                amp=amp,
                amp_dtype=compute_dtype,
            ),
        }
    )
    # Keep encoder.vgg checkpoint keys while using the shared GFM encoder.
    encoder = nn.Module()
    encoder.vgg = VGG(size="19", amp=amp, amp_dtype=compute_dtype)
    decoder = Decoder(conv_refiner, descriptor_dim=descriptor_dim)
    return encoder, decoder
