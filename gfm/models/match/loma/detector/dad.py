# Adapted from DaD and DeDoDe for Kosmo-GFM inference.
# Copyright (c) 2025 Johan Edstedt (DaD); 2023 Johan Edstedt (DeDoDe).
# MIT Licenses: licenses/DaD-LICENSE.txt and licenses/DeDoDe-LICENSE.txt.

from dataclasses import dataclass
from typing import Literal
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as transforms
from ..device import device, amp_dtype, resolve_amp_dtype
from .utils import sample_keypoints
logger = logging.getLogger(__name__)

class DaD(nn.Module):
    @dataclass(frozen=True)
    class Cfg:
        compile: bool = True
        precision: Literal["auto", "fp16", "bf16"] = "auto"
        remove_borders: bool = False
        resize: int = 1024
        nms_size: int = 3
        increase_coverage: bool = False
        coverage_pow: int | None = None
        coverage_size: int | None = None
        subpixel: bool = True
        subpixel_temp: float = 0.5
        keep_aspect_ratio: bool = True
        coverage_from_sparse: bool = False
        arch: Literal["dedode_s"] = "dedode_s"
        is_sigmoid: bool = False

    def __init__(
        self,
        cfg: Cfg | None = None,
        init_device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = DaD.Cfg()
        
        if cfg.arch == "dedode_s":
            encoder, decoder = dedode_detector_S(compute_dtype=resolve_amp_dtype(cfg.precision))
        else:
            raise ValueError(f"Architecture {cfg.arch} not supported")
        self.cfg = cfg
        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.encoder = encoder
        self.decoder = decoder
        self.remove_borders = cfg.remove_borders
        self.resize = cfg.resize
        self.increase_coverage = cfg.increase_coverage
        self.coverage_pow = cfg.coverage_pow
        self.coverage_size = cfg.coverage_size
        self.nms_size = cfg.nms_size
        self.keep_aspect_ratio = cfg.keep_aspect_ratio
        self.subpixel = cfg.subpixel
        self.subpixel_temp = cfg.subpixel_temp
        self.coverage_from_sparse = cfg.coverage_from_sparse
        if cfg.compile:
            logger.info("Compiling DaD...")
            self.compile()
        self.to(device if init_device is None else init_device)

    @property
    def topleft(self):
        return 0.5

    def forward_impl(
        self,
        images,
    ):
        images = self.normalizer(images)
        features, sizes = self.encoder(images)
        logits: torch.Tensor = images[:, :1].new_zeros(())
        context = None
        scales = ["8", "4", "2", "1"]
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales)):
            delta_logits, context = self.decoder(
                feature_map, context=context, scale=scale
            )
            logits = (
                logits + delta_logits.float()
            )  # ensure float (need bf16 doesnt have f.interpolate)
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                logits = F.interpolate(
                    logits, size=size, mode="bicubic", align_corners=False
                )
                context = F.interpolate(
                    context.float(), size=size, mode="bilinear", align_corners=False
                )
        return logits.float()

    def forward(
        self,
        images: torch.Tensor,
        num_keypoints: int,
        *,
        return_dense_probs: bool = False,
    ) -> dict[str, torch.Tensor]:
        scoremap = self.forward_impl(images)
        B, K, H, W = scoremap.shape
        dense_probs = (
            scoremap.reshape(B, K * H * W)
            .softmax(dim=-1)
            .reshape(B, K, H * W)
            .sum(dim=1)
        )
        dense_probs = dense_probs.reshape(B, H, W)
        keypoints, confidence = sample_keypoints(
            dense_probs,
            use_nms=True,
            nms_size=self.nms_size,
            sample_topk=True,
            num_samples=num_keypoints,
            return_probs=True,
            increase_coverage=self.increase_coverage,
            coverage_from_sparse=self.coverage_from_sparse,
            remove_borders=self.remove_borders,
            coverage_pow=float(self.coverage_pow)
            if self.coverage_pow is not None
            else 0.5,
            coverage_size=self.coverage_size if self.coverage_size is not None else 51,
            subpixel=self.subpixel,
            subpixel_temp=self.subpixel_temp,
            scoremap=scoremap.reshape(B, H, W),
        )
        result = {"keypoints": keypoints, "keypoint_probs": confidence}
        if return_dense_probs:
            result["dense_probs"] = dense_probs
        return result



class Decoder(nn.Module):
    def __init__(
        self, layers, *args, super_resolution=False, num_prototypes=1, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.super_resolution = super_resolution
        self.num_prototypes = num_prototypes

    def forward(self, features, context=None, scale=None):
        if context is not None:
            features = torch.cat((features, context), dim=1)
        stuff = self.layers[scale](features)
        logits, context = (
            stuff[:, : self.num_prototypes],
            stuff[:, self.num_prototypes :],
        )
        return logits, context


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=True,
        kernel_size=5,
        hidden_blocks=5,
        amp=True,
        residual=False,
        amp_dtype=amp_dtype,
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
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.residual = residual

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
            if self.residual:
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
        # print(f"device.type: {self.device}")
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


def dedode_detector_S(compute_dtype=None):
    compute_dtype = amp_dtype if compute_dtype is None else compute_dtype
    residual = True
    hidden_blocks = 3
    amp = True
    NUM_PROTOTYPES = 1
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=compute_dtype,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=compute_dtype,
            ),
            "2": ConvRefiner(
                128 + 128,
                64,
                32 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=compute_dtype,
            ),
            "1": ConvRefiner(
                64 + 32,
                32,
                1 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=compute_dtype,
            ),
        }
    )
    encoder = VGG(size="11", amp=amp, amp_dtype=compute_dtype)
    decoder = Decoder(conv_refiner)
    return encoder, decoder
