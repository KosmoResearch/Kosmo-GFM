"""Camera-prompt encoder used by camera-conditioned GFM checkpoints.

This is an inference-only implementation kept entirely inside :mod:`gfm`.
It converts known world-to-camera matrices and pinhole intrinsics to the same
nine-dimensional pose representation used by the camera head, then embeds and
mixes the per-view tokens with a small transformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gfm.models.layers.block import Block
from gfm.models.layers.mlp import Mlp
from gfm.utils.pose_enc import extri_intri_to_pose_encoding


class CameraPromptEncoder(nn.Module):
    """Encode known cameras as one prompt token per input view."""

    def __init__(
        self,
        dim_out: int = 1536,
        dim_in: int = 9,
        trunk_depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    qkv_bias=True,
                    ln_eps=1e-5,
                )
                for _ in range(trunk_depth)
            ]
        )
        self.token_norm = nn.LayerNorm(dim_out)
        self.trunk_norm = nn.LayerNorm(dim_out)
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_out // 2,
            out_features=dim_out,
            drop=0,
        )

    def forward(
        self,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Return camera prompt tokens with shape ``[B, N, dim_out]``.

        ``extrinsics`` are OpenCV world-to-camera transforms. Both ``3x4`` and
        homogeneous ``4x4`` matrices are accepted.
        """
        if extrinsics.ndim != 4 or extrinsics.shape[-2:] not in ((3, 4), (4, 4)):
            raise ValueError(
                "prompt_extrinsics must be [B,N,3,4] or [B,N,4,4], "
                f"got {tuple(extrinsics.shape)}"
            )
        if intrinsics.ndim != 4 or intrinsics.shape[-2:] != (3, 3):
            raise ValueError(
                f"prompt_intrinsics must be [B,N,3,3], got {tuple(intrinsics.shape)}"
            )
        if extrinsics.shape[:2] != intrinsics.shape[:2]:
            raise ValueError("prompt extrinsics and intrinsics must have the same B,N axes")

        pose_encoding = extri_intri_to_pose_encoding(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_size_hw=image_size_hw,
        )
        pose_tokens = self.pose_branch(pose_encoding)
        pose_tokens = self.token_norm(pose_tokens)
        pose_tokens = self.trunk(pose_tokens)
        return self.trunk_norm(pose_tokens)
