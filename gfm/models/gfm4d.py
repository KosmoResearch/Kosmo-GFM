"""Standalone GFM4D inference network (RGB multi-view + motion).

While ``gfm3d.py`` targets static 3D reconstruction, GFM4D is the 4D variant:
it reuses the DA3 ``DinoV2`` fuse encoder and ``CameraHead``,
and adds a ``QueryBank``, a ``PairCrossAttnAggregator`` and two MLP heads
to predict per-view geometry plus cross-view 2D/3D motion.

Only the branches the config actually exercises are kept:
    Stage 1  fuse_encoder            -> patch features
    Stage 2  query_bank              -> dense UV query grid
    Stage 5  camera_head             -> pose_enc -> extrinsic / intrinsic
    Stage 7  pair cross-attn + heads -> per-view depth / points + cross-view motion
The single-view depth head and the Gaussian-splatting branch are intentionally
absent (they are not configured for this model).

Per-view geometry is read off the self-pairs ``(v, v)`` and cross-view motion is
kept per pair:
    pair_idx = [(0, i) for i in 1..N-1] + [(i, i) for i in 0..N-1]
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from gfm.models.aggregator import DinoV2
from gfm.models.heads.camera_head import CameraHead
from gfm.models.heads.mlp_head import Head
from gfm.models.pair_aggregator import PairCrossAttnAggregator
from gfm.models.query_bank import BaseQuery, QueryBank
from gfm.utils.geometry import normalize_cameras_to_first
from gfm.utils.pose_enc import pose_encoding_to_extri_intri


# Backbone config for the DA3 ViT-g encoder used by the 4D model.
_VITG = dict(embed_dim=1536, out_layers=[19, 27, 33, 39], alt_start=13)


class GFM4D(nn.Module):
    """GFM4D inference network."""

    def __init__(
        self,
        patch_size: int = 14,
        chunk_size: Optional[int] = 30000,
        num_register_tokens: int = 4,
        enable_warp2d_visibility: bool = True,
    ) -> None:
        super().__init__()
        if num_register_tokens < 0:
            raise ValueError("num_register_tokens must be >= 0")
        self.patch_size = patch_size
        if chunk_size is not None and (not isinstance(chunk_size, int) or chunk_size < 1):
            raise ValueError("chunk_size must be a positive integer or None")
        self.chunk_size = chunk_size
        self.enable_warp2d_visibility = enable_warp2d_visibility

        cfg = _VITG
        dim_in = cfg["embed_dim"] * 2  # cat_token=True -> 3072

        self.fuse_encoder = DinoV2(
            name="vitg",
            out_layers=cfg["out_layers"],
            alt_start=cfg["alt_start"],
            qknorm_start=cfg["alt_start"],
            rope_start=cfg["alt_start"],
            cat_token=True,
            num_register_tokens=num_register_tokens,
            register_attention_block_indices=[],
            patch_size=patch_size,
        )

        self.query_banck = QueryBank()

        self.camera_head = CameraHead(
            dim_in=dim_in,
            trunk_depth=4,
            pose_encoding_type="absT_quaR_FoV",
            num_heads=16,
            mlp_ratio=4,
            init_values=0.01,
            trans_act="linear",
            quat_act="linear",
            fl_act="relu",
            num_iterations=4,
        )

        self.query_pair_feats_aggregator = PairCrossAttnAggregator(
            patch_size=patch_size,
            intermediate_layer_idx=[0, 1, 2, 3],
            in_chans=[dim_in, dim_in, dim_in, dim_in],
            embed_dims=[256, 256, 256, 256],
            upsample_scales=[4, 2, 1, 1],
            mode="bilinear",
            interp_refinenet_cfg=dict(
                type="Resnet",
                layer_num=1,
                use_bn=False,
            ),
            cross_attn_dim=256,
            cross_attn_heads=8,
            cross_attn_layers=4,
            cross_ff_ratio=4,
            cross_dropout=0.0,
            align_corners=True,
        )

        self.query_pair_decoder = Head(
            in_chan=256,
            hidden_dim=64,
            names=dict(
                warp3d=3, warp3d_confidence=1,
                warp3d_delta_direction=3, warp3d_delta_magnitude=1, warp3d_delta_confidence=1,
                warp2d=2, warp2d_confidence=4,
                **({"warp2d_visibility": 1} if enable_warp2d_visibility else {}),
            ),
            acts=dict(
                warp3d="inv_log", warp3d_confidence="",
                warp3d_delta_direction="norm", warp3d_delta_magnitude="softplus", warp3d_delta_confidence="",
                warp2d="", warp2d_confidence="",
                **({"warp2d_visibility": ""} if enable_warp2d_visibility else {}),
            ),
        )

        self.query_pair_decoder2 = Head(
            in_chan=256,
            hidden_dim=64,
            names=dict(pair_depth=1, pair_confidence=1),
            acts=dict(pair_depth="exp", pair_confidence=""),
        )

    @staticmethod
    def _build_pair_idx(num_views: int):
        """Cross-pairs (0, i) for every other view plus one self-pair (i, i) per view."""
        cross = [(0, i) for i in range(1, num_views)]
        self_pairs = [(i, i) for i in range(num_views)]
        return cross + self_pairs

    def pair_forward(self, src_pyramids, memory_tokens, query):
        """Query-dependent pair decode + two MLP decoders (return_src_hidden path).

        ``src_pyramids`` / ``memory_tokens`` come from
        ``query_pair_feats_aggregator.encode`` and are reused across query chunks,
        so the query-independent feature extraction is not recomputed per chunk.
        """
        feats, single_feats = self.query_pair_feats_aggregator.decode(
            src_pyramids, memory_tokens, query,
        )
        results = self.query_pair_decoder(x=feats)
        single_results = self.query_pair_decoder2(x=single_feats)
        results.update(single_results)
        return results

    def forward_queries(self, images: torch.Tensor, query_uv: torch.Tensor) -> dict:
        """Decode sparse normalized UV queries directly, without a dense pointmap.

        ``query_uv`` is [Q,2], in the shared image canvas's [0,1] UV convention.
        The same source queries are used for every pair. Outputs have shape
        [B,P,Q,C]; ``pair_idx`` gives their source/target view indices. Cameras
        are unnecessary for these reference-frame positions and are not decoded.
        """
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if images.ndim != 5 or images.shape[2] != 3 or images.shape[1] < 1:
            raise ValueError("Expected RGB images [B,N,3,H,W] with N >= 1")
        b, n, _, h, w = images.shape
        if h < self.patch_size or w < self.patch_size or h % self.patch_size or w % self.patch_size:
            raise ValueError("Image H/W must be positive multiples of patch_size")
        query_uv = torch.as_tensor(query_uv, device=images.device, dtype=torch.float32)
        if query_uv.ndim != 2 or query_uv.shape[1] != 2 or query_uv.shape[0] < 1:
            raise ValueError("Expected nonempty query_uv [Q,2]")
        if not torch.isfinite(query_uv).all():
            raise ValueError("query_uv must be finite")
        pair_idx = self._build_pair_idx(n)
        count = query_uv.shape[0]
        chunk_size = self.chunk_size if self.chunk_size is not None else count
        with torch.autocast(device_type=images.device.type, dtype=torch.float16,
                            enabled=images.is_cuda):
            features, start = self.fuse_encoder(images)
            tokens = [feature[:, :, start:] for feature in features]
            pyramids, memory = self.query_pair_feats_aggregator.encode(
                tokens, pair_idx, h // self.patch_size, w // self.patch_size)
            chunks = []
            for offset in range(0, count, chunk_size):
                query = BaseQuery(uv=query_uv[offset:offset + chunk_size],
                                  full_uv=False, width=w, height=h)
                chunks.append(self.pair_forward(pyramids, memory, query))
            outputs = {key: torch.cat([chunk[key] for chunk in chunks], dim=1)
                       .reshape(b, len(pair_idx), count, -1) for key in chunks[0]}
        outputs["pair_idx"] = pair_idx
        return outputs

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            images: [N,3,H,W] or [B,N,3,H,W], values in [0, 1].
        Returns:
            dict with per-view depth / confidence / global_points / global_confidence,
            pose_enc / extrinsic / intrinsic, cross-view motion (warp*), pair_idx,
            and the input images.
        """
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if images.ndim != 5:
            raise ValueError(f"Expected images [B,N,3,H,W], got {tuple(images.shape)}")

        b, n, _, h, w = images.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"H/W must be divisible by patch_size={self.patch_size}, got {(h, w)}"
            )

        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        pair_idx = self._build_pair_idx(n)
        num_pair = len(pair_idx)

        amp_dtype = torch.float16
        device_type = images.device.type

        pose_enc = None
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=images.is_cuda):
            # Stage 1: dense feature extraction
            patch_features, patch_start_idx = self.fuse_encoder(images)
            if not isinstance(patch_features, (list, tuple)):
                patch_features = [patch_features]
            patch_tokens = [f[:, :, patch_start_idx:] for f in patch_features]

            # Stage 2: dense full-grid query
            query = self.query_banck(images)
            query_nums = query.query_nums

            # Stage 5: camera tokens -> pose encoding
            camera_tokens = [patch_features[-1][:, :, :patch_start_idx]]
            pose_enc = self.camera_head(camera_tokens)

            # Stage 7: pair-wise cross-view matching.
            # The pyramid features / cross-attn memory are query-independent, so build
            # them once and only sample + cross-attend per query chunk.
            src_pyramids, memory_tokens = self.query_pair_feats_aggregator.encode(
                patch_tokens, pair_idx, patch_h, patch_w,
            )
            if self.chunk_size is not None and query_nums > self.chunk_size:
                chunk_list = []
                for q0 in range(0, query_nums, self.chunk_size):
                    q1 = min(q0 + self.chunk_size, query_nums)
                    chunk_query = BaseQuery(uv=query.uv[q0:q1])
                    chunk_list.append(self.pair_forward(src_pyramids, memory_tokens, chunk_query))
                pair_results = {}
                for key in chunk_list[0].keys():
                    pair_results[key] = torch.cat([res[key] for res in chunk_list], dim=1)
            else:
                pair_results = self.pair_forward(src_pyramids, memory_tokens, query)

        # [B, P, Q, C] with Q the row-major (H, W) grid.
        pair_results = {k: v.float().view(b, num_pair, h, w, v.shape[-1]) for k, v in pair_results.items()}

        predictions: Dict[str, torch.Tensor] = {}

        # Per-view geometry from the self-pairs (v, v): [B, N, C, H, W].
        self_ids = [pair_idx.index((v, v)) for v in range(n)]
        predictions["depth"] = pair_results["pair_depth"][:, self_ids].permute(0, 1, 4, 2, 3).contiguous()
        predictions["confidence"] = pair_results["pair_confidence"][:, self_ids].permute(0, 1, 4, 2, 3).contiguous()
        predictions["global_points"] = pair_results["warp3d"][:, self_ids].permute(0, 1, 4, 2, 3).contiguous()
        predictions["global_confidence"] = pair_results["warp3d_confidence"][:, self_ids].permute(0, 1, 4, 2, 3).contiguous()

        # Cross-view motion (kept per pair): [B, P, C, H, W].
        # ``warp3d`` is the absolute predicted world point of the src-view query at
        # the tgt frame; ``warp3d_delta`` is its (independently decoded) scene flow.
        for key in ("warp3d", "warp3d_confidence",
                    "warp3d_delta_direction", "warp3d_delta_magnitude", "warp3d_delta_confidence",
                    "warp2d", "warp2d_confidence"):
            predictions[key] = pair_results[key].permute(0, 1, 4, 2, 3).contiguous()
        if self.enable_warp2d_visibility:
            predictions["warp2d_visibility"] = pair_results["warp2d_visibility"].permute(0, 1, 4, 2, 3).contiguous()
        predictions["pair_idx"] = pair_idx

        # Decode cameras in fp32 OUTSIDE autocast (quaternion->matrix + normalization
        # are precision sensitive; fp16 makes poses drift).
        if isinstance(pose_enc, (list, tuple)):
            pose_enc = pose_enc[-1]
        predictions["pose_enc"] = pose_enc
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_encoding=pose_enc.float(),
            image_size_hw=(h, w),
            build_intrinsics=True,
        )
        extrinsics = normalize_cameras_to_first(extrinsics)
        predictions["extrinsic"] = extrinsics[..., :3, :]
        predictions["intrinsic"] = intrinsics

        predictions["images"] = images
        return predictions

    def load_checkpoint(self, checkpoint, strict: bool = True, map_location: str = "cpu") -> None:
        """Load a state_dict; strips a leading ``model.`` prefix if present."""
        state = torch.load(os.fspath(checkpoint), map_location=map_location, weights_only=True)
        if all(k.startswith("model.") for k in state.keys()):
            state = {k[len("model."):]: v for k, v in state.items()}
        missing, unexpected = self.load_state_dict(state, strict=strict)
        if missing:
            print(f"[GFM4D] missing keys ({len(missing)}): {missing[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"[GFM4D] unexpected keys ({len(unexpected)}): {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
