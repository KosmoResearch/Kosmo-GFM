"""Standalone GFM inference model (RGB multi-view reconstruction)."""

from __future__ import annotations

import os
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from gfm.models.aggregator import DinoV2
from gfm.models.camera_encoder import CameraPromptEncoder
from gfm.models.heads.camera_head import CameraHead
from gfm.models.heads.depth_head import DPTHead
from gfm.models.heads.metric_head import FrameMetricHead, MetricHead
from gfm.utils.geometry import normalize_cameras_to_first
from gfm.utils.pose_enc import pose_encoding_to_extri_intri


# Per-backbone architecture configuration for the GFM checkpoints.
# ``dim_in`` is 2 * embed_dim because the encoder runs with cat_token=True.
# alt_start == qknorm_start == rope_start for every size.
MODEL_CONFIGS = {
    "vits": dict(embed_dim=384, out_layers=[5, 7, 9, 11], alt_start=4, depth_features=64),
    "vitb": dict(embed_dim=768, out_layers=[5, 7, 9, 11], alt_start=4, depth_features=128),
    "vitl": dict(embed_dim=1024, out_layers=[11, 15, 19, 23], alt_start=8, depth_features=256),
    "vitg": dict(embed_dim=1536, out_layers=[19, 27, 33, 39], alt_start=13, depth_features=256),
}


class GFM(nn.Module):
    """GFM DA3 RGB feed-forward reconstructor.

    Architecture matches training config (selectable backbone size):
      fuse_encoder = DA3 DinoV2 ViT-S/B/L/G (alt/qk/rope from ``alt_start``)
      depth_head   = DPTHead (local metric depth + confidence)
      camera_head  = CameraHead (absT_quaR_FoV)
    """

    def __init__(
        self,
        model_size: str = "vitg",
        patch_size: int = 14,
        num_register_tokens: int = 4,
        use_camera_prompt: bool = False,
        enable_depth: bool = True,
        enable_camera: bool = True,
        enable_metric: bool = False,
        metric_type: str = "frame",
        enable_mvloma: bool = False,
        mvloma_precision: str = "bf16",
    ) -> None:
        super().__init__()
        if metric_type not in ("frame", "global"):
            raise ValueError("metric_type must be 'frame' or 'global'")
        self.metric_type = metric_type
        if model_size not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model_size {model_size!r}; expected one of {list(MODEL_CONFIGS)}"
            )
        cfg = MODEL_CONFIGS[model_size]
        self.model_size = model_size
        self.patch_size = patch_size
        if num_register_tokens < 0:
            raise ValueError("num_register_tokens must be >= 0")
        self.num_register_tokens = num_register_tokens
        self.use_camera_prompt = use_camera_prompt

        self.fuse_encoder = DinoV2(
            name=model_size,
            out_layers=cfg["out_layers"],
            alt_start=cfg["alt_start"],
            qknorm_start=cfg["alt_start"],
            rope_start=cfg["alt_start"],
            cat_token=True,
            num_register_tokens=num_register_tokens,
            register_attention_block_indices=[],
            use_prompt_camera=use_camera_prompt,
            prompt_camera_token_every=use_camera_prompt,
            patch_size=patch_size,
        )

        self.camera_encoder = (
            CameraPromptEncoder(dim_out=cfg["embed_dim"])
            if use_camera_prompt
            else None
        )

        dim_in = cfg["embed_dim"] * 2  # 2 * embed_dim with cat_token=True
        self.loma_match_head = None
        if enable_mvloma:
            from gfm.models.heads.mvloma_head import MVLoMAHead
            self.loma_match_head = MVLoMAHead(
                fuse_dim=dim_in, patch_size=patch_size, precision=mvloma_precision,
            )
        head = FrameMetricHead if metric_type == "frame" else MetricHead
        self.metric_head = head(dim_in=dim_in) if enable_metric else None
        self.depth_head = (
            DPTHead(
                in_channels=[dim_in, dim_in, dim_in, dim_in],
                mid_channels=[256, 256, 256, 256],
                patch_size=patch_size,
                features=cfg["depth_features"],
                features2=32,
                depth_channel=1,
                depth_act="exp",
                interp_refinenet_cfg=dict(
                    layer_num=3,
                    use_bn=False,
                ),
                pred_confidence=True,
                return_features=False,
                chunk_size=8,
            )
            if enable_depth
            else None
        )


        self.camera_head = (
            CameraHead(
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
            if enable_camera
            else None
        )

    def forward(
        self,
        images: torch.Tensor,
        prompt_extrinsics: Optional[torch.Tensor] = None,
        prompt_intrinsics: Optional[torch.Tensor] = None,
        mvloma_inputs: Optional[dict] = None,
        return_mvloma_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            images: [N,3,H,W] or [B,N,3,H,W], values in [0, 1].
            mvloma_inputs: Optional decoder inputs: normalized keypoints,
                raw01 descriptor_images, and optional pair_idx/transforms.
                describe_only=True returns sampled descriptors for full-STAR
                matching. Requires enable_mvloma or an attached decoder.
        Returns:
            dict with depth / confidence / pose_enc / extrinsic / intrinsic / images
            (geometry fields depend on enabled heads). Inference runners derive
            world_points_from_depth by unprojecting depth with the cameras.
            With mvloma_inputs, loma_match contains keypoints/descriptions and
            (unless describe_only) scores_by_pair from the shared features.
            With enable_metric, metric / log_metric are [B,N] for frame mode
            or [B] for global mode. Depth is already restored to pose units;
            exporters apply only the scene pose scale.
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

        amp_dtype = torch.float16
        device_type = images.device.type

        predictions: Dict[str, torch.Tensor] = {}
        pose_enc = None
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=images.is_cuda):
            camera_prompt = None
            if (prompt_extrinsics is None) != (prompt_intrinsics is None):
                raise ValueError(
                    "prompt_extrinsics and prompt_intrinsics must be supplied together"
                )
            if prompt_extrinsics is not None:
                if self.camera_encoder is None:
                    raise ValueError(
                        "camera prompts were supplied but use_camera_prompt=False"
                    )
                if prompt_extrinsics.ndim == 3:
                    prompt_extrinsics = prompt_extrinsics.unsqueeze(0)
                if prompt_intrinsics.ndim == 3:
                    prompt_intrinsics = prompt_intrinsics.unsqueeze(0)
                if prompt_extrinsics.shape[:2] != (b, n):
                    raise ValueError(
                        "prompt camera count must match images: expected "
                        f"{(b, n)}, got {tuple(prompt_extrinsics.shape[:2])}"
                    )
                camera_prompt = self.camera_encoder(
                    prompt_extrinsics.to(device=images.device),
                    prompt_intrinsics.to(device=images.device),
                    image_size_hw=(h, w),
                )

            patch_features, patch_start_idx = self.fuse_encoder(
                images, cam_token=camera_prompt
            )

            if self.metric_head is not None:
                predictions.update(self.metric_head(patch_features))

            if self.depth_head is not None:
                depth_out = self.depth_head(
                    patch_features,
                    patch_h=patch_h,
                    patch_w=patch_w,
                    patch_start_idx=patch_start_idx,
                )
                for key, val in depth_out.items():
                    predictions[key] = val.view(b, n, *val.shape[-3:])


            if self.camera_head is not None:
                camera_tokens = [patch_features[-1][:, :, :patch_start_idx]]
                pose_enc_list = self.camera_head(camera_tokens)
                pose_enc = pose_enc_list[-1] if isinstance(pose_enc_list, (list, tuple)) else pose_enc_list

            if return_mvloma_features:
                # Opt-in runtime: export the shared encoder output, not the
                # per-view descriptor loop, across the GFM compile boundary.
                predictions["mvloma_features"] = patch_features[-1][:, :, patch_start_idx:]
            if mvloma_inputs is not None:
                if self.loma_match_head is None:
                    raise ValueError("mvloma_inputs requires an enabled MVLoMA decoder")
                predictions["loma_match"] = self.loma_match_head(
                    images, patch_features, patch_start_idx, **mvloma_inputs,
                )

        # Restore frame- or chunk-normalized depth to pose units in fp32.
        # Camera translation and confidence do not use this ratio.
        if self.metric_head is not None and "depth" in predictions:
            ratio = predictions["metric"]
            if self.metric_type == "global":
                ratio = ratio[:, None]
            predictions["depth"] = (
                predictions["depth"].float()
                * ratio[..., None, None, None]
            )

        # Decode the camera pose in fp32 outside autocast.
        # Quaternion->matrix and camera normalization are precision
        # sensitive; running them in fp16 makes poses drift slightly (e.g. rpe_r).
        if pose_enc is not None:
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

    def load_checkpoint(
        self,
        checkpoint: Union[str, os.PathLike],
        strict: bool = True,
        map_location: str = "cpu",
    ) -> None:
        """Load a full state_dict checkpoint or a training-output directory."""
        path = os.fspath(checkpoint)
        if os.path.isdir(path):
            candidates = (
                os.path.join(path, "ckpt.pth"),
                os.path.join(path, "best", "ckpt.pth"),
                os.path.join(path, "latest", "ckpt.pth"),
                os.path.join(path, "checkpoint", "best", "ckpt.pth"),
                os.path.join(path, "checkpoint", "latest", "ckpt.pth"),
            )
            path = next((item for item in candidates if os.path.isfile(item)), "")
            if not path:
                raise FileNotFoundError(
                    f"No ckpt.pth found in checkpoint directory {checkpoint}"
                )
        elif not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(
            path, map_location=map_location, weights_only=True, mmap=True
        )
        if isinstance(state, dict):
            for wrapper in ("model", "state_dict"):
                if isinstance(state.get(wrapper), dict):
                    state = state[wrapper]
                    break
        missing, unexpected = self.load_state_dict(state, strict=strict)
        if missing:
            print(f"[GFM] missing keys ({len(missing)}): {missing[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            print(
                f"[GFM] unexpected keys ({len(unexpected)}): "
                f"{unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
            )
