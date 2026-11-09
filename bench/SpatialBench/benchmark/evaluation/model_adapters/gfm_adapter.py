"""SpatialBench adapters backed by the bundled GFM models.

RGB-only models never receive ground-truth depth or cameras. The explicit
GFM3D camera-conditioned variant receives calibrated cameras, as in the prompt
reference protocol. All reported poses come from the model camera head.
"""
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

# Run from the bundled tree without installing a second model implementation.
for parent in Path(__file__).resolve().parents:
    if (parent / 'gfm/models/gfm3d.py').is_file():
        sys.path.insert(0, str(parent))
        break

from gfm.models import GFM, GFM4D
from benchmark.evaluation.model_adapters import register_adapter
from benchmark.evaluation.model_adapters.base_adapter import ModelAdapter


class GFMAdapter(ModelAdapter):
    model_kind = 'gfm3d'

    def __init__(self):
        self.model = None
        self.device = 'cuda'
        self.model_size = 'vitg'
        self.model_name = None
        self.process_res = 504
        self.patch_size = 14
        self.num_register_tokens = 4
        self.use_camera_prompt = False
        self.enable_metric = False
        self.metric_type = 'frame'
        self.resize_input = False
        self.scale_value = 1.0
        self.camera_scale_mode = 'max'
        self.scale_min = None
        self.use_amp = True
        self.amp_dtype = 'float16'
        self.chunk_size = 30000
        self.enable_warp2d_visibility = False

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.__dict__ or key in ('model', 'device'):
                raise ValueError(f'Unsupported GFM parameter: {key}')
            setattr(self, key, value)
        if self.process_res < self.patch_size or self.patch_size < 1:
            raise ValueError('process_res must be >= patch_size > 0')
        if self.camera_scale_mode not in ('max', 'mean'):
            raise ValueError('camera_scale_mode must be max or mean')
        if not np.isfinite(self.scale_value) or self.scale_value <= 0:
            raise ValueError('scale_value must be finite and positive')
        if self.scale_min is not None and (not np.isfinite(self.scale_min) or self.scale_min <= 0):
            raise ValueError('scale_min must be finite and positive')
        if not self.use_amp or self.amp_dtype != 'float16':
            raise ValueError('The reference GFM models use CUDA float16 autocast')
        if self.model_kind == 'gfm4d' and (self.use_camera_prompt or self.enable_metric or self.model_size != 'vitg'):
            raise ValueError('GFM4D uses ViT-g without camera prompts or a metric head')

    def name(self):
        return self.model_name or (f'{self.model_kind}_camera_cond' if self.use_camera_prompt else self.model_kind)

    def load_model(self, checkpoint, device='cuda'):
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError(f'GFM checkpoint not found: {checkpoint}')
        self.device = torch.device(device)
        if self.model_kind == 'gfm4d':
            self.model = GFM4D(patch_size=self.patch_size, chunk_size=self.chunk_size,
                                   num_register_tokens=self.num_register_tokens,
                                   enable_warp2d_visibility=self.enable_warp2d_visibility)
        else:
            self.model = GFM(model_size=self.model_size, patch_size=self.patch_size,
                             num_register_tokens=self.num_register_tokens,
                             use_camera_prompt=self.use_camera_prompt,
                             enable_metric=self.enable_metric, metric_type=self.metric_type)
        # Fail on any missing/unexpected key or mismatched tensor shape.
        self.model.load_checkpoint(checkpoint, strict=True)
        self.model = self.model.eval().to(self.device)

    @staticmethod
    def _relative_w2c(gt_c2w):
        poses = np.asarray(gt_c2w, dtype=np.float32)
        if poses.ndim != 3 or poses.shape[1:] not in ((3, 4), (4, 4)):
            raise ValueError('Expected c2w cameras [N,3,4] or [N,4,4]')
        c2w = np.broadcast_to(np.eye(4, dtype=np.float32), (len(poses), 4, 4)).copy()
        c2w[:, :3] = poses[:, :3]
        return (np.linalg.inv(c2w) @ c2w[0:1]).astype(np.float32)

    def _camera_prompt_inputs(self, scene, out_h, out_w):
        relative_w2c = self._relative_w2c(scene['extrinsic'])
        intrinsics = np.asarray(scene['intrinsic'], dtype=np.float32).copy()
        if intrinsics.shape != (len(relative_w2c), 3, 3):
            raise ValueError('Expected one [3,3] intrinsic matrix per camera')
        input_h, input_w = scene['images_raw'].shape[-2:]
        intrinsics[:, 0] *= out_w / input_w
        intrinsics[:, 1] *= out_h / input_h
        distances = np.linalg.norm(np.linalg.inv(relative_w2c)[:, :3, 3], axis=-1)
        if self.camera_scale_mode == 'max':
            scale = float(distances.max())
        else:
            nonzero = distances[distances > 0]
            scale = float(nonzero.mean()) if len(nonzero) else 0.
        scale = max(scale, 1e-8 if self.scale_min is None else float(self.scale_min))
        # The standalone GFM API expects already scale-normalized translations.
        relative_w2c[:, :3, 3] /= scale
        return relative_w2c, intrinsics, scale

    @torch.inference_mode()
    def predict(self, scene):
        images = scene['images_raw']  # [N,3,H,W], RGB in [0,1]; normalized inside GFM.
        n, _, h, w = images.shape
        out_h, out_w = h, w
        if self.resize_input:
            scale = self.process_res / float(max(h, w))
            out_h = max(self.patch_size, int(h * scale) // self.patch_size * self.patch_size)
            out_w = max(self.patch_size, int(w * scale) // self.patch_size * self.patch_size)
            images = torch.stack([
                torch.from_numpy(cv2.resize(frame.permute(1, 2, 0).cpu().numpy(),
                    (out_w, out_h), interpolation=cv2.INTER_LINEAR)).permute(2, 0, 1)
                for frame in images
            ])
        elif h % self.patch_size or w % self.patch_size:
            raise ValueError(f'Reference GFM3D input must be patch-aligned, got {(h,w)}; '
                             'use resolution_override in the benchmark config')
        kwargs = {}
        output_scale = float(self.scale_value)
        if self.use_camera_prompt:
            extrinsic, intrinsic, output_scale = self._camera_prompt_inputs(scene, out_h, out_w)
            kwargs = dict(prompt_extrinsics=torch.from_numpy(extrinsic).to(self.device),
                          prompt_intrinsics=torch.from_numpy(intrinsic).to(self.device))
        prediction = self.model(images.to(self.device), **kwargs)
        depth = prediction['depth'][0, :, 0].float().cpu().numpy() * output_scale
        confidence = prediction['confidence'][0, :, 0].float().cpu().numpy()
        intrinsic = prediction['intrinsic'][0].float().cpu().numpy().copy()
        w2c = prediction['extrinsic'][0].float().cpu().numpy().copy()
        w2c[:, :3, 3] *= output_scale
        if (out_h, out_w) != (h, w):
            depth = np.stack([cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR) for d in depth])
            confidence = np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in confidence])
            # Apply the half-pixel inverse resize to the predicted intrinsics.
            sx, sy = out_w / float(w), out_h / float(h)
            intrinsic[:, 0, 0] /= sx
            intrinsic[:, 1, 1] /= sy
            intrinsic[:, 0, 2] = (intrinsic[:, 0, 2] + .5) / sx - .5
            intrinsic[:, 1, 2] = (intrinsic[:, 1, 2] + .5) / sy - .5
        return dict(pred_depth=depth.astype(np.float32), pred_confidence=confidence.astype(np.float32),
                    pred_intrinsic=intrinsic, w2c_extrinsics=w2c,
                    pred_pose=self._invert_se3(w2c))

    def normalize_gt_poses(self, scene):
        if self.use_camera_prompt:
            return self._relative_w2c(scene['extrinsic'])[:, :3]
        return self.normalize_camera_extrinsics_and_points(
            scene['extrinsic'], scene['world_points'], scene['valid_mask'])


@register_adapter('gfm3d')
class GFM3DAdapter(GFMAdapter):
    pass


@register_adapter('gfm4d')
class GFM4DAdapter(GFMAdapter):
    model_kind = 'gfm4d'

    def __init__(self):
        super().__init__()
        self.resize_input = True
        self.enable_warp2d_visibility = True
