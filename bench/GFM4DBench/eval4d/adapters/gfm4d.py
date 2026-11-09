"""Sparse-query evaluation using the bundled GFM4D model.

Only RGB and the protocol's frame-0 query UVs enter the model. Ground-truth
positions, cameras, depths and scales are never passed to inference. Queries
are decoded directly by the pair aggregator, not sampled from dense outputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.sequences import load_mf_files, resolve_worldtrack_path
from .base import AdapterPrediction

DEFAULT_CONFIG = dict(process_res=504, patch_size=14, num_register_tokens=4,
                      enable_warp2d_visibility=True, chunk_size=2048)


def _require_gfm(feature: str) -> None:
    """Raise an actionable error when the Kosmo-GFM package is unavailable."""
    try:
        import gfm  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"The GFM4D adapter ({feature}) requires the Kosmo-GFM package. "
            "From the repository root run 'pip install -e .', then install the "
            "benchmark with 'pip install -e \"bench/GFM4DBench[gfm]\"'."
        ) from exc


class RGBSequenceDataset:
    """Read the first T frames, preserving the order of the sequence JSON."""

    def __init__(self, json_path, data_root, config, num_frames, sequence_names=None):
        self.json_path = Path(json_path)
        self.data_root = Path(data_root)
        self.sequences = load_mf_files(self.json_path)
        self.sequence_names = (sorted(self.sequences) if sequence_names is None
                               else list(sequence_names))
        self.config = config
        self.num_frames = int(num_frames)
        if self.num_frames < 1:
            raise ValueError('num_frames must be positive')

    def __len__(self):
        return len(self.sequence_names)

    def __getitem__(self, index):
        _require_gfm('frame loading')
        from gfm.utils.load_fn import load_and_preprocess_images

        name = self.sequence_names[index]
        frames = self.sequences[name][:self.num_frames]
        if not frames or any(len(frame.get('views', [])) != 1 for frame in frames):
            raise ValueError(f'{name}: expected one RGB view per video frame')
        paths = [str(resolve_worldtrack_path(frame['views'][0]['rgb'],
                     data_root=self.data_root, json_path=self.json_path)) for frame in frames]
        images, transforms = load_and_preprocess_images(
            paths, max_size=self.config['process_res'],
            patch_size=self.config['patch_size'], return_transforms=True)
        if len({t.original_hw for t in transforms}) != 1:
            raise ValueError(f'{name}: video frames must have the same original resolution')
        return dict(images=images, original_hw=transforms[0].original_hw,
                    sequence_name=name)

    @staticmethod
    def get_collate_fn():
        return _single_sequence


def _single_sequence(items):
    if len(items) != 1:
        raise ValueError('GFM4D evaluation uses one sequence per batch')
    return items[0]


class GFM4DAdapter:
    """GFM4D inference with independent position and flow heads."""

    name = 'gfm4d'

    def __init__(self, *, pred_3d_source='warp3d'):
        if pred_3d_source not in ('warp3d', 'warp3d_delta'):
            raise ValueError('pred_3d_source must be warp3d or warp3d_delta')
        self.pred_3d_source = pred_3d_source

    def load_config(self, config_path):
        import yaml

        values = {} if config_path is None else yaml.safe_load(Path(config_path).read_text())
        if not isinstance(values, dict):
            raise ValueError('Model configuration must be a YAML mapping')
        unknown = set(values) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f'Unknown GFM4D configuration fields: {sorted(unknown)}')
        config = dict(DEFAULT_CONFIG, **values)
        for key in ('process_res', 'patch_size', 'chunk_size'):
            if not isinstance(config[key], int) or config[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        if config['process_res'] % config['patch_size']:
            raise ValueError('process_res must be divisible by patch_size')
        if not isinstance(config['num_register_tokens'], int) or config['num_register_tokens'] < 0:
            raise ValueError('num_register_tokens must be nonnegative')
        if not isinstance(config['enable_warp2d_visibility'], bool):
            raise ValueError('enable_warp2d_visibility must be boolean')
        return config

    def load_model(self, config, checkpoint_path, device):
        import torch
        _require_gfm('live inference')
        from gfm.models import GFM4D

        device = torch.device(device)
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('GFM4D live inference requires CUDA; offline scoring runs on CPU')
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        model = GFM4D(**{key: config[key] for key in DEFAULT_CONFIG if key != 'process_res'})
        model.load_checkpoint(str(checkpoint_path), strict=True)
        return model.eval().to(device)

    def build_dataset(self, json_path, data_root, config, num_frames,
                      sequence_names=None, coord_convention='opend4rt',
                      dynamic_query_threshold=None):
        # Coordinate conventions and dynamic filters belong to the scorer;
        # the runner injects the exact ordered query UVs after loading the GT.
        return RGBSequenceDataset(json_path, data_root, dict(DEFAULT_CONFIG, **config),
                                  num_frames, sequence_names)

    def infer(self, model, batch, mixed_precision='fp16'):
        import torch

        if mixed_precision != 'fp16':
            raise ValueError('Bundled GFM4D uses FP16 model inference')
        images = batch['images']
        if images.shape[0] != batch['worldtrack_num_frames']:
            raise ValueError('RGB and evaluation clip lengths differ')
        device = next(model.parameters()).device
        uv = torch.as_tensor(batch['worldtrack_query_uv'], dtype=torch.float32, device=device).clone()
        height, width = batch['original_hw']
        # Normalize query UVs to [0, 1] with the benchmark convention.
        uv[:, 0] /= max(width - 1, 1)
        uv[:, 1] /= max(height - 1, 1)
        with torch.inference_mode():
            outputs = model.forward_queries(images.to(device), uv)
        return {key: value.detach().float().cpu().numpy() if torch.is_tensor(value) else value
                for key, value in outputs.items()}

    def extract(self, outputs, *, num_frames, num_queries):
        pairs = [tuple(map(int, pair)) for pair in outputs['pair_idx']]
        if len(pairs) != len(set(pairs)):
            raise ValueError('Duplicate output pairs')
        expected = (1, len(pairs), num_queries, 3)
        warp = np.asarray(outputs['warp3d'], dtype=np.float64)
        direction = np.asarray(outputs['warp3d_delta_direction'], dtype=np.float64)
        magnitude = np.asarray(outputs['warp3d_delta_magnitude'], dtype=np.float64)
        if warp.shape != expected or direction.shape != expected or magnitude.shape != (*expected[:-1], 1):
            raise ValueError(f'Expected sparse predictions {expected}; got {warp.shape}')
        if (0, 0) not in pairs:
            raise ValueError('Frame-0 self-pair is required for predicted source points')
        # Frame-0 source points: warp3d(0, 0).
        source = warp[0, pairs.index((0, 0))]
        tracks = np.empty((num_frames, num_queries, 3), dtype=np.float64)
        flow = np.zeros_like(tracks)
        tracks[0] = source
        # Decode the delta in FP32 after the AMP model heads.
        delta = (direction.astype(np.float32) * magnitude.astype(np.float32)).astype(np.float64)
        for frame in range(1, num_frames):
            if (0, frame) not in pairs:
                raise ValueError(f'Missing prediction pair (0, {frame})')
            index = pairs.index((0, frame))
            flow[frame] = delta[0, index]
            tracks[frame] = (warp[0, index] if self.pred_3d_source == 'warp3d'
                             else source + flow[frame])
        return AdapterPrediction(tracks_ref0=tracks, flow_ref0=flow)
