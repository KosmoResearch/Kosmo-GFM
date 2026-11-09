#!/usr/bin/env python3
"""Run GFM with MVLoMA: images to depth, cameras, point clouds, and matches."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from gfm.models import GFM
from gfm.models.heads.mvloma_head import MVLoMAHead
from gfm.utils.load_fn import load_and_preprocess_images
from gfm.utils.geometry import unproject_depth_map_to_point_map
from gfm.models.match.loma.loma import filter_matches, to_pixel_coords
from gfm.models.match.preprocess import read_loma_inputs
from tools.infer_3d import collect_images, tensor_to_numpy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='GFM geometry checkpoint')
    parser.add_argument('--mvloma-checkpoint', required=True, help='MVLoMA head or full checkpoint')
    parser.add_argument('--mvloma-precision', choices=('bf16', 'fp16', 'auto'), default='bf16',
                        help='MVLoMA AMP precision (default: bf16); GFM geometry stays fp16')
    parser.add_argument('--input', required=True, help='Image directory or glob; first sorted view is the anchor')
    parser.add_argument('--output', default='outputs/gfm_with_mvloma')
    parser.add_argument('--conf-thres', type=float, default=20.0, help='Point-cloud confidence percentile')
    parser.add_argument('--no-ply', action='store_true', help='Skip scene.ply and cameras.ply')
    parser.add_argument('--no-glb', action='store_true', help='Skip scene.glb')
    parser.add_argument('--model-size', choices=('vits', 'vitb', 'vitl', 'vitg'), default='vitg')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--max-size', type=int, default=504)
    parser.add_argument('--patch-size', type=int, default=14)
    parser.add_argument('--descriptor-size', type=int, default=784)
    parser.add_argument('--detector-size', type=int, default=1024)
    parser.add_argument('--num-keypoints', type=int, default=2048)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--all-pairs', action='store_true', help='Export all view pairs using the same anchor context')
    args = parser.parse_args()
    if args.num_keypoints < 1 or args.detector_size < 8:
        parser.error('num-keypoints must be positive and detector-size must be >= 8')
    if args.descriptor_size < 16 or args.descriptor_size % 14:
        parser.error('descriptor-size must be >= 16 and divisible by 14')
    if not 0 <= args.threshold <= 1:
        parser.error('threshold must be in [0, 1]')
    return args


def export_geometry(predictions, paths, output, *, conf_thres=20.0,
                    save_ply=True, save_glb=True):
    """Export geometry from the same joint forward pass that produced matches."""
    geometry = tensor_to_numpy({
        key: value for key, value in predictions.items()
        if isinstance(value, torch.Tensor)
    })
    # Match the GFM 3D export layout: [N,H,W,1] for depth and confidence.
    for key in ('depth', 'confidence'):
        value = geometry[key]
        if value.ndim == 4 and value.shape[1] == 1:
            value = np.transpose(value, (0, 2, 3, 1))
        geometry[key] = value
    geometry['world_points_from_depth'] = unproject_depth_map_to_point_map(
        geometry['depth'], geometry['extrinsic'], geometry['intrinsic'])
    geometry['image_names'] = np.asarray([Path(path).name for path in paths])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'predictions.npz', **geometry)
    if save_ply or save_glb:
        from gfm.utils.export import cameras_to_ply, predictions_to_glb, predictions_to_ply
        options = dict(conf_thres=conf_thres, conf_key='confidence',
                       points_key='world_points_from_depth')
        if save_ply:
            predictions_to_ply(geometry, str(output / 'scene.ply'), **options)
            cameras_to_ply(geometry, str(output / 'cameras.ply'))
        if save_glb:
            predictions_to_glb(geometry, **options).export(str(output / 'scene.glb'))
    print(f'Saved {output / "predictions.npz"}: depth, cameras, and world points')


@torch.inference_mode()
def main():
    args = parse_args()
    paths = collect_images(args.input)
    if args.max_frames is not None:
        if args.max_frames < 2:
            raise ValueError('MVLoMA requires --max-frames >= 2')
        paths = paths[:args.max_frames]
    if len(paths) < 2:
        raise ValueError('MVLoMA requires at least two images')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')

    model = GFM(model_size=args.model_size, patch_size=args.patch_size)
    model.load_checkpoint(args.checkpoint, strict=True)
    head = MVLoMAHead(fuse_dim=model.fuse_encoder.pretrained.embed_dim * 2,
                      patch_size=args.patch_size, precision=args.mvloma_precision)
    head.load_checkpoint(args.mvloma_checkpoint)
    print(f"GFM AMP: fp16; MVLoMA AMP: {head.amp_dtype} ({args.mvloma_precision})")
    model.loma_match_head = head
    model = model.eval().cuda()

    images, transforms = load_and_preprocess_images(
        paths, max_size=args.max_size, patch_size=args.patch_size, return_transforms=True)
    keypoints, descriptors, pixels, shapes = [], [], [], []
    for path in paths:
        detector, descriptor, h, w = read_loma_inputs(
            path, 'cuda', descriptor_size=args.descriptor_size,
            detector_long_edge=args.detector_size)
        if args.num_keypoints > detector.shape[-2] * detector.shape[-1]:
            raise ValueError('num-keypoints exceeds the detector image pixel count')
        points = head.detect(detector, num_keypoints=args.num_keypoints)
        keypoints.append(points)
        descriptors.append(descriptor)
        pixels.append(to_pixel_coords(points[0], h, w).float().cpu().numpy())
        shapes.append((h, w))
    keypoints = torch.stack(keypoints, dim=1)
    descriptor_images = torch.stack(descriptors, dim=1)
    pairs = ([(a, b) for a in range(len(paths)) for b in range(a + 1, len(paths))]
             if args.all_pairs else [(0, b) for b in range(1, len(paths))])
    predictions = model(images.cuda(), mvloma_inputs=dict(
        keypoints=keypoints, descriptor_images=descriptor_images,
        transforms=transforms, pair_idx=pairs))
    result = predictions['loma_match']
    payload = dict(image_names=np.asarray([Path(p).name for p in paths]),
                   image_shapes=np.asarray(shapes, dtype=np.int64),
                   keypoints=np.stack(pixels), pair_idx=np.asarray(pairs, dtype=np.int64))
    for (a, b), scores in result['scores_by_pair'].items():
        matches, _, confidence, _ = filter_matches(scores, args.threshold)
        valid = matches[0] >= 0
        indices = torch.stack((torch.where(valid)[0], matches[0, valid]), dim=-1)
        payload[f'matches_{a}_{b}'] = indices.cpu().numpy()
        payload[f'scores_{a}_{b}'] = confidence[0, valid].float().cpu().numpy()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    export_geometry(predictions, paths, output, conf_thres=args.conf_thres,
                    save_ply=not args.no_ply, save_glb=not args.no_glb)
    np.savez_compressed(output / 'matches.npz', **payload)
    print(f'Saved {output / "matches.npz"}: {len(paths)} images, {len(pairs)} pairs')


if __name__ == '__main__':
    main()
