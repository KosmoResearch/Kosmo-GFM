#!/usr/bin/env python3
"""CLI inference for GFM (standalone, no training code)."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running without pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gfm.models import GFM
from gfm.utils.geometry import unproject_depth_map_to_point_map
from gfm.utils.load_fn import load_and_preprocess_images

def parse_args():
    parser = argparse.ArgumentParser(description="GFM standalone inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint dir (ckpt.pth) or state_dict path",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Image file, directory of images, or glob pattern",
    )
    parser.add_argument("--output", type=str, default="outputs/gfm_infer")
    parser.add_argument(
        "--model-size",
        type=str,
        default="vitg",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Backbone size (must match the checkpoint)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="If set, keep at most this many images (truncates the sorted list)",
    )
    parser.add_argument("--max-size", type=int, default=504)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--conf-thres", type=float, default=20.0)
    parser.add_argument("--save-npz", action="store_true", help="Save predictions.npz")
    parser.add_argument("--no-glb", action="store_true", help="Skip writing scene.glb")
    parser.add_argument("--no-ply", action="store_true", help="Skip writing scene.ply")
    return parser.parse_args()


def collect_images(input_path: str):
    if os.path.isdir(input_path):
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.PNG"):
            paths.extend(glob.glob(os.path.join(input_path, ext)))
        return sorted(paths)
    if any(ch in input_path for ch in "*?[]"):
        return sorted(glob.glob(input_path))
    if os.path.isfile(input_path):
        return [input_path]
    raise FileNotFoundError(f"Input not found: {input_path}")


def tensor_to_numpy(predictions: dict) -> dict:
    out = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            out[key] = value
        else:
            out[key] = value
    return out


@torch.inference_mode()
def main():
    args = parse_args()
    image_paths = collect_images(args.input)
    if not image_paths:
        raise RuntimeError(f"No images found under {args.input}")

    if args.max_frames is not None and args.max_frames > 0 and len(image_paths) > args.max_frames:
        print(f"Truncating {len(image_paths)} -> {args.max_frames} frames (--max-frames)")
        image_paths = image_paths[: args.max_frames]

    os.makedirs(args.output, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU inference is not supported")
    device = torch.device("cuda")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = GFM(model_size=args.model_size, patch_size=args.patch_size).eval()
    model.load_checkpoint(args.checkpoint, strict=True)
    model = model.to(device)

    print(f"Loading {len(image_paths)} images ...")
    images = load_and_preprocess_images(
        image_paths,
        max_size=args.max_size,
        patch_size=args.patch_size,
    ).to(device)
    print(f"Preprocessed images: {tuple(images.shape)}")

    predictions = model(images)

    pred_np = tensor_to_numpy(predictions)

    # Exports use CPU arrays; release GPU resources after inference.
    del predictions, images, model
    torch.cuda.empty_cache()

    # depth: [S,1,H,W] -> [S,H,W,1] for unproject helper consistency
    depth = pred_np["depth"]
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth_for_pts = np.transpose(depth, (0, 2, 3, 1))
    else:
        depth_for_pts = depth[..., None] if depth.ndim == 3 else depth

    conf = pred_np["confidence"]
    if conf.ndim == 4 and conf.shape[1] == 1:
        conf = np.transpose(conf, (0, 2, 3, 1))


    extrinsic = pred_np["extrinsic"]
    intrinsic = pred_np["intrinsic"]
    world_points = unproject_depth_map_to_point_map(depth_for_pts, extrinsic, intrinsic)

    pred_np["depth"] = depth_for_pts
    pred_np["confidence"] = conf
    pred_np["world_points_from_depth"] = world_points

    if args.save_npz:
        npz_path = os.path.join(args.output, "predictions.npz")
        np.savez_compressed(
            npz_path, **{key: value for key, value in pred_np.items() if value is not None}
        )
        print(f"Saved {npz_path}")

    if not args.no_ply or not args.no_glb:
        from gfm.utils.export import cameras_to_ply, predictions_to_glb, predictions_to_ply

    if not args.no_ply:
        ply_path = os.path.join(args.output, "scene.ply")
        predictions_to_ply(
            pred_np,
            ply_path,
            conf_thres=args.conf_thres,
            conf_key="confidence",
            points_key="world_points_from_depth",
        )
        print(f"Saved {ply_path}")

        cameras_ply_path = os.path.join(args.output, "cameras.ply")
        cameras_to_ply(pred_np, cameras_ply_path)
        print(f"Saved {cameras_ply_path}")

    if not args.no_glb:
        glb_path = os.path.join(args.output, "scene.glb")
        scene = predictions_to_glb(
            pred_np,
            conf_thres=args.conf_thres,
            conf_key="confidence",
            points_key="world_points_from_depth",
        )
        scene.export(glb_path)
        print(f"Saved {glb_path}")

    print("Done.")
    print(f"  frames={len(image_paths)} depth={depth_for_pts.shape} "
          f"extrinsic={extrinsic.shape} intrinsic={intrinsic.shape}")



if __name__ == "__main__":
    main()
