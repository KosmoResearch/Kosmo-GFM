#!/usr/bin/env python3
"""CLI inference for Camera Condition GFM with calibrated camera inputs."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running without pip install -e .
GFM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GFM_ROOT))

from gfm.models import GFM
from gfm.utils.geometry import camera_w2c_to_opencv, unproject_depth_map_to_point_map
from gfm.utils.load_fn import ImageTransform, load_and_preprocess_images




def parse_max_points(value: str) -> int | None:
    """Parse a non-negative point cap; 'none' disables output subsampling."""
    if value.strip().lower() in {"none", "null", "unlimited", "inf"}:
        return None
    try:
        max_points = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a non-negative integer or 'none'"
        ) from exc
    if max_points < 0:
        raise argparse.ArgumentTypeError("must be non-negative or 'none'")
    return max_points


def parse_args():
    parser = argparse.ArgumentParser(description="Camera Condition GFM standalone inference")
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
    parser.add_argument("--num-register-tokens", type=int, default=4)
    parser.add_argument(
        "--enable-metric", action=argparse.BooleanOptionalAction, default=False,
        help="Enable GFM 1.1 metric prediction and depth scale recovery.",
    )
    parser.add_argument("--metric-type", choices=("frame", "global"), default="frame",
                        help="Metric decoder scope; must match the training checkpoint.")
    parser.add_argument("--camera-npz", required=True,
                        help="NPZ with known extrinsics and intrinsics in input-image order")
    parser.add_argument(
        "--camera-convention",
        choices=["w2c", "c2w"],
        default="w2c",
        help="Convention of the extrinsics stored in --camera-npz",
    )
    parser.add_argument(
        "--camera-axis-convention",
        choices=["opencv", "flu"],
        default="opencv",
        help=(
            "Camera axes used by --camera-npz. 'flu' means x-forward, "
            "y-left, z-up and is converted to OpenCV before it reaches GFM."
        ),
    )
    parser.add_argument(
        "--camera-scale",
        type=float,
        default=None,
        help=(
            "Scene scale used in training. Overrides scale or "
            "sparse_pointmap_max_range in the camera NPZ."
        ),
    )
    camera_frame = parser.add_mutually_exclusive_group()
    camera_frame.add_argument(
        "--keep-camera-world-frame",
        dest="keep_camera_world_frame",
        action="store_true",
        default=True,
        help="Keep the supplied prompt cameras in their source world frame (default)",
    )
    camera_frame.add_argument(
        "--normalize-camera-world-frame",
        dest="keep_camera_world_frame",
        action="store_false",
        help="Use the first prompt camera as the local-world origin",
    )
    parser.add_argument(
        "--projection-camera",
        choices=["auto", "prompt", "model"],
        default="auto",
        help=(
            "Camera used to unproject depth for PLY/GLB export. 'auto' uses the "
            "known prompt camera when supplied, otherwise the model camera head "
            "(default: auto)."
        ),
    )
    parser.add_argument("--conf-thres", type=float, default=20.0)
    parser.add_argument(
        "--max-points",
        type=parse_max_points,
        default=300000,
        help=(
            "Maximum points written to scene.ply / scene.glb after filtering "
            "(default: 300000). Use 'none' for no output point limit."
        ),
    )
    parser.add_argument(
        "--exclude-mask-dir",
        type=str,
        default=None,
        help=(
            "Directory containing masks named like the input images. Non-zero "
            "pixels are excluded from exported points (confidence is set to NaN). "
            "Masks must match the original image dimensions."
        ),
    )
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


def load_prompt_model(
    checkpoint: str | Path,
    *,
    model_size: str = "vitg",
    patch_size: int = 14,
    num_register_tokens: int = 4,
    enable_metric: bool = False,
    metric_type: str = "frame",
    device: str | torch.device = "cuda",
) -> GFM:
    """Load the camera-prompt model for CLI or pipeline reuse."""
    model = GFM(
        model_size=model_size,
        patch_size=patch_size,
        num_register_tokens=num_register_tokens,
        use_camera_prompt=True,
        enable_metric=enable_metric,
        metric_type=metric_type,
    ).eval()
    model.load_checkpoint(str(checkpoint), strict=True)
    return model.to(torch.device(device))


@torch.inference_mode()
def infer_prompt_group(
    model: GFM,
    images: torch.Tensor,
    prompt_extrinsics: torch.Tensor,
    prompt_intrinsics: torch.Tensor,
) -> dict:
    """Run one prompt inference group and return detached NumPy outputs."""
    device = next(model.parameters()).device
    predictions = model(
        images.to(device, non_blocking=True),
        prompt_extrinsics=prompt_extrinsics.to(device, non_blocking=True),
        prompt_intrinsics=prompt_intrinsics.to(device, non_blocking=True),
    )
    return tensor_to_numpy(predictions)


def _first_npz_array(data, names):
    for name in names:
        if name in data:
            return np.asarray(data[name])
    raise KeyError(f"Camera NPZ is missing one of these keys: {', '.join(names)}")


def _reorder_cameras(arrays, data, image_paths, input_image_count=None):
    if "image_names" not in data:
        # Unnamed cameras follow the complete sorted image list. Select the
        # same prefix when --max-frames truncates that list. Already selected
        # camera arrays remain supported; unrelated counts fail validation.
        if input_image_count is not None and all(
            len(array) == input_image_count for array in arrays
        ):
            return [array[:len(image_paths)] for array in arrays]
        return arrays
    names = [
        name.decode("utf-8") if isinstance(name, bytes) else str(name)
        for name in np.asarray(data["image_names"]).reshape(-1)
    ]
    by_name = {}
    for index, name in enumerate(names):
        by_name[name] = index
        by_name[Path(name).name] = index
    order = []
    for path in image_paths:
        index = by_name.get(path)
        if index is None:
            index = by_name.get(Path(path).name)
        if index is None:
            raise ValueError(f"Image {path!r} is absent from camera NPZ image_names")
        order.append(index)
    return [array[order] for array in arrays]


def _pose_max_scale(extrinsics: np.ndarray) -> float:
    """Match training's ``with_pose_max_scale`` for OpenCV w2c cameras."""
    camera_centers = np.linalg.inv(extrinsics)[:, :3, 3]
    relative_centers = camera_centers - camera_centers[0:1]
    return max(float(np.linalg.norm(relative_centers, axis=-1).max()), 1e-8)


def _load_camera_prompt(args, image_paths, image_transforms, *, input_image_count=None):
    """Load ``--camera-npz`` and adapt cameras to the preprocessed images.

    Required NPZ fields (numeric arrays, preferably float32):
        extrinsics: [N,4,4] homogeneous poses or [N,3,4] matrices [R|t].
            Defaults to world-to-camera: X_camera = R @ X_world + t.
            Accepted keys, in lookup order: extrinsic, extrinsics, w2c, cameras.
        intrinsics: [N,3,3] pinhole K = [[fx,0,cx], [0,fy,cy], [0,0,1]],
            in pixels at each ORIGINAL image resolution. Do not pre-resize K;
            this function applies the image resize and padding transforms.
            Accepted keys: intrinsic, intrinsics, K.
        Both arrays also accept a leading singleton batch dimension [1,N,...].

    Optional NPZ fields:
        image_names: [N] string/bytes array (not dtype=object). Cameras are
            reordered by full path or basename to match the selected images.
            Use unique names. Without this field, cameras must follow the
            filename-sorted image order; --max-frames selects the same prefix.
        scale: Positive finite scene scalar; aliases sparse_pointmap_max_range
            and prompt_scale are also accepted. --camera-scale overrides it.
            If omitted, use the maximum camera-center distance from the first
            camera, clamped to at least 1e-8. Prompt translations are divided
            by this scale; main() restores scale for depth and output cameras.
        prompt_normalized: Boolean scalar, default False. Set True only when
            prompt translations have already been divided by the scene scale;
            supply that original scale explicitly to recover output units.
        projection_extrinsic: Optional [N,4,4] or [N,3,4] cameras for exporting
            geometry in a global world frame, in ORIGINAL scene units. These
            use the same image ordering and camera conventions as extrinsics,
            but are not scale-normalized or rebased to the first camera.
            They share the prompt intrinsics and are ignored for model-camera
            projection. Omit for ordinary camera-conditioned inference.

    Conventions:
        Default camera axes are OpenCV: X right, Y down, Z forward. For
        camera-to-world matrices, pass --camera-convention c2w (the NPZ key
        alone does not select the convention). For X forward, Y left, Z up,
        pass --camera-axis-convention flu. The input world frame is retained
        unless --normalize-camera-world-frame is supplied. NPZ loading uses
        allow_pickle=False, so do not store Python objects in any field.

    Example (two 640x480 images; replace with your calibrated cameras)::

        import numpy as np

        w2c = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
        w2c[1, 0, 3] = -0.2  # Second camera center is at world X = +0.2.
        K = np.repeat(np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                               dtype=np.float32)[None], 2, axis=0)
        np.savez("cameras.npz", extrinsics=w2c, intrinsics=K,
                 image_names=np.asarray(["000.png", "001.png"], dtype=str))

    Returns:
        Float32 CPU prompt tensors [N,4,4] and [N,3,3], the scene scale,
        and optional OpenCV world-to-camera projection cameras as a NumPy
        array [N,4,4] (or None).
    """
    with np.load(args.camera_npz, allow_pickle=False) as data:
        extrinsics = _first_npz_array(data, ("extrinsic", "extrinsics", "w2c", "cameras"))
        intrinsics = _first_npz_array(data, ("intrinsic", "intrinsics", "K"))
        projection_extrinsics = (
            np.asarray(data["projection_extrinsic"])
            if "projection_extrinsic" in data else None
        )
        prompt_normalized = bool(np.asarray(data["prompt_normalized"]).reshape(-1)[0]) \
            if "prompt_normalized" in data else False
        if extrinsics.ndim == 4 and extrinsics.shape[0] == 1:
            extrinsics = extrinsics[0]
        if intrinsics.ndim == 4 and intrinsics.shape[0] == 1:
            intrinsics = intrinsics[0]
        values = [extrinsics, intrinsics]
        if projection_extrinsics is not None:
            values.append(projection_extrinsics)
        values = _reorder_cameras(values, data, image_paths, input_image_count)
        extrinsics, intrinsics = values[:2]
        projection_extrinsics = values[2] if len(values) > 2 else None

        scale = args.camera_scale
        if scale is None:
            for key in ("scale", "sparse_pointmap_max_range", "prompt_scale"):
                if key in data:
                    stored_scale = np.asarray(data[key]).reshape(-1)
                    if stored_scale.size != 1:
                        raise ValueError(f"Camera NPZ {key!r} must be a scene scalar")
                    scale = float(stored_scale[0])
                    break

    num_images = len(image_paths)
    if extrinsics.shape[0] != num_images or intrinsics.shape[0] != num_images:
        raise ValueError(
            "Camera count must match selected images: "
            f"images={num_images}, extrinsics={extrinsics.shape[0]}, "
            f"intrinsics={intrinsics.shape[0]}"
        )
    if extrinsics.shape[-2:] == (3, 4):
        homogeneous = np.broadcast_to(
            np.eye(4, dtype=extrinsics.dtype), (num_images, 4, 4)
        ).copy()
        homogeneous[:, :3] = extrinsics
        extrinsics = homogeneous
    if projection_extrinsics is not None and projection_extrinsics.shape[-2:] == (3, 4):
        homogeneous = np.broadcast_to(
            np.eye(4, dtype=projection_extrinsics.dtype), (num_images, 4, 4)
        ).copy()
        homogeneous[:, :3] = projection_extrinsics
        projection_extrinsics = homogeneous
    if extrinsics.shape[-2:] != (4, 4) or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected extrinsics [N,3/4,4] and intrinsics [N,3,3], got "
            f"{extrinsics.shape} and {intrinsics.shape}"
        )
    extrinsics = extrinsics.astype(np.float32, copy=True)
    if projection_extrinsics is not None:
        if projection_extrinsics.shape != extrinsics.shape:
            raise ValueError("projection_extrinsic shape must match prompt extrinsic")
        projection_extrinsics = projection_extrinsics.astype(np.float32, copy=True)
    intrinsics = intrinsics.astype(np.float32, copy=True)
    if args.camera_convention == "c2w":
        extrinsics = np.linalg.inv(extrinsics).astype(np.float32)
        if projection_extrinsics is not None:
            projection_extrinsics = np.linalg.inv(projection_extrinsics).astype(np.float32)
    extrinsics = camera_w2c_to_opencv(extrinsics, args.camera_axis_convention)
    if projection_extrinsics is not None:
        projection_extrinsics = camera_w2c_to_opencv(
            projection_extrinsics, args.camera_axis_convention
        )
    if not args.keep_camera_world_frame:
        extrinsics = extrinsics @ np.linalg.inv(extrinsics[0:1])

    if scale is None:
        scale = _pose_max_scale(extrinsics)
        print(
            "Camera NPZ has no scene scale; computed the training-compatible "
            f"pose max scale: {scale:.8g}"
        )
    else:
        scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"camera scale must be finite and positive, got {scale}")
    if not prompt_normalized:
        extrinsics[:, :3, 3] /= scale

    for index, transform in enumerate(image_transforms):
        orig_h, orig_w = transform.original_hw
        resized_h, resized_w = transform.resized_hw
        intrinsics[index, 0, :] *= resized_w / orig_w
        intrinsics[index, 1, :] *= resized_h / orig_h
        intrinsics[index, 0, 2] += transform.pad_left
        intrinsics[index, 1, 2] += transform.pad_top

    return (
        torch.from_numpy(extrinsics), torch.from_numpy(intrinsics), scale,
        projection_extrinsics,
    )


def apply_exclusion_masks(
    confidence: np.ndarray, image_paths, mask_dir: str, *,
    image_transforms: list[ImageTransform],
) -> np.ndarray:
    """Resize and pad original-resolution masks exactly like their RGB images."""
    from PIL import Image

    root = Path(mask_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Exclude mask directory not found: {root}")
    conf = confidence.copy()
    if conf.ndim == 4 and conf.shape[-1] == 1:
        height, width = conf.shape[1:3]
        conf_view = conf[..., 0]
    elif conf.ndim == 3:
        height, width = conf.shape[1:3]
        conf_view = conf
    else:
        raise ValueError(f"Unsupported confidence shape for exclusion masks: {conf.shape}")
    if len(image_paths) != len(conf_view) or len(image_transforms) != len(image_paths):
        raise ValueError("Expected one image and preprocessing transform per confidence map")
    excluded = 0
    for index, (image_path, transform) in enumerate(zip(image_paths, image_transforms)):
        stem = Path(image_path).stem
        mask_path = next(
            (root / f"{stem}{extension}" for extension in (".png", ".PNG", ".jpg", ".JPG") if (root / f"{stem}{extension}").is_file()),
            None,
        )
        if mask_path is None:
            raise FileNotFoundError(f"No exclude mask for {Path(image_path).name} under {root}")
        resized_h, resized_w = transform.resized_hw
        left, top = transform.pad_left, transform.pad_top
        if (min(resized_h, resized_w) < 1 or min(left, top) < 0
                or top + resized_h > height or left + resized_w > width):
            raise ValueError(f"Mask transform does not fit the confidence canvas: {transform}")
        with Image.open(mask_path) as image:
            if image.size != transform.original_hw[::-1]:
                raise ValueError(f"Mask {mask_path.name} must match original image size "
                                 f"{transform.original_hw[::-1]}, got {image.size}")
            resized = np.asarray(image.convert("L").resize(
                (resized_w, resized_h), Image.Resampling.NEAREST))
        mask = np.zeros((height, width), dtype=bool)
        mask[top:top + resized_h, left:left + resized_w] = resized != 0
        excluded += int(np.count_nonzero(mask))
        conf_view[index][mask != 0] = np.nan
    print(f"Applied exclusion masks from {root}: excluded_pixels={excluded}")
    return conf


def _scale_w2c_translation(extrinsic: np.ndarray, scale: float) -> np.ndarray:
    """Return W2C extrinsics expressed in the same length unit as depth."""
    extrinsic = np.asarray(extrinsic, dtype=np.float32).copy()
    if scale != 1.0:
        extrinsic[:, :3, 3] *= scale
    return extrinsic


@torch.inference_mode()
def main():
    args = parse_args()
    image_paths = collect_images(args.input)
    if not image_paths:
        raise RuntimeError(f"No images found under {args.input}")

    input_image_count = len(image_paths)
    if args.max_frames is not None and args.max_frames > 0 and len(image_paths) > args.max_frames:
        print(f"Truncating {len(image_paths)} -> {args.max_frames} frames (--max-frames)")
        image_paths = image_paths[: args.max_frames]

    os.makedirs(args.output, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU inference is not supported")
    device = torch.device("cuda")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_prompt_model(
        args.checkpoint,
        model_size=args.model_size,
        patch_size=args.patch_size,
        num_register_tokens=args.num_register_tokens,
        enable_metric=args.enable_metric,
        metric_type=args.metric_type,
        device=device,
    )

    print(f"Loading {len(image_paths)} images ...")
    images, image_transforms = load_and_preprocess_images(
        image_paths,
        max_size=args.max_size,
        patch_size=args.patch_size,
        return_transforms=True,
    )
    prompt_extrinsics, prompt_intrinsics, output_scale, projection_extrinsics = _load_camera_prompt(
        args, image_paths, image_transforms, input_image_count=input_image_count
    )
    images = images.to(device)
    if prompt_extrinsics is not None:
        prompt_extrinsics = prompt_extrinsics.to(device)
        prompt_intrinsics = prompt_intrinsics.to(device)
    print(f"Preprocessed images: {tuple(images.shape)}")

    predictions = model(
        images,
        prompt_extrinsics=prompt_extrinsics,
        prompt_intrinsics=prompt_intrinsics,
    )

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


    if args.exclude_mask_dir:
        conf = apply_exclusion_masks(
            conf, image_paths, args.exclude_mask_dir, image_transforms=image_transforms,
        )

    model_extrinsic = np.asarray(pred_np["extrinsic"], dtype=np.float32)
    model_intrinsic = np.asarray(pred_np["intrinsic"], dtype=np.float32)
    prompt_extrinsic = None
    prompt_intrinsic = None
    if prompt_extrinsics is not None:
        prompt_extrinsic = prompt_extrinsics.detach().float().cpu().numpy()
        prompt_intrinsic = prompt_intrinsics.detach().float().cpu().numpy()

    projection_in_global_world = projection_extrinsics is not None and args.projection_camera != "model"
    if args.projection_camera == "model" or prompt_extrinsic is None:
        if args.projection_camera == "prompt" and prompt_extrinsic is None:
            raise ValueError("--projection-camera prompt requires --camera-npz")
        projection_camera = "model"
        extrinsic = model_extrinsic
        intrinsic = model_intrinsic
    else:
        # A known-camera prompt defines the common frame across independently run
        # chunks. The camera head is still saved for pose-quality diagnostics.
        projection_camera = "global_prompt" if projection_in_global_world else "prompt"
        extrinsic = projection_extrinsics if projection_in_global_world else prompt_extrinsic
        intrinsic = prompt_intrinsic

    if output_scale != 1.0:
        depth_for_pts = depth_for_pts * output_scale
    model_extrinsic = _scale_w2c_translation(model_extrinsic, output_scale)
    if not projection_in_global_world:
        extrinsic = _scale_w2c_translation(extrinsic, output_scale)
    world_points = unproject_depth_map_to_point_map(depth_for_pts, extrinsic, intrinsic)

    pred_np["depth"] = depth_for_pts
    pred_np["confidence"] = conf
    pred_np["extrinsic"] = extrinsic
    pred_np["intrinsic"] = intrinsic
    pred_np["model_extrinsic"] = model_extrinsic
    pred_np["model_intrinsic"] = model_intrinsic
    pred_np["world_points_from_depth"] = world_points
    print(
        f"Export projection uses {projection_camera} camera "
        f"({'global prompt frame' if args.keep_camera_world_frame else 'first-camera local frame'})."
    )

    if not args.no_ply or not args.no_glb:
        try:
            from gfm.utils.export import (
                cameras_to_ply,
                predictions_to_glb,
                predictions_to_ply,
            )
        except ImportError as exc:
            raise RuntimeError(
                "PLY/GLB export dependencies are unavailable; install "
                "requirements-optional.txt or pass --no-ply --no-glb"
            ) from exc

    if args.save_npz:
        npz_path = os.path.join(args.output, "predictions.npz")
        npz_payload = dict(
            depth=pred_np["depth"],
            confidence=pred_np["confidence"],
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            model_extrinsic=model_extrinsic,
            model_intrinsic=model_intrinsic,
            projection_camera=np.asarray(projection_camera),
            pose_enc=pred_np.get("pose_enc"),
            world_points_from_depth=world_points,
            images=pred_np["images"],
            scale=np.float32(output_scale),
        )
        for key in ("global_points", "global_confidence", "metric", "log_metric"):
            if pred_np.get(key) is not None:
                npz_payload[key] = pred_np[key]
        if prompt_extrinsic is not None:
            npz_payload["prompt_extrinsic"] = _scale_w2c_translation(
                prompt_extrinsic, output_scale
            )
            npz_payload["prompt_intrinsic"] = prompt_intrinsic
        np.savez_compressed(npz_path, **npz_payload)
        print(f"Saved {npz_path}")

    if not args.no_ply:
        ply_path = os.path.join(args.output, "scene.ply")
        predictions_to_ply(
            pred_np,
            ply_path,
            conf_thres=args.conf_thres,
            conf_key="confidence",
            points_key="world_points_from_depth",
            max_points=args.max_points,
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
            max_points=args.max_points,
            align_to_first_camera=not args.keep_camera_world_frame,
        )
        scene.export(glb_path)
        print(f"Saved {glb_path}")

    print("Done.")
    print(f"  frames={len(image_paths)} depth={depth_for_pts.shape} "
          f"extrinsic={extrinsic.shape} intrinsic={intrinsic.shape}")



if __name__ == "__main__":
    main()
