"""Image loading / preprocessing matched to GFM training transforms."""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile


@dataclass(frozen=True)
class ImageTransform:
    """Pixel transform applied while preparing one image for inference."""

    original_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    pad_left: int = 0
    pad_top: int = 0


def load_and_preprocess_images(
    image_path_list: Sequence[str],
    max_size: int = 504,
    patch_size: int = 14,
    num_workers: int = 16,
    return_transforms: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[ImageTransform]]:
    """Load images and preprocess for GFM inference.

    Training used ResizeKeepRatio(max_size=504, patch_size=14, low_resolution=True);
    outputs are float tensors in [0, 1] with shape [N, 3, H, W], H/W divisible by
    patch_size. The encoder applies ImageNet normalization internally.

    Images use OpenCV RGB decoding when available (with a truncated-image
    PIL fallback), float conversion in ``[0, 1]``, then Torch
    bilinear resize with ``align_corners=False``. Loading/resizing runs in a
    thread pool while preserving input order.
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    if max_size % patch_size != 0:
        raise ValueError("max_size must be divisible by patch_size")

    def process_one(image_path: str) -> tuple[torch.Tensor, ImageTransform]:
        image = _load_rgb_tensor(image_path)
        height, width = image.shape[-2:]

        target_h, target_w = _keep_ratio_target_shape(
            height, width, max_size, patch_size
        )

        image = F.interpolate(
            image.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return image, ImageTransform(
            original_hw=(height, width),
            resized_hw=(target_h, target_w),
        )

    workers = min(max(1, int(num_workers)), len(image_path_list))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            processed = list(executor.map(process_one, image_path_list))
    else:
        processed = [process_one(p) for p in image_path_list]

    images = [item[0] for item in processed]
    image_transforms = [item[1] for item in processed]

    shapes = {(tensor.shape[1], tensor.shape[2]) for tensor in images}

    if len(shapes) > 1:
        warnings.warn(
            f"Found images with different shapes: {shapes}; padding to a common size.",
            stacklevel=2,
        )
        images, padding = _pad_images_to_common_size(
            images, shapes, pad_value=1.0, return_padding=True
        )
        image_transforms = [
            ImageTransform(
                original_hw=transform.original_hw,
                resized_hw=transform.resized_hw,
                pad_left=pad[0],
                pad_top=pad[1],
            )
            for transform, pad in zip(image_transforms, padding)
        ]

    stacked = torch.stack(images)
    return (stacked, image_transforms) if return_transforms else stacked


def _load_rgb_tensor(image_path: str) -> torch.Tensor:
    """Decode an image and return RGB ``(3,H,W)`` in ``[0,1]``."""
    rgb: np.ndarray | None = None
    try:
        import cv2

        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:  # noqa: BLE001 - decoding failures fall back to PIL.
        pass

    if rgb is None:
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous

    return (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .permute(2, 0, 1)
        .contiguous()
        .float()
        .mul_(1.0 / 255.0)
    )


def _keep_ratio_target_shape(
    height: int,
    width: int,
    max_size: int,
    patch_size: int,
) -> tuple[int, int]:
    # Match ResizeKeepRatio(low_resolution=True): do not upscale small images.
    if width < max_size and height < max_size:
        new_width = int(width / patch_size) * patch_size
        new_height = int(height / patch_size) * patch_size
        new_width = max(patch_size, new_width)
        new_height = max(patch_size, new_height)
        return new_height, new_width

    if width >= height:
        new_width = max_size
        new_height = int(round(height * (new_width / width) / patch_size)) * patch_size
    else:
        new_height = max_size
        new_width = int(round(width * (new_height / height) / patch_size)) * patch_size

    new_width = max(patch_size, new_width)
    new_height = max(patch_size, new_height)
    return new_height, new_width


def _pad_images_to_common_size(
    images,
    shapes,
    pad_value: float = 0.0,
    return_padding: bool = False,
):
    max_height = max(shape[0] for shape in shapes)
    max_width = max(shape[1] for shape in shapes)

    padded = []
    padding = []
    for image in images:
        h_padding = max_height - image.shape[1]
        w_padding = max_width - image.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            image = torch.nn.functional.pad(
                image,
                (pad_left, pad_right, pad_top, pad_bottom),
                value=pad_value,
            )
        padded.append(image)
        padding.append(
            (pad_left if w_padding > 0 else 0, pad_top if h_padding > 0 else 0)
        )
    return (padded, padding) if return_padding else padded
