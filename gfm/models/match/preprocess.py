"""Image preprocessing for MVLoMA detection and description."""
import numpy as np
import torch
from PIL import Image


def detector_shape(
    original_height: int,
    original_width: int,
    long_edge: int = 1024,
) -> tuple[int, int]:
    """Return MVLoMA's aspect-preserving, align-8 detector shape."""
    if original_height <= 0 or original_width <= 0:
        raise ValueError(
            f"invalid original image shape {(original_height, original_width)}"
        )
    if long_edge < 8:
        raise ValueError(f"detector long edge must be >= 8, got {long_edge}")
    scale = float(long_edge) / max(original_height, original_width)
    height = max(8, int((scale * original_height) // 8 * 8))
    width = max(8, int((scale * original_width) // 8 * 8))
    return height, width


def read_loma_inputs(
    image_path: str,
    device: torch.device,
    descriptor_size: int = 784,
    detector_long_edge: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Load dual-resolution MVLoMA frontend tensors.

    DaD detection uses an aspect-preserving long edge of 1024 aligned down to
    8 pixels. DeDoDe description uses a direct 784x784 resize. Both use PIL
    bicubic, matching the model preprocessing.

    Returns:
        ``(detector, descriptor, orig_h, orig_w)``.
    """
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
        orig_w, orig_h = image.size
        detect_h, detect_w = detector_shape(
            orig_h, orig_w, detector_long_edge
        )
        detector = image.resize(
            (detect_w, detect_h), resample=Image.Resampling.BICUBIC
        )
        descriptor = image.resize(
            (descriptor_size, descriptor_size),
            resample=Image.Resampling.BICUBIC,
        )

    def tensor(value: Image.Image) -> torch.Tensor:
        array = np.asarray(value, dtype=np.uint8).copy()
        return (
            torch.from_numpy(array)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            .mul_(1.0 / 255.0)
            .unsqueeze(0)
            .to(device)
        )

    return tensor(detector), tensor(descriptor), orig_h, orig_w
