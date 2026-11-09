import numpy as np
import torch


def camera_w2c_to_opencv(extrinsic: np.ndarray, camera_axes: str = "opencv") -> np.ndarray:
    """Convert a W2C matrix from a source camera-axis convention to OpenCV.

    The pose's world frame and camera centre are retained.  Only the camera
    basis changes.  ``flu`` is the robotics body-camera basis ``(forward,
    left, up)``; OpenCV uses ``(right, down, forward)``.
    """
    if camera_axes == "opencv":
        return extrinsic
    if camera_axes != "flu":
        raise ValueError(f"Unsupported camera axis convention: {camera_axes!r}")
    flu_to_opencv = np.array(
        [[0.0, -1.0, 0.0, 0.0],
         [0.0, 0.0, -1.0, 0.0],
         [1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.asarray(extrinsic).dtype,
    )
    return flu_to_opencv @ extrinsic


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """Unproject depth maps to world points.

    Args:
        depth_map: [S, H, W] or [S, H, W, 1]
        extrinsic: [S, 3, 4] or [S, 4, 4] world-to-camera
        intrinsic: [S, 3, 3]
    Returns:
        world_points: [S, H, W, 3]
    """
    if depth_map.ndim == 4:
        depth = depth_map[..., 0]
    else:
        depth = depth_map

    num_frames, height, width = depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )

def normalize_cameras_to_first(extrinsics: torch.Tensor) -> torch.Tensor:
    """Make the first camera the identity reference. extrinsics: [B,S,4,4] w2c."""
    w2c = extrinsics
    base_c2w = torch.linalg.inv(w2c[:, 0:1])
    return w2c @ base_c2w
