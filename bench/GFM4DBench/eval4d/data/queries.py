"""Frame-0 query selection rules.

The WorldTrack protocol selects, for every sequence, the set of tracks that are
*queried* in the first frame. The rules are fixed and define the benchmark:

1. the track must be **visible** in frame 0;
2. its depth in frame 0 must be finite and ``|z| > 1e-8``;
3. its pixel coordinates must be obtained by **projecting** ``trajs_3d`` with the
   camera intrinsics -- the on-disk ``trajs_2d`` is *not* used;
4. the projected pixel must lie **inside the image** (``0 <= u < W``, ``0 <= v < H``),
   where ``(W, H)`` is the real resolution of the frame-0 RGB.

The JSON ``valids`` array is deliberately **not** used for query selection.

This module contains pure array maths only; the JSON-level entry points live in
:mod:`eval4d.data.sequences`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .conventions import project_cam_to_uv

__all__ = [
    "max_displacement_mask",
    "compute_opend4rt_frame0_query_indices",
    "compute_frame0_query_indices_from_uv_and_depth",
    "compute_frame0_query_indices",
    "opend4rt_query_count_from_arrays",
    "filter_query_indices_by_dynamic_mask",
    "assert_query_uv_in_image_bounds",
]


def max_displacement_mask(
    tracks_tn3: np.ndarray,
    threshold: float = 0.01,
) -> np.ndarray:
    """True for tracks whose displacement from frame 0 exceeds ``threshold`` (metres).

    The displacement is the **maximum** over time of the distance to the frame-0
    position. Use **world** ``trajs_3d`` for original PointOdyssey data (static
    background is static in world coordinates) and **ref0-world** tracks for
    TapVid3D / ``opend4rt``.
    """
    tracks = np.asarray(tracks_tn3, dtype=np.float64)
    if tracks.ndim != 3:
        raise ValueError(f"Expected tracks [T,N,3], got {tracks.shape}")
    disp = np.linalg.norm(tracks - tracks[0:1], axis=-1)
    with np.errstate(invalid="ignore"):
        max_disp = np.nanmax(disp, axis=0)
    return np.isfinite(max_disp) & (max_disp > float(threshold))


def compute_opend4rt_frame0_query_indices(
    tracks_xyz_cam: np.ndarray,
    visibility: np.ndarray,
    intrinsics: np.ndarray,
    *,
    image_hw: Tuple[int, int] = (480, 640),
) -> np.ndarray:
    """Frame-0 query indices from camera-space tracks.

    1. ``visible_mask = visibility[0]``
    2. on visible tracks: finite projected UV and finite depth with ``|z| > 1e-8``
    3. the projected UV must fall inside ``image_hw``
    """
    tracks_uv = project_cam_to_uv(tracks_xyz_cam, intrinsics)
    visible_mask = np.asarray(visibility[0], dtype=bool)
    if not np.any(visible_mask):
        return np.zeros((0,), dtype=np.int64)

    idx_vis = np.flatnonzero(visible_mask)
    query_uv = np.asarray(tracks_uv[0, idx_vis], dtype=np.float64)
    depth0 = np.asarray(tracks_xyz_cam[0, idx_vis, 2], dtype=np.float64)
    finite_mask = np.isfinite(query_uv).all(axis=-1)
    finite_mask &= np.isfinite(depth0) & (np.abs(depth0) > 1e-8)
    # Screen-space gate: projected UV must be inside image bounds.
    h, w = int(image_hw[0]), int(image_hw[1])
    finite_mask &= (query_uv[:, 0] >= 0.0) & (query_uv[:, 0] < float(w))
    finite_mask &= (query_uv[:, 1] >= 0.0) & (query_uv[:, 1] < float(h))
    if not np.any(finite_mask):
        return np.zeros((0,), dtype=np.int64)
    return idx_vis[finite_mask].astype(np.int64)


def compute_frame0_query_indices_from_uv_and_depth(
    query_uv_frame0: np.ndarray,
    depth0: np.ndarray,
    visibility_frame0: np.ndarray,
    *,
    image_hw: Tuple[int, int] = (480, 640),
) -> np.ndarray:
    """Frame-0 query indices using provided UV + depth."""
    visible_mask = np.asarray(visibility_frame0, dtype=bool)
    if not np.any(visible_mask):
        return np.zeros((0,), dtype=np.int64)

    idx_vis = np.flatnonzero(visible_mask)
    uv0 = np.asarray(query_uv_frame0, dtype=np.float64)[idx_vis]
    z0 = np.asarray(depth0, dtype=np.float64)[idx_vis]
    finite_mask = np.isfinite(uv0).all(axis=-1)
    finite_mask &= np.isfinite(z0) & (np.abs(z0) > 1e-8)
    # Screen-space gate: projected UV must be inside image bounds.
    h, w = int(image_hw[0]), int(image_hw[1])
    finite_mask &= (uv0[:, 0] >= 0.0) & (uv0[:, 0] < float(w))
    finite_mask &= (uv0[:, 1] >= 0.0) & (uv0[:, 1] < float(h))
    if not np.any(finite_mask):
        return np.zeros((0,), dtype=np.int64)
    return idx_vis[finite_mask].astype(np.int64)


def compute_frame0_query_indices(
    tracks_xyz_cam: np.ndarray,
    visibility: np.ndarray,
    valids: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Alias for :func:`compute_opend4rt_frame0_query_indices` (``valids`` ignored)."""
    del valids
    return compute_opend4rt_frame0_query_indices(tracks_xyz_cam, visibility, intrinsics)


def opend4rt_query_count_from_arrays(
    tracks_xyz_cam: np.ndarray,
    visibility: np.ndarray,
    intrinsics: np.ndarray,
) -> int:
    """Number of frame-0 queries for the given arrays."""
    return int(
        compute_opend4rt_frame0_query_indices(tracks_xyz_cam, visibility, intrinsics).size
    )


def filter_query_indices_by_dynamic_mask(
    query_indices: np.ndarray,
    dynamic_mask: np.ndarray,
) -> np.ndarray:
    """Keep only query indices marked dynamic in ``dynamic_mask`` (length N)."""
    qi = np.asarray(query_indices, dtype=np.int64).reshape(-1)
    mask = np.asarray(dynamic_mask, dtype=bool).reshape(-1)
    if qi.size == 0:
        return qi
    keep = mask[qi]
    return qi[keep]


def assert_query_uv_in_image_bounds(
    query_uv: np.ndarray,
    image_hw: Tuple[int, int],
) -> None:
    """Raise if any frame-0 query UV lies outside ``[0, W) x [0, H)``."""
    uv = np.asarray(query_uv, dtype=np.float64)
    h, w = int(image_hw[0]), int(image_hw[1])
    if uv.size == 0:
        return
    if not np.isfinite(uv).all():
        bad = int(np.sum(~np.isfinite(uv)))
        raise ValueError(f"{bad} query UVs are non-finite")
    oob = (
        (uv[:, 0] < 0.0)
        | (uv[:, 0] >= float(w))
        | (uv[:, 1] < 0.0)
        | (uv[:, 1] >= float(h))
    )
    if np.any(oob):
        n_oob = int(np.sum(oob))
        raise ValueError(
            f"{n_oob}/{uv.shape[0]} query UVs outside image bounds (H={h}, W={w})"
        )


def image_hw_from_video_rgb(video_rgb: np.ndarray) -> Tuple[int, int]:
    """Return ``(H, W)`` of a ``[T, H, W, 3]`` RGB stack."""
    arr = np.asarray(video_rgb)
    if arr.ndim != 4:
        raise ValueError(f"Expected video_rgb [T,H,W,3], got {arr.shape}")
    return int(arr.shape[1]), int(arr.shape[2])


def default_image_hw(image_hw: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """Return the provided resolution or the legacy 480x640 default."""
    if image_hw is None:
        return (480, 640)
    return (int(image_hw[0]), int(image_hw[1]))
