"""Coordinate conventions and projection helpers.

Two ``mf_files`` layouts are supported:

``opend4rt`` (default)
    ``trajs_3d`` already lives in the **frame-0 reference world frame** and
    ``extrinsics`` are world-to-camera. This is the TapVid3D / WorldTrack layout.

``pointodyssey``
    ``trajs_3d`` are in **global world coordinates** and ``extrinsics`` are
    camera-to-world. This is the PointOdyssey / SynthVerse layout; the conversion
    to the reference frame happens in memory.

The reference frame ("ref0") is the camera frame of the first video frame, so
``ref0 world`` and ``frame-0 camera`` are the same point set expressed in two
naming conventions.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

__all__ = [
    "CoordConvention",
    "DEFAULT_COORD_CONVENTION",
    "normalize_coord_convention",
    "project_cam_to_uv",
    "normalize_uv_to_01",
    "ref0_to_cam_tracks",
    "tracks_cam_to_ref0_world",
    "extrinsics_c2w_to_w2c",
    "pointodyssey_world_to_ref0_world",
    "pointodyssey_cam_to_ref0_world",
    "normalize_mf_tracks_to_opend4rt",
]

CoordConvention = Literal["opend4rt", "pointodyssey"]
DEFAULT_COORD_CONVENTION: CoordConvention = "opend4rt"


def normalize_coord_convention(coord_convention: Optional[str] = None) -> CoordConvention:
    """Normalize CLI / config aliases to a loader convention name."""
    if coord_convention is None or str(coord_convention).strip() == "":
        return DEFAULT_COORD_CONVENTION
    cc = str(coord_convention).strip().lower()
    if cc == "opend4rt":
        return "opend4rt"
    if cc in ("pointodyssey", "po", "po_synthverse", "synthverse", "point_odyssey"):
        return "pointodyssey"
    raise ValueError(
        f"Unknown coord_convention {coord_convention!r}; "
        "use 'opend4rt' (TapVid3D / WorldTrack) or 'pointodyssey' (PO / SynthVerse)."
    )


def project_cam_to_uv(
    tracks_xyz_cam: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Project camera-space points ``[..., 3]`` to pixel coordinates ``[..., 2]``.

    ``intrinsics`` is ``[fx, fy, cx, cy]``.
    """
    fx, fy, cx, cy = np.asarray(intrinsics, dtype=np.float64).reshape(-1)[:4]
    z = tracks_xyz_cam[..., 2]
    safe_z = np.where(np.abs(z) > 1e-8, z, np.nan)
    u = (tracks_xyz_cam[..., 0] / safe_z) * fx + cx
    v = (tracks_xyz_cam[..., 1] / safe_z) * fy + cy
    return np.stack([u, v], axis=-1)


def normalize_uv_to_01(
    query_uv: np.ndarray,
    image_hw: Tuple[int, int],
) -> np.ndarray:
    """Convert pixel coordinates to ``[0, 1]`` normalized coordinates."""
    h, w = int(image_hw[0]), int(image_hw[1])
    out = np.asarray(query_uv, dtype=np.float64).copy()
    out[:, 0] /= float(max(w - 1, 1))
    out[:, 1] /= float(max(h - 1, 1))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def ref0_to_cam_tracks(
    points_ref0_tq3: np.ndarray,
    extrinsics_w2c: np.ndarray,
) -> np.ndarray:
    """Frame-0 ref coordinates ``[T, Q, 3]`` -> per-frame native camera coords.

    ``X_cam_t = w2c_t @ c2w_0 @ X_ref0`` with ``c2w_0 = inv(w2c_0)``.
    """
    pts = np.asarray(points_ref0_tq3, dtype=np.float64)
    ext = np.asarray(extrinsics_w2c, dtype=np.float64)
    c2w0 = np.linalg.inv(ext[0])
    out = np.full_like(pts, np.nan, dtype=np.float64)
    for t in range(int(pts.shape[0])):
        w2c_t = ext[t]
        p = pts[t]
        fin = np.isfinite(p).all(axis=-1)
        if not np.any(fin):
            continue
        ph = np.concatenate([p[fin], np.ones((int(fin.sum()), 1), dtype=np.float64)], axis=-1)
        world = (ph @ c2w0.T)[:, :3]
        wh = np.concatenate([world, np.ones((world.shape[0], 1), dtype=np.float64)], axis=-1)
        out[t, fin] = (wh @ w2c_t.T)[:, :3]
    return out


def tracks_cam_to_ref0_world(
    tracks_xyz_cam: np.ndarray,
    extrinsics_w2c: np.ndarray,
) -> np.ndarray:
    """Camera-space tracks ``[T, N, 3]`` -> ref0 world ``[T, N, 3]``.

    Extrinsics are world-to-camera; frame 0 is used as the reference.
    """
    tracks_xyz_cam = np.asarray(tracks_xyz_cam, dtype=np.float64)
    extrinsics_w2c = np.asarray(extrinsics_w2c, dtype=np.float64)
    frame_count = tracks_xyz_cam.shape[0]
    first_inv = np.linalg.inv(extrinsics_w2c[0])
    extrinsics_n = np.asarray([extr @ first_inv for extr in extrinsics_w2c[:frame_count]], dtype=np.float64)
    extrinsics_c2w = np.linalg.inv(extrinsics_n)
    tracks_xyz_world = np.empty_like(tracks_xyz_cam, dtype=np.float64)
    for frame_idx in range(frame_count):
        rot = extrinsics_c2w[frame_idx, :3, :3]
        trans = extrinsics_c2w[frame_idx, :3, 3]
        tracks_xyz_world[frame_idx] = (rot @ tracks_xyz_cam[frame_idx].T).T + trans
    return tracks_xyz_world


def extrinsics_c2w_to_w2c(extrinsics_c2w: np.ndarray) -> np.ndarray:
    """Invert PointOdyssey / SynthVerse ``extrinsics`` (c2w) to the w2c layout."""
    ext = np.asarray(extrinsics_c2w, dtype=np.float64)
    if ext.ndim == 2:
        return np.linalg.inv(ext)
    out = np.full_like(ext, np.nan, dtype=np.float64)
    for t in range(int(ext.shape[0])):
        out[t] = np.linalg.inv(ext[t])
    return out


def pointodyssey_world_to_ref0_world(
    tracks_world_tn3: np.ndarray,
    extrinsics_c2w: np.ndarray,
) -> np.ndarray:
    """Convert PO / SynthVerse world ``trajs_3d`` + c2w extrinsics to ref0-world layout."""
    pts = np.asarray(tracks_world_tn3, dtype=np.float64)
    c2w = np.asarray(extrinsics_c2w, dtype=np.float64)
    if pts.ndim != 3 or c2w.ndim != 3:
        raise ValueError(
            f"Expected tracks [T,N,3] and extrinsics [T,4,4], got {pts.shape} and {c2w.shape}"
        )
    w2c0 = np.linalg.inv(c2w[0])
    out = np.full_like(pts, np.nan, dtype=np.float64)
    for t in range(int(pts.shape[0])):
        p = pts[t]
        fin = np.isfinite(p).all(axis=-1)
        if not np.any(fin):
            continue
        ph = np.concatenate([p[fin], np.ones((int(fin.sum()), 1), dtype=np.float64)], axis=-1)
        out[t, fin] = (ph @ w2c0.T)[:, :3]
    return out


def pointodyssey_cam_to_ref0_world(
    tracks_cam_tn3: np.ndarray,
    extrinsics_c2w: np.ndarray,
) -> np.ndarray:
    """Convert per-frame camera ``trajs_3d`` to ref0-world layout (legacy PO layout)."""
    pts = np.asarray(tracks_cam_tn3, dtype=np.float64)
    c2w = np.asarray(extrinsics_c2w, dtype=np.float64)
    if pts.ndim != 3 or c2w.ndim != 3:
        raise ValueError(
            f"Expected tracks [T,N,3] and extrinsics [T,4,4], got {pts.shape} and {c2w.shape}"
        )
    w2c0 = np.linalg.inv(c2w[0])
    out = np.full_like(pts, np.nan, dtype=np.float64)
    for t in range(int(pts.shape[0])):
        p = pts[t]
        fin = np.isfinite(p).all(axis=-1)
        if not np.any(fin):
            continue
        ph = np.concatenate([p[fin], np.ones((int(fin.sum()), 1), dtype=np.float64)], axis=-1)
        world = (ph @ c2w[t].T)[:, :3]
        wh = np.concatenate([world, np.ones((world.shape[0], 1), dtype=np.float64)], axis=-1)
        out[t, fin] = (wh @ w2c0.T)[:, :3]
    return out


def normalize_mf_tracks_to_opend4rt(
    tracks_per_frame_tn3: np.ndarray,
    extrinsics_raw: np.ndarray,
    *,
    coord_convention: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(tracks_ref0, extrinsics_w2c)`` for metrics and query gating."""
    cc = normalize_coord_convention(coord_convention)
    tracks = np.asarray(tracks_per_frame_tn3, dtype=np.float64)
    extr = np.asarray(extrinsics_raw, dtype=np.float64)
    if cc == "opend4rt":
        return tracks, extr
    # PO / SynthVerse mf JSON: world ``trajs_3d`` + c2w ``extrinsics``.
    tracks_ref0 = pointodyssey_world_to_ref0_world(tracks, extr)
    extrinsics_w2c = extrinsics_c2w_to_w2c(extr)
    return tracks_ref0, extrinsics_w2c
