"""A minimal track overlay renderer.

The renderer is deliberately small: it draws the 2D projections of one or more
track sets on top of the video frames. It is meant for eyeballing a result, not
for producing publication-quality figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "PALETTE",
    "render_layers",
    "render_tracks",
    "write_video",
    "save_overlay_video",
    "project_ref0_to_uv",
]

#: A fixed, high-contrast colour palette (BGR order, as expected by OpenCV).
PALETTE: Sequence[tuple] = (
    (0, 200, 255),
    (255, 160, 0),
    (0, 255, 120),
    (255, 80, 200),
    (80, 200, 255),
    (200, 255, 60),
    (255, 120, 60),
    (140, 90, 220),
    (60, 255, 220),
    (220, 220, 90),
)

#: Colours used for the ground-truth / prediction layers of the overlay.
GT_COLOR = (0, 220, 120)
PRED_COLOR = (0, 200, 255)


def _to_uint8_rgb(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float64)
    if float(np.nanmax(arr)) <= 1.0 + 1e-3:
        arr = arr * 255.0
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def render_layers(
    video_rgb: np.ndarray,
    layers: Sequence[Dict[str, Any]],
    *,
    radius: int = 2,
    draw_index: bool = False,
) -> np.ndarray:
    """Draw several track layers on the video frames.

    Args:
        video_rgb: ``[T, H, W, 3]`` RGB frames.
        layers: One dict per layer, with the keys ``uv`` (``[T, Q, 2]`` pixel
            coordinates), and optionally ``visibility`` (``[T, Q]``),
            ``color`` (BGR tuple), ``radius`` and ``palette_offset``.
        radius: Default radius of the drawn points, in pixels.
        draw_index: Draw the query index next to each point.

    Returns:
        ``[T, H, W, 3]`` uint8 RGB frames with the layers drawn on them.
    """
    import cv2

    frames = _to_uint8_rgb(video_rgb)
    if frames.ndim != 4:
        raise ValueError(f"Expected video_rgb [T,H,W,3], got {frames.shape}")

    num_frames = frames.shape[0]
    height, width = frames.shape[1], frames.shape[2]
    out = frames.copy()

    for layer in layers:
        uv = np.asarray(layer["uv"], dtype=np.float64)
        if uv.ndim != 3 or uv.shape[-1] != 2:
            raise ValueError(f"Expected layer uv [T,Q,2], got {uv.shape}")
        visibility = layer.get("visibility")
        color = layer.get("color")
        layer_radius = int(layer.get("radius", radius))
        offset = int(layer.get("palette_offset", 0))

        for t in range(min(num_frames, uv.shape[0])):
            canvas = cv2.cvtColor(out[t], cv2.COLOR_RGB2BGR)
            for q in range(uv.shape[1]):
                if visibility is not None and not bool(np.asarray(visibility)[t, q]):
                    continue
                u = float(uv[t, q, 0])
                v = float(uv[t, q, 1])
                if not (np.isfinite(u) and np.isfinite(v)):
                    continue
                if not (0 <= u < width and 0 <= v < height):
                    continue
                bgr = color if color is not None else PALETTE[(q + offset) % len(PALETTE)]
                cv2.circle(canvas, (int(round(u)), int(round(v))), layer_radius, bgr, -1)
                if draw_index:
                    cv2.putText(
                        canvas,
                        str(q),
                        (int(round(u)) + 4, int(round(v)) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        bgr,
                        1,
                        cv2.LINE_AA,
                    )
            out[t] = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return out


def render_tracks(
    video_rgb: np.ndarray,
    tracks_uv: np.ndarray,
    *,
    visibility: Optional[np.ndarray] = None,
    radius: int = 2,
    color: Optional[tuple] = None,
    draw_index: bool = False,
) -> np.ndarray:
    """Draw a single track set on the video frames."""
    layer: Dict[str, Any] = {"uv": tracks_uv}
    if visibility is not None:
        layer["visibility"] = visibility
    if color is not None:
        layer["color"] = color
    return render_layers(
        video_rgb,
        [layer],
        radius=radius,
        draw_index=draw_index,
    )


def write_video(frames: np.ndarray, path: Any, fps: float = 15.0) -> Path:
    """Write RGB frames to a video file."""
    import cv2

    arr = _to_uint8_rgb(frames)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    height, width = arr.shape[1], arr.shape[2]
    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    try:
        for frame in arr:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return out


def save_overlay_video(
    video_rgb: np.ndarray,
    layers: Sequence[Dict[str, Any]],
    path: Any,
    *,
    fps: float = 15.0,
    radius: int = 2,
) -> Path:
    """Render the layers and write them to a video file."""
    frames = render_layers(video_rgb, layers, radius=radius)
    return write_video(frames, path, fps=fps)


def project_ref0_to_uv(
    tracks_ref0: np.ndarray,
    extrinsics_w2c: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Project reference-frame 3D positions into per-frame pixel coordinates."""
    from .data.conventions import project_cam_to_uv, ref0_to_cam_tracks

    tracks = np.asarray(tracks_ref0, dtype=np.float64)
    cam = ref0_to_cam_tracks(tracks, np.asarray(extrinsics_w2c, dtype=np.float64))
    return project_cam_to_uv(cam, intrinsics)
