"""Visualization helpers for GFM inference (GLB / PLY export)."""

from __future__ import annotations

import numpy as np
import trimesh
from matplotlib import colormaps
from scipy.spatial.transform import Rotation


def extract_colored_point_cloud(
    predictions: dict,
    conf_thres: float = 20.0,
    conf_key: str = "confidence",
    points_key: str = "world_points_from_depth",
    max_points: int | None = 300000,
    filter_depth_edges: bool = True,
    depth_edge_rtol: float = 0.03,
    frame_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return filtered (vertices [N,3], colors uint8 [N,3])."""
    conf_thres = max(2.0, float(conf_thres))

    points = predictions[points_key]
    conf = predictions[conf_key]
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if conf.ndim == 4 and conf.shape[1] == 1:
        conf = conf[:, 0]

    if filter_depth_edges and "depth" in predictions:
        depth = predictions["depth"]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        conf = conf.copy()
        # Model confidence is not necessarily non-negative, so a zero value
        # can survive a percentile cutoff.  NaN is excluded unconditionally
        # by the later isfinite mask, just like sky-filtered pixels.
        conf[depth_edge(depth, rtol=depth_edge_rtol)] = np.nan

    images = predictions["images"]
    vertices = points.reshape(-1, 3)
    colors = _images_to_rgb(images).reshape(-1, 3)
    colors = (colors * 255).clip(0, 255).astype(np.uint8)
    conf = conf.reshape(-1)

    # NaN conf marks dropped points (e.g. sky). Rely on isfinite + the percentile
    # threshold; do NOT assume conf >= 0 (the model confidence is not activated and may
    # be negative, so a fixed positive floor would wrongly discard valid points).
    mask = np.isfinite(vertices).all(axis=1) & np.isfinite(conf)
    if conf_thres > 0 and np.any(mask):
        conf_threshold = np.percentile(conf[mask], conf_thres)
        mask &= conf >= conf_threshold

    # Ownership is applied after the full-chunk percentile calculation.
    if frame_indices is not None:
        indices = np.asarray(frame_indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= points.shape[0]):
            raise ValueError("frame_indices must index frames in the prediction chunk")
        owned = np.zeros(points.shape[0], dtype=bool)
        owned[indices] = True
        mask &= np.broadcast_to(owned[:, None, None], points.shape[:-1]).reshape(-1)

    vertices = vertices[mask]
    colors = colors[mask]
    vertices, colors = _limit_points(vertices, colors, max_points)

    if vertices.size == 0:
        vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        colors = np.array([[255, 255, 255]], dtype=np.uint8)

    return vertices.astype(np.float32), colors


def predictions_to_ply(
    predictions: dict,
    ply_path: str,
    conf_thres: float = 20.0,
    conf_key: str = "confidence",
    points_key: str = "world_points_from_depth",
    max_points: int | None = 300000,
    filter_depth_edges: bool = True,
    depth_edge_rtol: float = 0.03,
    frame_indices: list[int] | None = None,
) -> str:
    """Export filtered colored point cloud to a PLY file."""
    vertices, colors = extract_colored_point_cloud(
        predictions,
        conf_thres=conf_thres,
        conf_key=conf_key,
        points_key=points_key,
        max_points=max_points,
        filter_depth_edges=filter_depth_edges,
        depth_edge_rtol=depth_edge_rtol,
        frame_indices=frame_indices,
    )
    cloud = trimesh.PointCloud(vertices=vertices, colors=colors)
    cloud.export(ply_path)
    return ply_path


def predictions_to_glb(
    predictions: dict,
    conf_thres: float = 20.0,
    conf_key: str = "confidence",
    points_key: str = "world_points_from_depth",
    show_cam: bool = True,
    max_points: int | None = 300000,
    filter_depth_edges: bool = True,
    depth_edge_rtol: float = 0.03,
    align_to_first_camera: bool = True,
) -> trimesh.Scene:
    """Convert GFM predictions to a GLB / trimesh scene."""
    vertices, colors = extract_colored_point_cloud(
        predictions,
        conf_thres=conf_thres,
        conf_key=conf_key,
        points_key=points_key,
        max_points=max_points,
        filter_depth_edges=filter_depth_edges,
        depth_edge_rtol=depth_edge_rtol,
    )

    if vertices.shape[0] <= 1:
        scene_scale = 1.0
    else:
        lower = np.percentile(vertices, 5, axis=0)
        upper = np.percentile(vertices, 95, axis=0)
        scene_scale = float(np.linalg.norm(upper - lower))
        if scene_scale <= 0:
            scene_scale = 1.0

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=vertices, colors=colors))

    camera_matrices = predictions["extrinsic"]
    extrinsics = np.zeros((len(camera_matrices), 4, 4), dtype=np.float64)
    extrinsics[:, :3, :4] = camera_matrices[:, :3, :4]
    extrinsics[:, 3, 3] = 1.0

    if show_cam:
        colormap = colormaps.get_cmap("gist_rainbow")
        for i, world_to_camera in enumerate(extrinsics):
            camera_to_world = np.linalg.inv(world_to_camera)
            rgba = colormap(i / max(len(extrinsics), 1))
            color = tuple(int(255 * x) for x in rgba[:3])
            integrate_camera_into_scene(scene, camera_to_world, color, scene_scale)

    if align_to_first_camera:
        return apply_scene_alignment(scene, extrinsics)
    return scene


def depth_edge(depth: np.ndarray, rtol: float = 0.03) -> np.ndarray:
    depth = depth.astype(np.float32)
    dx = np.abs(depth[:, :, 1:] - depth[:, :, :-1])
    dy = np.abs(depth[:, 1:, :] - depth[:, :-1, :])
    edge = np.zeros_like(depth, dtype=bool)
    edge[:, :, 1:] |= dx > rtol * np.maximum(depth[:, :, 1:], depth[:, :, :-1])
    edge[:, :, :-1] |= dx > rtol * np.maximum(depth[:, :, 1:], depth[:, :, :-1])
    edge[:, 1:, :] |= dy > rtol * np.maximum(depth[:, 1:, :], depth[:, :-1, :])
    edge[:, :-1, :] |= dy > rtol * np.maximum(depth[:, 1:, :], depth[:, :-1, :])
    return edge


def _images_to_rgb(images: np.ndarray) -> np.ndarray:
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    if images.min() < -0.1:
        images = (images + 1.0) * 0.5
    return images.clip(0.0, 1.0)


def _limit_points(vertices, colors, max_points: int | None):
    """Randomly cap points, or preserve every point when *max_points* is None."""
    if max_points is None or len(vertices) <= max_points:
        return vertices, colors
    idx = np.random.choice(len(vertices), max_points, replace=False)
    return vertices[idx], colors[idx]


def make_camera_cone(transform, face_colors, scene_scale) -> trimesh.Trimesh:
    """Build a single camera-frustum cone mesh in world coordinates."""
    # Build at unit scene scale before shrinking: trimesh's mesh processing
    # removes tiny faces when a small camera baseline is passed to cone().
    cam_width = 0.05
    cam_height = 0.1
    cone = trimesh.creation.cone(radius=cam_width, height=cam_height, sections=4)
    cone.apply_scale(scene_scale)
    rot_x = np.eye(4)
    rot_x[:3, :3] = Rotation.from_euler("x", 180, degrees=True).as_matrix()
    align = np.eye(4)
    align[:3, :3] = Rotation.from_euler("x", -90, degrees=True).as_matrix()
    transform = transform @ align @ rot_x
    cone.apply_transform(transform)
    cone.visual.face_colors = list(face_colors) + [200]
    return cone


def integrate_camera_into_scene(scene, transform, face_colors, scene_scale):
    scene.add_geometry(make_camera_cone(transform, face_colors, scene_scale))


def _extrinsics_to_4x4(camera_matrices: np.ndarray) -> np.ndarray:
    extrinsics = np.zeros((len(camera_matrices), 4, 4), dtype=np.float64)
    extrinsics[:, :3, :4] = camera_matrices[:, :3, :4]
    extrinsics[:, 3, 3] = 1.0
    return extrinsics


def cameras_to_ply(
    predictions: dict,
    ply_path: str,
    scene_scale: float | None = None,
) -> str:
    """Export camera frustums as a colored mesh PLY (world coords, matches scene.ply)."""
    extrinsics = _extrinsics_to_4x4(predictions["extrinsic"])
    cams_to_world = np.linalg.inv(extrinsics)
    centers = cams_to_world[:, :3, 3]

    if scene_scale is None:
        if len(centers) > 1:
            scene_scale = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
        else:
            scene_scale = 0.0
        if scene_scale <= 0:
            scene_scale = 1.0

    colormap = colormaps.get_cmap("gist_rainbow")
    cones = []
    for i, camera_to_world in enumerate(cams_to_world):
        rgba = colormap(i / max(len(cams_to_world), 1))
        color = tuple(int(255 * x) for x in rgba[:3])
        cones.append(make_camera_cone(camera_to_world, color, scene_scale))

    cameras = trimesh.util.concatenate(cones)
    cameras.export(ply_path)
    return ply_path


def apply_scene_alignment(scene: trimesh.Scene, extrinsics: np.ndarray) -> trimesh.Scene:
    opengl = np.eye(4)
    opengl[1, 1] = -1
    opengl[2, 2] = -1
    first_c2w = np.linalg.inv(extrinsics[0])
    align = opengl @ first_c2w
    scene.apply_transform(align)
    return scene
