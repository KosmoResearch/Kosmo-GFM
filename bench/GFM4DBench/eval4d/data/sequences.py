"""Sequence loading for WorldTrack / TapVid3D ``mf_files`` JSON and official NPZ.

The ``mf_files`` JSON maps a sequence name to a list of frames, each holding a
single ``view`` with:

===============  ==========================================================
``rgb``          path to the frame image
``trajs_3d``     path to the ``[N, 3]`` 3D tracks (world or ref0, see conventions)
``trajs_2d``     path to the ``[N, 2]`` pixel tracks (**not** used for queries)
``visibs``       path to the ``[N]`` visibility flags
``valids``       path to the ``[N]`` validity flags (**not** used for queries)
``extrinsics``   path to the ``[4, 4]`` camera extrinsics
``cam_in``       ``[fx, fy, cx, cy]`` camera intrinsics
===============  ==========================================================

Paths may be stored either as strings (resolved against ``data_root``) or as
inline arrays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .conventions import (
    normalize_coord_convention,
    normalize_mf_tracks_to_opend4rt,
    normalize_uv_to_01,
    project_cam_to_uv,
    ref0_to_cam_tracks,
    tracks_cam_to_ref0_world,
)
from .queries import (
    compute_frame0_query_indices_from_uv_and_depth,
    compute_opend4rt_frame0_query_indices,
    filter_query_indices_by_dynamic_mask,
    max_displacement_mask,
)

__all__ = [
    "PathLike",
    "WorldTrackSequence",
    "resolve_worldtrack_path",
    "read_sequence_frame0_image_hw",
    "infer_trajs_2d_fixed_rel_path",
    "trajs_2d_path_candidates_from_view",
    "resolve_trajs_2d_for_view",
    "prefer_worldtrack_json_path",
    "resolve_opend4rt_frame_count",
    "list_sequences_in_json",
    "load_mf_files",
    "frame0_query_indices_for_sequence",
    "dynamic_query_mask_for_sequence",
    "filter_sequences_with_frame0_queries",
    "load_sequence_from_json",
    "load_sequence_from_npz",
    "compare_json_vs_npz",
]

PathLike = Union[str, Path]


@dataclass
class WorldTrackSequence:
    """One video sequence ready for WorldTrack metrics and model inference."""

    video_name: str
    num_frames: int
    num_tracks: int
    video_rgb: np.ndarray  # [T, H, W, 3] uint8
    tracks_xyz_cam: np.ndarray  # [T, N, 3] float64
    tracks_uv: np.ndarray  # [T, N, 2] float64
    visibility: np.ndarray  # [T, N] bool
    valids: np.ndarray  # [T, N] bool
    intrinsics: np.ndarray  # [4] fx, fy, cx, cy
    extrinsics_w2c: np.ndarray  # [T, 4, 4] float64
    gt_tracks_world: np.ndarray  # [T, N, 3] ref0 world
    query_indices: np.ndarray  # [Q] int64 indices into N
    query_uv: np.ndarray  # [Q, 2] pixel coords at frame 0
    query_uv_norm: np.ndarray  # [Q, 2] in [0, 1]
    json_path: Optional[str] = None
    data_root: Optional[str] = None

    @property
    def num_queries(self) -> int:
        return int(self.query_indices.shape[0])

    @property
    def image_hw(self) -> Tuple[int, int]:
        """``(H, W)`` of the video frames."""
        return int(self.video_rgb.shape[1]), int(self.video_rgb.shape[2])

    def gt_tracks_world_queries(self) -> np.ndarray:
        """GT restricted to the frame-0 queries: ``[T, Q, 3]``."""
        return self.gt_tracks_world[:, self.query_indices]


def resolve_worldtrack_path(
    rel_path: str,
    *,
    data_root: PathLike,
    json_path: Optional[PathLike] = None,
) -> Path:
    """Resolve a path stored in the ``mf_files`` JSON."""
    p = Path(rel_path)
    if p.is_file():
        return p

    root = Path(data_root)
    candidates: List[Path] = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(root / p)
        candidates.append(root / p.name)
        if "Tapvid3d_mini_extracted/" in rel_path.replace("\\", "/"):
            suffix = rel_path.split("Tapvid3d_mini_extracted/")[-1]
            # Release archives place the subset directories directly under root.
            candidates.append(root / suffix)
            candidates.append(root / "Tapvid3d_mini_extracted" / suffix)
            candidates.append(root.parent / "Tapvid3d_mini_extracted" / suffix)

    if json_path is not None:
        jp = Path(json_path)
        candidates.append(jp.parent / p)
        candidates.append(jp.parent.parent / p)

    for cand in candidates:
        if cand.is_file():
            return cand

    raise FileNotFoundError(
        f"Could not resolve path for {rel_path!r} under data_root={root}"
    )


def read_sequence_frame0_image_hw(
    json_path: PathLike,
    sequence_name: str,
    *,
    data_root: Optional[PathLike] = None,
) -> Tuple[int, int]:
    """Return ``(H, W)`` of the frame-0 RGB for the screen-space query gate.

    The real image header is always read (via OpenCV); the resolution is never
    assumed.
    """
    import cv2

    json_path = Path(json_path)
    if data_root is None:
        data_root = json_path.parent.parent
    data_root = Path(data_root)

    mf = load_mf_files(json_path)
    if sequence_name not in mf:
        raise KeyError(f"Sequence {sequence_name!r} not in {json_path}")
    frames = mf[sequence_name]
    if not frames:
        raise RuntimeError(f"Empty frame list for {sequence_name}")

    view = frames[0]["views"][0]
    rgb_path = resolve_worldtrack_path(view["rgb"], data_root=data_root, json_path=json_path)
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read RGB for image_hw: {rgb_path}")
    return int(bgr.shape[0]), int(bgr.shape[1])


def infer_trajs_2d_fixed_rel_path(trajs_2d_rel: str) -> Optional[str]:
    """Infer the fixed ``trajs_2d`` relative path from the original TapVid3D layout.

    Example::

        Tapvid3d_mini_extracted/adt_mini/SEQ/trajs_2d/000000.npy
        -> Tapvid3d_mini_extracted/adt_mini_trajs2d_fixed/adt_mini/SEQ/trajs_2d/000000.npy
    """
    rel = str(trajs_2d_rel).replace("\\", "/")
    prefix = "Tapvid3d_mini_extracted/"
    if not rel.startswith(prefix) or "/trajs_2d/" not in rel:
        return None
    suffix = rel[len(prefix) :]
    subset, rest = suffix.split("/", 1)
    if not subset or not rest:
        return None
    return f"{prefix}{subset}_trajs2d_fixed/{subset}/{rest}"


def trajs_2d_path_candidates_from_view(view: Dict[str, Any]) -> List[str]:
    """Candidate ``trajs_2d`` paths: explicit ``trajs_2d_fixed`` then ``trajs_2d``."""
    refs: List[str] = []
    fixed = view.get("trajs_2d_fixed")
    if isinstance(fixed, str) and fixed.strip():
        refs.append(fixed.strip())

    orig = view.get("trajs_2d")
    if isinstance(orig, str) and orig.strip() and orig.strip() not in refs:
        refs.append(orig.strip())
    return refs


def resolve_trajs_2d_for_view(
    view: Dict[str, Any],
    *,
    data_root: PathLike,
    json_path: Optional[PathLike] = None,
) -> Path:
    """Resolve the on-disk ``trajs_2d`` for one mf view (prefers ``trajs_2d_fixed``)."""
    refs = trajs_2d_path_candidates_from_view(view)
    if not refs:
        raise FileNotFoundError("view has no trajs_2d / trajs_2d_fixed path")

    last_err: Optional[FileNotFoundError] = None
    for ref in refs:
        try:
            return resolve_worldtrack_path(ref, data_root=data_root, json_path=json_path)
        except FileNotFoundError as exc:
            last_err = exc
    assert last_err is not None
    raise last_err


def prefer_worldtrack_json_path(
    json_path: PathLike,
    *,
    data_root: Optional[PathLike] = None,
) -> Path:
    """Return the resolved ``mf_files`` JSON path."""
    del data_root
    return Path(json_path).resolve()


def resolve_opend4rt_frame_count(
    num_frames: int,
    *,
    num_images: int,
    num_track_frames: int,
    num_visibility_frames: int,
) -> int:
    """Clip length: the minimum over the requested length and the available arrays."""
    fc = min(
        int(num_frames),
        int(num_images),
        int(num_track_frames),
        int(num_visibility_frames),
    )
    if fc <= 0:
        raise ValueError(
            f"Non-positive clip length: num_frames={num_frames} images={num_images} "
            f"tracks={num_track_frames} visibility={num_visibility_frames}"
        )
    return fc


def load_mf_files(json_path: PathLike) -> Dict[str, Any]:
    """Load the ``mf_files`` mapping from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    mf = payload.get("mf_files", payload)
    if not isinstance(mf, dict):
        raise ValueError(f"Invalid mf_files JSON: {json_path}")
    return mf


def list_sequences_in_json(json_path: PathLike) -> List[str]:
    """Sorted sequence names contained in an ``mf_files`` JSON."""
    return sorted(load_mf_files(json_path).keys())


def _load_per_frame_npy_sequence(
    json_path: Path,
    sequence_name: str,
    *,
    data_root: Path,
    num_frames: int,
    key: str,
) -> np.ndarray:
    mf = load_mf_files(json_path)
    frames = mf[sequence_name][: min(int(num_frames), len(mf[sequence_name]))]

    def _load_npy(ref: Any) -> np.ndarray:
        if isinstance(ref, str):
            p = resolve_worldtrack_path(ref, data_root=data_root, json_path=json_path)
            return np.load(p, allow_pickle=False)
        return np.asarray(ref)

    return np.stack([_load_npy(frame["views"][0][key]) for frame in frames], axis=0)


def dynamic_query_mask_for_sequence(
    json_path: PathLike,
    sequence_name: str,
    *,
    data_root: Optional[PathLike] = None,
    num_frames: int = 64,
    coord_convention: Optional[str] = None,
    dynamic_query_threshold: float = 0.01,
) -> np.ndarray:
    """Per-track dynamic mask ``[N]`` for one sequence."""
    json_path = Path(json_path)
    if data_root is None:
        data_root = json_path.parent.parent
    data_root = Path(data_root)
    cc = normalize_coord_convention(coord_convention)

    tracks = _load_per_frame_npy_sequence(
        json_path,
        sequence_name,
        data_root=data_root,
        num_frames=num_frames,
        key="trajs_3d",
    )
    if cc == "pointodyssey":
        # World coordinates: the static background is static in world space.
        return max_displacement_mask(tracks, dynamic_query_threshold)

    extr_raw = _load_per_frame_npy_sequence(
        json_path,
        sequence_name,
        data_root=data_root,
        num_frames=num_frames,
        key="extrinsics",
    )
    tracks_ref0, _ = normalize_mf_tracks_to_opend4rt(
        tracks,
        extr_raw,
        coord_convention=cc,
    )
    return max_displacement_mask(tracks_ref0, dynamic_query_threshold)


def frame0_query_indices_for_sequence(
    json_path: PathLike,
    sequence_name: str,
    *,
    data_root: Optional[PathLike] = None,
    num_frames: int = 64,
    coord_convention: Optional[str] = None,
    dynamic_query_threshold: Optional[float] = None,
) -> np.ndarray:
    """Frame-0 query indices for one sequence (empty array if there are none)."""
    json_path = Path(json_path)
    if data_root is None:
        data_root = json_path.parent.parent
    data_root = Path(data_root)
    cc = normalize_coord_convention(coord_convention)

    mf = load_mf_files(json_path)
    if sequence_name not in mf:
        raise KeyError(f"Sequence {sequence_name!r} not in {json_path}")

    frames = mf[sequence_name]
    if not frames:
        return np.zeros((0,), dtype=np.int64)

    view = frames[0]["views"][0]

    def _load_npy(key: str) -> np.ndarray:
        ref = view[key]
        if isinstance(ref, str):
            p = resolve_worldtrack_path(ref, data_root=data_root, json_path=json_path)
            return np.load(p)
        return np.asarray(ref)

    tracks_raw = np.asarray(_load_npy("trajs_3d"), dtype=np.float64)
    visibility = np.asarray(_load_npy("visibs"), dtype=bool)
    extr_raw = np.asarray(_load_npy("extrinsics"), dtype=np.float64)
    intrinsics = np.asarray(view["cam_in"], dtype=np.float64).reshape(-1)[:4]

    tracks_ref0, extr_w2c = normalize_mf_tracks_to_opend4rt(
        tracks_raw[np.newaxis, ...],
        extr_raw[np.newaxis, ...],
        coord_convention=cc,
    )
    cam0 = ref0_to_cam_tracks(tracks_ref0, extr_w2c)[0]
    depth0 = cam0[:, 2]
    uv0 = project_cam_to_uv(cam0[np.newaxis, ...], intrinsics)[0]
    image_hw = read_sequence_frame0_image_hw(json_path, sequence_name, data_root=data_root)
    vis0 = visibility[0] if visibility.ndim == 2 else visibility
    query_indices = compute_frame0_query_indices_from_uv_and_depth(
        uv0, depth0, vis0, image_hw=image_hw
    )
    if dynamic_query_threshold is not None:
        dyn_mask = dynamic_query_mask_for_sequence(
            json_path,
            sequence_name,
            data_root=data_root,
            num_frames=num_frames,
            coord_convention=cc,
            dynamic_query_threshold=float(dynamic_query_threshold),
        )
        query_indices = filter_query_indices_by_dynamic_mask(query_indices, dyn_mask)
    return query_indices


def filter_sequences_with_frame0_queries(
    json_path: PathLike,
    sequence_names: Sequence[str],
    *,
    data_root: Optional[PathLike] = None,
    num_frames: int = 64,
    coord_convention: Optional[str] = None,
    dynamic_query_threshold: Optional[float] = None,
) -> Tuple[List[str], List[str]]:
    """Split sequences into those with valid frame-0 queries vs skipped."""
    valid: List[str] = []
    skipped: List[str] = []
    for name in sequence_names:
        qi = frame0_query_indices_for_sequence(
            json_path,
            name,
            data_root=data_root,
            num_frames=num_frames,
            coord_convention=coord_convention,
            dynamic_query_threshold=dynamic_query_threshold,
        )
        if int(qi.size) > 0:
            valid.append(name)
        else:
            skipped.append(name)
    return valid, skipped


def load_sequence_from_json(
    json_path: PathLike,
    sequence_name: str,
    *,
    data_root: Optional[PathLike] = None,
    num_frames: int = 64,
    load_rgb: bool = True,
    coord_convention: Optional[str] = None,
    dynamic_query_threshold: Optional[float] = None,
) -> WorldTrackSequence:
    """Load one sequence from an ``mf_files`` JSON (first ``num_frames`` frames).

    Args:
        coord_convention: ``opend4rt`` (default) means ``trajs_3d`` is ref0 world and
            ``extrinsics`` is w2c (TapVid3D); ``pointodyssey`` means ``trajs_3d`` is
            world and ``extrinsics`` is c2w (original PointOdyssey layout).
    """
    json_path = Path(json_path)
    if data_root is None:
        data_root = json_path.parent.parent
    data_root = Path(data_root)
    cc = normalize_coord_convention(coord_convention)

    mf = load_mf_files(json_path)
    if sequence_name not in mf:
        raise KeyError(f"Sequence {sequence_name!r} not in {json_path}")

    frames = mf[sequence_name]
    if not frames:
        raise RuntimeError(f"Empty frame list for {sequence_name}")

    # Pre-cap the JSON frame list; the final clip may shrink further after loading.
    list_cap = min(int(num_frames), len(frames))
    frames = frames[:list_cap]

    rgbs: List[np.ndarray] = []
    tracks_ref0_list: List[np.ndarray] = []
    vis_list: List[np.ndarray] = []
    val_list: List[np.ndarray] = []
    extr_list: List[np.ndarray] = []
    intrinsics: Optional[np.ndarray] = None

    for frame in frames:
        if len(frame.get("views", [])) != 1:
            raise ValueError(
                f"Expected 1 view per frame for {sequence_name}, got {len(frame.get('views', []))}"
            )
        view = frame["views"][0]

        if load_rgb:
            rgb_path = resolve_worldtrack_path(
                view["rgb"], data_root=data_root, json_path=json_path
            )
            import cv2

            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Failed to read RGB: {rgb_path}")
            rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        def _load_npy(key: str) -> np.ndarray:
            ref = view[key]
            if isinstance(ref, str):
                p = resolve_worldtrack_path(ref, data_root=data_root, json_path=json_path)
                return np.load(p)
            return np.asarray(ref)

        tracks_ref0_list.append(np.asarray(_load_npy("trajs_3d"), dtype=np.float64))
        vis_list.append(np.asarray(_load_npy("visibs"), dtype=bool))
        val_list.append(np.asarray(_load_npy("valids"), dtype=bool) if "valids" in view
                        else np.ones_like(vis_list[-1], dtype=bool))

        if intrinsics is None:
            intrinsics = np.asarray(view["cam_in"], dtype=np.float64).reshape(-1)[:4]
        extr_list.append(np.asarray(_load_npy("extrinsics"), dtype=np.float64))

    tracks_raw_stack = np.stack(tracks_ref0_list, axis=0)
    visibility = np.stack(vis_list, axis=0)
    valids = np.stack(val_list, axis=0)
    extrinsics_raw = np.stack(extr_list, axis=0)

    num_images = len(rgbs) if load_rgb else int(tracks_raw_stack.shape[0])
    frame_count = resolve_opend4rt_frame_count(
        num_frames,
        num_images=num_images,
        num_track_frames=int(tracks_raw_stack.shape[0]),
        num_visibility_frames=int(visibility.shape[0]),
    )
    if load_rgb:
        rgbs = rgbs[:frame_count]
    tracks_raw_stack = tracks_raw_stack[:frame_count]
    visibility = visibility[:frame_count]
    valids = valids[:frame_count]
    extrinsics_raw = extrinsics_raw[:frame_count]

    tracks_ref0, extrinsics_w2c = normalize_mf_tracks_to_opend4rt(
        tracks_raw_stack,
        extrinsics_raw,
        coord_convention=cc,
    )

    # Keep a camera-space copy for sanity checks / UV projection.
    tracks_xyz_cam = ref0_to_cam_tracks(tracks_ref0, extrinsics_w2c)
    tracks_uv = project_cam_to_uv(tracks_xyz_cam, intrinsics)
    gt_tracks_world = np.asarray(tracks_ref0, dtype=np.float64)
    if load_rgb:
        h, w = rgbs[0].shape[:2]
        video_rgb = np.stack(rgbs, axis=0)
    else:
        h, w = read_sequence_frame0_image_hw(
            json_path, sequence_name, data_root=data_root
        )
        video_rgb = np.zeros((frame_count, h, w, 3), dtype=np.uint8)

    query_indices = compute_frame0_query_indices_from_uv_and_depth(
        tracks_uv[0],
        tracks_xyz_cam[0, :, 2],
        visibility[0],
        image_hw=(h, w),
    )
    if dynamic_query_threshold is not None:
        if cc == "pointodyssey":
            dyn_mask = max_displacement_mask(tracks_raw_stack, dynamic_query_threshold)
        else:
            dyn_mask = max_displacement_mask(tracks_ref0, dynamic_query_threshold)
        query_indices = filter_query_indices_by_dynamic_mask(query_indices, dyn_mask)

    if query_indices.size == 0:
        raise RuntimeError(f"No valid frame-0 queries for {sequence_name}")

    query_uv = tracks_uv[0, query_indices]
    query_uv_norm = normalize_uv_to_01(query_uv, (h, w))

    return WorldTrackSequence(
        video_name=sequence_name,
        num_frames=frame_count,
        num_tracks=int(tracks_xyz_cam.shape[1]),
        video_rgb=video_rgb,
        tracks_xyz_cam=tracks_xyz_cam,
        tracks_uv=tracks_uv,
        visibility=visibility,
        valids=valids,
        intrinsics=intrinsics,
        extrinsics_w2c=extrinsics_w2c,
        gt_tracks_world=gt_tracks_world,
        query_indices=query_indices,
        query_uv=query_uv.astype(np.float32),
        query_uv_norm=query_uv_norm,
        json_path=str(json_path),
        data_root=str(data_root),
    )


def load_sequence_from_npz(
    npz_path: PathLike,
    *,
    num_frames: int = 64,
    load_rgb: bool = True,
    trusted_pickle: bool = False,
) -> WorldTrackSequence:
    """Load a WorldTrack NPZ for cross-checking.

    Object arrays are rejected by default. Set ``trusted_pickle=True`` only for
    verified archives that require pickled JPEG byte arrays: pickle loading can
    execute code. Numeric per-frame NPY inputs never enable pickle.
    """
    import cv2

    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=trusted_pickle) as pack:
        images_jpeg_bytes = np.asarray(pack["images_jpeg_bytes"])
        tracks_xyz_cam = np.asarray(pack["tracks_XYZ"], dtype=np.float64)
        intrinsics = np.asarray(pack["fx_fy_cx_cy"], dtype=np.float64)
        visibility = np.asarray(pack["visibility"], dtype=bool)
        extrinsics_w2c = np.asarray(pack["extrinsics_w2c"], dtype=np.float64)

    frame_count = resolve_opend4rt_frame_count(
        num_frames,
        num_images=int(images_jpeg_bytes.shape[0]),
        num_track_frames=int(tracks_xyz_cam.shape[0]),
        num_visibility_frames=int(visibility.shape[0]),
    )
    images_jpeg_bytes = images_jpeg_bytes[:frame_count]
    tracks_xyz_cam = tracks_xyz_cam[:frame_count]
    visibility = visibility[:frame_count]
    extrinsics_w2c = extrinsics_w2c[:frame_count]

    if load_rgb:
        rgbs = []
        for frame_bytes in images_jpeg_bytes:
            arr = np.frombuffer(frame_bytes, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Failed to decode JPEG in {npz_path}")
            rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        video_rgb = np.stack(rgbs, axis=0)
    else:
        video_rgb = np.zeros((frame_count, 480, 640, 3), dtype=np.uint8)

    tracks_uv = project_cam_to_uv(tracks_xyz_cam, intrinsics)
    valids = visibility.copy()
    # The NPZ stores camera-space tracks; convert them to the ref0 world frame.
    gt_tracks_world = tracks_cam_to_ref0_world(tracks_xyz_cam, extrinsics_w2c)
    query_indices = compute_opend4rt_frame0_query_indices(
        tracks_xyz_cam, visibility, intrinsics
    )
    query_uv = tracks_uv[0, query_indices]
    h, w = video_rgb.shape[1:3]
    query_uv_norm = normalize_uv_to_01(query_uv, (h, w))

    return WorldTrackSequence(
        video_name=npz_path.stem,
        num_frames=frame_count,
        num_tracks=int(tracks_xyz_cam.shape[1]),
        video_rgb=video_rgb,
        tracks_xyz_cam=tracks_xyz_cam,
        tracks_uv=tracks_uv,
        visibility=visibility,
        valids=valids,
        intrinsics=intrinsics,
        extrinsics_w2c=extrinsics_w2c,
        gt_tracks_world=gt_tracks_world,
        query_indices=query_indices,
        query_uv=query_uv.astype(np.float32),
        query_uv_norm=query_uv_norm,
        json_path=None,
        data_root=str(npz_path.parent),
    )


def compare_json_vs_npz(
    json_path: PathLike,
    npz_path: PathLike,
    sequence_name: str,
    *,
    data_root: Optional[PathLike] = None,
    num_frames: int = 64,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """Verify that the JSON loader matches the NPZ loader and the ground truth."""
    seq_json = load_sequence_from_json(
        json_path, sequence_name, data_root=data_root, num_frames=num_frames, load_rgb=False
    )
    seq_npz = load_sequence_from_npz(npz_path, num_frames=num_frames, load_rgb=False)

    report: Dict[str, Any] = {
        "sequence": sequence_name,
        "num_frames": seq_json.num_frames,
        "num_tracks": seq_json.num_tracks,
        "num_queries_json": int(seq_json.num_queries),
        "num_queries_npz": int(seq_npz.num_queries),
    }

    def _max_diff(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")

    report["tracks_xyz_cam_max_diff"] = _max_diff(seq_json.tracks_xyz_cam, seq_npz.tracks_xyz_cam)
    report["visibility_equal"] = bool(np.array_equal(seq_json.visibility, seq_npz.visibility))
    report["gt_tracks_world_max_diff"] = _max_diff(seq_json.gt_tracks_world, seq_npz.gt_tracks_world)
    report["query_indices_equal"] = bool(np.array_equal(seq_json.query_indices, seq_npz.query_indices))

    report["aligned"] = (
        report["tracks_xyz_cam_max_diff"] < atol
        and report["visibility_equal"]
        and report["gt_tracks_world_max_diff"] < atol
        and report["query_indices_equal"]
    )
    return report
