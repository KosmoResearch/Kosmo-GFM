"""Data access for the WorldTrack evaluation protocol.

This subpackage depends only on numpy (and OpenCV, imported lazily where it is
needed to read images). It never imports torch.
"""

from __future__ import annotations

from .conventions import (
    DEFAULT_COORD_CONVENTION,
    CoordConvention,
    extrinsics_c2w_to_w2c,
    normalize_coord_convention,
    normalize_mf_tracks_to_opend4rt,
    normalize_uv_to_01,
    pointodyssey_cam_to_ref0_world,
    pointodyssey_world_to_ref0_world,
    project_cam_to_uv,
    ref0_to_cam_tracks,
    tracks_cam_to_ref0_world,
)
from .queries import (
    assert_query_uv_in_image_bounds,
    compute_frame0_query_indices,
    compute_frame0_query_indices_from_uv_and_depth,
    compute_opend4rt_frame0_query_indices,
    filter_query_indices_by_dynamic_mask,
    max_displacement_mask,
    opend4rt_query_count_from_arrays,
)
from .sequences import (
    PathLike,
    WorldTrackSequence,
    compare_json_vs_npz,
    dynamic_query_mask_for_sequence,
    filter_sequences_with_frame0_queries,
    frame0_query_indices_for_sequence,
    list_sequences_in_json,
    load_mf_files,
    load_sequence_from_json,
    load_sequence_from_npz,
    prefer_worldtrack_json_path,
    read_sequence_frame0_image_hw,
    resolve_opend4rt_frame_count,
    resolve_worldtrack_path,
)

__all__ = [
    "DEFAULT_COORD_CONVENTION",
    "CoordConvention",
    "PathLike",
    "WorldTrackSequence",
    "assert_query_uv_in_image_bounds",
    "compare_json_vs_npz",
    "compute_frame0_query_indices",
    "compute_frame0_query_indices_from_uv_and_depth",
    "compute_opend4rt_frame0_query_indices",
    "dynamic_query_mask_for_sequence",
    "extrinsics_c2w_to_w2c",
    "filter_sequences_with_frame0_queries",
    "filter_query_indices_by_dynamic_mask",
    "frame0_query_indices_for_sequence",
    "list_sequences_in_json",
    "load_mf_files",
    "load_sequence_from_json",
    "load_sequence_from_npz",
    "max_displacement_mask",
    "normalize_coord_convention",
    "normalize_mf_tracks_to_opend4rt",
    "normalize_uv_to_01",
    "opend4rt_query_count_from_arrays",
    "pointodyssey_cam_to_ref0_world",
    "pointodyssey_world_to_ref0_world",
    "prefer_worldtrack_json_path",
    "project_cam_to_uv",
    "read_sequence_frame0_image_hw",
    "ref0_to_cam_tracks",
    "resolve_opend4rt_frame_count",
    "resolve_worldtrack_path",
    "tracks_cam_to_ref0_world",
]
