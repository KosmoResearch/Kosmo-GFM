"""Metrics for the WorldTrack evaluation protocol.

The two tasks are deliberately kept in separate modules so that a Dynamic Points
result can never be mixed with a Scene Flow result:

* :mod:`eval4d.metrics.dynamic_points` -- 3D tracking (APD / tau@0.1m / EPE).
* :mod:`eval4d.metrics.scene_flow` -- displacement flow (tau@0.1m / flow EPE).

:mod:`eval4d.metrics.scene_flow` does **not** import
:mod:`eval4d.metrics.dynamic_points`; the shared thresholds and the scale
alignment live in :mod:`eval4d.metrics.constants` and
:mod:`eval4d.metrics.alignment` instead.
"""

from __future__ import annotations

from .alignment import (
    compute_average_pts_within_thresh,
    compute_scale_factor_global,
    estimate_sim3_closed_form,
    estimate_sim3_ransac,
    finite_correspondence_pairs,
    scale_per_trajectory,
)
from .constants import (
    APD_THRESHOLDS,
    DEFAULT_DYNAMIC_THRESHOLD_M,
    DP_APD_GLOBAL_KEY,
    DP_EPE_GLOBAL_KEY,
    DP_TAU_GLOBAL_KEY,
    DP_TAU_GLOBAL_DYN_KEY,
    SCALE_SOURCE_ALL_EMPTY,
    SCALE_SOURCE_ALL_POSITIONS,
    SF_APD_DYN_KEY,
    SF_APD_GLOBAL_KEY,
    SF_APD_PRED_DYN_KEY,
    SF_DYN_FLOW_COUNT_KEY,
    SF_EPE_DYN_KEY,
    SF_EPE_GLOBAL_KEY,
    SF_EPE_PRED_DYN_KEY,
    TAU_M,
)

__all__ = [
    "APD_THRESHOLDS",
    "DEFAULT_DYNAMIC_THRESHOLD_M",
    "DP_APD_GLOBAL_KEY",
    "DP_EPE_GLOBAL_KEY",
    "DP_TAU_GLOBAL_KEY",
    "DP_TAU_GLOBAL_DYN_KEY",
    "SCALE_SOURCE_ALL_EMPTY",
    "SCALE_SOURCE_ALL_POSITIONS",
    "SF_APD_DYN_KEY",
    "SF_APD_GLOBAL_KEY",
    "SF_APD_PRED_DYN_KEY",
    "SF_DYN_FLOW_COUNT_KEY",
    "SF_EPE_DYN_KEY",
    "SF_EPE_GLOBAL_KEY",
    "SF_EPE_PRED_DYN_KEY",
    "TAU_M",
    "compute_average_pts_within_thresh",
    "compute_scale_factor_global",
    "estimate_sim3_closed_form",
    "estimate_sim3_ransac",
    "finite_correspondence_pairs",
    "scale_per_trajectory",
]
