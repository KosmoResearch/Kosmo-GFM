"""Shared metric constants for the WorldTrack evaluation protocol.

Keeping the thresholds and summary key names in one module lets
:mod:`eval4d.metrics.dynamic_points` and :mod:`eval4d.metrics.scene_flow` stay
independent of each other: the two tasks must never share a metric module.
"""

from __future__ import annotations

from typing import Dict

# TAPVid-style threshold keys -> metric thresholds in metres.
# The key is the TAPVid pixel-distance bin index; the value is the metric threshold.
APD_THRESHOLDS: Dict[int, float] = {
    1: 0.1,
    2: 0.3,
    4: 0.5,
    8: 1.0,
}

# tau threshold (metres). ``fractions_global[1]`` uses ``APD_THRESHOLDS[1]``.
TAU_M: float = 0.1

# --- Dynamic Points summary keys ---------------------------------------------
DP_APD_GLOBAL_KEY = "avg_pts_global"
DP_TAU_GLOBAL_KEY = "tau_global"
DP_EPE_GLOBAL_KEY = "epe_global"
DP_TAU_GLOBAL_DYN_KEY = "tau_global_dyn"

# --- Scene Flow summary keys --------------------------------------------------
SF_APD_GLOBAL_KEY = "avg_sf_global"
SF_EPE_GLOBAL_KEY = "epe_sf_global"
# GT-dynamic query subset: scene-flow APD/EPE (same global scale, no re-fit).
SF_APD_DYN_KEY = "avg_sf_global_dyn"
SF_EPE_DYN_KEY = "epe_sf_global_dyn"
SF_APD_PRED_DYN_KEY = "avg_sf_pred_dyn"
SF_EPE_PRED_DYN_KEY = "epe_sf_pred_dyn"
SF_DYN_FLOW_COUNT_KEY = "dyn_flow_count"

# Scale-alignment provenance strings.
SCALE_SOURCE_ALL_POSITIONS = "all_positions_vs_gt_trajs"
SCALE_SOURCE_ALL_EMPTY = "all_positions_empty"

# Default motion threshold (metres) separating dynamic from static queries.
DEFAULT_DYNAMIC_THRESHOLD_M: float = 0.01
