"""``eval4d`` -- evaluation protocol for 4D point tracking and scene flow.

The package implements two distinct evaluation tasks on the WorldTrack /
TapVid3D benchmark:

**Dynamic Points** (:mod:`eval4d.metrics.dynamic_points`)
    3D tracking of frame-0 queries. The model predicts the 3D position of every
    query in the frame-0 reference frame. Reported as APD (``avg_pts_global``),
    tau@0.1m (``tau_global``) and EPE (``epe_global``).

**Scene Flow** (:mod:`eval4d.metrics.scene_flow`)
    Per-frame displacement of every query relative to frame 0. Reported as
    tau@0.1m (``avg_sf_global``) and flow EPE (``epe_sf_global``).

The two tasks share the same data loading and query-selection rules but **not**
their metrics. Keep them apart: a Dynamic Points number can never be read off a
Scene Flow summary, and vice versa.

Typical use::

    from eval4d import DynamicPointsTask, load_sequence_from_json

    task = DynamicPointsTask()
    seq = load_sequence_from_json(json_path, "Apartment_release_clean_seq131_0",
                                  data_root=data_root)
    metrics = task.score_sequence(seq, pred_tracks_ref0)
"""

from __future__ import annotations

__version__ = "0.1.0"

from .data import (
    WorldTrackSequence,
    list_sequences_in_json,
    load_sequence_from_json,
    load_sequence_from_npz,
)
from .io import SequencePrediction, load_prediction, save_prediction, score_prediction
from .metrics.constants import APD_THRESHOLDS, TAU_M
from .protocol.dynamic_points import DYNAMIC_POINTS_PRIMARY_KEYS, DynamicPointsTask
from .protocol.scene_flow import SCENE_FLOW_PRIMARY_KEYS, SceneFlowTask

__all__ = [
    "__version__",
    "APD_THRESHOLDS",
    "TAU_M",
    "DYNAMIC_POINTS_PRIMARY_KEYS",
    "SCENE_FLOW_PRIMARY_KEYS",
    "DynamicPointsTask",
    "SceneFlowTask",
    "SequencePrediction",
    "WorldTrackSequence",
    "list_sequences_in_json",
    "load_prediction",
    "load_sequence_from_json",
    "load_sequence_from_npz",
    "save_prediction",
    "score_prediction",
]
