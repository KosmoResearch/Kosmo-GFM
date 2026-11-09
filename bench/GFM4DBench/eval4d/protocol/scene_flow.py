"""The Scene Flow task: per-frame displacement of frame-0 queries.

The prediction is the displacement of every frame-0 query relative to its
frame-0 position, for every frame of the clip. ``flow[0]`` is zero by
construction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..metrics.constants import DEFAULT_DYNAMIC_THRESHOLD_M
from ..metrics.scene_flow import (
    SCENE_FLOW_PRIMARY_KEYS,
    aggregate_results,
    format_subset_summary,
    metrics_for_sequence_scene_flow,
)
from .base import BaseTask, PredictionLike, finalize_sequence_metrics, resolve_dynamic_threshold

__all__ = ["SceneFlowTask", "SCENE_FLOW_PRIMARY_KEYS"]


class SceneFlowTask(BaseTask):
    """Score predictions against the Scene Flow protocol."""

    name = "scene-flow"
    primary_keys = SCENE_FLOW_PRIMARY_KEYS

    def score_sequence(
        self,
        gt_tracks_world: np.ndarray,
        pred_flow_ref0: np.ndarray,
        *,
        pred_tracks_ref0: Optional[np.ndarray] = None,
        sequence_name: str = "",
        num_queries: Optional[int] = None,
        num_queries_json: Optional[int] = None,
        num_frames: Optional[int] = None,
        dynamic_threshold: Optional[float] = None,
        compute_dyn: bool = True,
        fixed_scale: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Score one sequence.

        Args:
            gt_tracks_world: ``[T, Q, 3]`` ground truth in the ref0 world frame.
            pred_flow_ref0: ``[T, Q, 3]`` predicted displacement flow.
            pred_tracks_ref0: Optional ``[T, Q, 3]`` predicted absolute positions.
                These are used to estimate the global scale; when omitted the flow is
                anchored on the GT frame-0 positions.
        """
        thr = resolve_dynamic_threshold(dynamic_threshold)
        metrics = metrics_for_sequence_scene_flow(
            gt_tracks_world=np.asarray(gt_tracks_world, dtype=np.float64),
            pred_flow_ref0=np.asarray(pred_flow_ref0, dtype=np.float64),
            pred_tracks_ref0=(
                None
                if pred_tracks_ref0 is None
                else np.asarray(pred_tracks_ref0, dtype=np.float64)
            ),
            compute_dyn=compute_dyn,
            dynamic_threshold=thr,
            fixed_scale=fixed_scale,
        )
        return finalize_sequence_metrics(
            metrics,
            sequence_name=sequence_name,
            num_queries=num_queries,
            num_queries_json=num_queries_json,
            num_frames=num_frames,
            extra=extra,
        )

    def score_prediction(
        self,
        gt_tracks_world: np.ndarray,
        prediction: PredictionLike,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Score one sequence from a prediction container."""
        if getattr(prediction, "flow_ref0", None) is None:
            raise ValueError(
                "The Scene Flow task needs a 'flow_ref0' prediction; "
                "an absolute-trajectory-only prediction cannot be scored with it."
            )
        kwargs.setdefault("pred_tracks_ref0", getattr(prediction, "tracks_ref0", None))
        return self.score_sequence(
            gt_tracks_world,
            prediction.flow_ref0,
            **kwargs,
        )

    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mean-pool the per-sequence Scene Flow metrics."""
        return aggregate_results(results)

    def format_summary(self, subset: str, summary: Dict[str, Any]) -> str:
        """Render a one-line Scene Flow summary."""
        return format_subset_summary(subset, summary)

    @staticmethod
    def default_dynamic_threshold() -> float:
        """Protocol default motion threshold (metres)."""
        return DEFAULT_DYNAMIC_THRESHOLD_M
