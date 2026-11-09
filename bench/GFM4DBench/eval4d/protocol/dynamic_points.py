"""The Dynamic Points task: 3D tracking of frame-0 queries.

The prediction is the absolute 3D position of every frame-0 query in the
frame-0 reference frame, for every frame of the clip.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..metrics.constants import DEFAULT_DYNAMIC_THRESHOLD_M
from ..metrics.dynamic_points import (
    DYNAMIC_POINTS_PRIMARY_KEYS,
    aggregate_results,
    format_subset_summary,
    metrics_for_sequence,
)
from .base import BaseTask, PredictionLike, finalize_sequence_metrics, resolve_dynamic_threshold

__all__ = ["DynamicPointsTask", "DYNAMIC_POINTS_PRIMARY_KEYS"]


class DynamicPointsTask(BaseTask):
    """Score predictions against the Dynamic Points protocol."""

    name = "dynamic-points"
    primary_keys = DYNAMIC_POINTS_PRIMARY_KEYS
    accepts_dynamic_only = True

    def score_sequence(
        self,
        gt_tracks_world: np.ndarray,
        pred_tracks_ref0: np.ndarray,
        *,
        sequence_name: str = "",
        num_queries: Optional[int] = None,
        num_queries_json: Optional[int] = None,
        num_frames: Optional[int] = None,
        dynamic_threshold: Optional[float] = None,
        eval_dynamic_only: bool = False,
        compute_dyn: bool = True,
        rng: Optional[np.random.Generator] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Score one sequence.

        Args:
            gt_tracks_world: ``[T, Q, 3]`` ground truth in the ref0 world frame.
            pred_tracks_ref0: ``[T, Q, 3]`` predicted 3D positions in the same frame.
            num_queries: Number of queries actually scored; overrides the value
                computed by the metrics module.
            eval_dynamic_only: Fit the scale on the dynamic queries only.
        """
        thr = resolve_dynamic_threshold(dynamic_threshold)
        metrics = metrics_for_sequence(
            gt_tracks_world=np.asarray(gt_tracks_world, dtype=np.float64),
            pred_tracks_ref0=np.asarray(pred_tracks_ref0, dtype=np.float64),
            compute_dyn=compute_dyn,
            dynamic_threshold=thr,
            eval_dynamic_only=bool(eval_dynamic_only),
            rng=rng,
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
        if getattr(prediction, "tracks_ref0", None) is None:
            raise ValueError(
                "The Dynamic Points task needs a 'tracks_ref0' prediction; "
                "a scene-flow-only prediction cannot be scored with it."
            )
        return self.score_sequence(
            gt_tracks_world,
            prediction.tracks_ref0,
            **kwargs,
        )

    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mean-pool the per-sequence Dynamic Points metrics."""
        return aggregate_results(results)

    def format_summary(self, subset: str, summary: Dict[str, Any]) -> str:
        """Render a one-line Dynamic Points summary."""
        return format_subset_summary(subset, summary)

    @staticmethod
    def default_dynamic_threshold() -> float:
        """Protocol default motion threshold (metres)."""
        return DEFAULT_DYNAMIC_THRESHOLD_M
