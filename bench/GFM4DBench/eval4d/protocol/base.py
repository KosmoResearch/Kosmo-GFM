"""Shared scaffolding for the evaluation tasks.

A *task* owns everything that is specific to one evaluation flavour: how a
prediction is turned into metrics, how per-sequence results are aggregated, and
how a summary is rendered. The runner and the CLI stay task-agnostic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

__all__ = [
    "BaseTask",
    "PredictionLike",
    "finalize_sequence_metrics",
    "resolve_dynamic_threshold",
]


class PredictionLike(Protocol):
    """The prediction fields a task may consume.

    ``tracks_ref0`` holds the absolute 3D positions in the frame-0 reference
    frame; ``flow_ref0`` holds the per-frame displacements relative to frame 0.
    A task only reads the fields it needs.
    """

    tracks_ref0: Optional[np.ndarray]
    flow_ref0: Optional[np.ndarray]



def resolve_dynamic_threshold(dynamic_threshold: Optional[float]) -> float:
    """Return the motion threshold to use, falling back to the protocol default."""
    from ..metrics.constants import DEFAULT_DYNAMIC_THRESHOLD_M

    if dynamic_threshold is None:
        return DEFAULT_DYNAMIC_THRESHOLD_M
    return float(dynamic_threshold)


def finalize_sequence_metrics(
    metrics: Dict[str, Any],
    *,
    sequence_name: str,
    num_queries: Optional[int] = None,
    num_queries_json: Optional[int] = None,
    num_frames: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach the per-sequence bookkeeping fields to a metric payload.

    ``num_queries`` is the number of queries that were actually scored; it
    overrides the value computed inside the metrics module so that the summary
    counts exactly the queries that were evaluated.
    """
    out = dict(metrics)
    if num_queries is not None:
        out["num_queries"] = int(num_queries)
    out["video_name"] = str(sequence_name)
    if num_queries_json is not None:
        out["num_queries_json"] = int(num_queries_json)
    if num_frames is not None:
        out["num_frames_eff"] = int(num_frames)
    if extra:
        out.update(dict(extra))
    return out


class BaseTask:
    """Interface implemented by the Dynamic Points and Scene Flow tasks."""

    #: Stable identifier used on the command line.
    name: str = ""
    #: The summary keys that define the reported result of this task.
    primary_keys: Tuple[str, ...] = ()
    #: Whether the task accepts the ``eval_dynamic_only`` scoring option.
    accepts_dynamic_only: bool = False

    def score_sequence(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Score one sequence."""
        raise NotImplementedError

    def score_prediction(
        self,
        gt_tracks_world: np.ndarray,
        prediction: PredictionLike,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Score one sequence from a prediction container.

        The container is duck-typed: it only has to expose ``tracks_ref0`` and
        ``flow_ref0``.
        """
        raise NotImplementedError

    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate per-sequence results into a subset summary."""
        raise NotImplementedError

    def format_summary(self, subset: str, summary: Dict[str, Any]) -> str:
        """Render a one-line summary of a subset."""
        raise NotImplementedError

    def primary_values(self, summary: Mapping[str, Any]) -> Tuple[float, ...]:
        """Extract the primary metric values from a summary, in key order."""
        values: List[float] = []
        for key in self.primary_keys:
            raw = summary.get(key, float("nan"))
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(float("nan"))
        return tuple(values)

    @staticmethod
    def mean_of_finite(values: Sequence[float]) -> float:
        arr = np.asarray(list(values), dtype=np.float64)
        if arr.size == 0 or not np.any(np.isfinite(arr)):
            return float("nan")
        return float(np.mean(arr[np.isfinite(arr)]))
