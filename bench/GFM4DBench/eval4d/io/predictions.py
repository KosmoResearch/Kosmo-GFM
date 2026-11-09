"""Prediction container and offline scoring.

A prediction for one sequence is a pair of arrays in the frame-0 reference
frame:

* ``tracks`` -- ``[T, Q, 3]`` absolute 3D positions (the **Dynamic Points**
  prediction).
* ``flow`` -- ``[T, Q, 3]`` per-frame displacements relative to frame 0 (the
  **Scene Flow** prediction), with ``flow[0] == 0``.

Storing predictions on disk decouples scoring from inference, which makes it
possible to re-score a model without a GPU and to diff two implementations
against each other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

__all__ = [
    "SequencePrediction",
    "save_prediction",
    "load_prediction",
    "write_predictions",
    "read_predictions",
    "score_prediction",
]

# Field names inside the saved ``.npz`` archive.
TRACKS_KEY = "tracks_ref0"
FLOW_KEY = "flow_ref0"


@dataclass
class SequencePrediction:
    """Predictions for a single sequence, in the frame-0 reference frame."""

    sequence_name: str
    tracks_ref0: Optional[np.ndarray] = None  # [T, Q, 3]
    flow_ref0: Optional[np.ndarray] = None  # [T, Q, 3]
    query_indices: Optional[np.ndarray] = None  # [Q], original GT track identities

    def __post_init__(self) -> None:
        if self.query_indices is not None:
            self.query_indices = np.asarray(self.query_indices, dtype=np.int64)
        if self.tracks_ref0 is not None:
            self.tracks_ref0 = np.asarray(self.tracks_ref0, dtype=np.float64)
        if self.flow_ref0 is not None:
            self.flow_ref0 = np.asarray(self.flow_ref0, dtype=np.float64)

    @property
    def num_frames(self) -> int:
        for arr in (self.tracks_ref0, self.flow_ref0):
            if arr is not None:
                return int(arr.shape[0])
        return 0

    @property
    def num_queries(self) -> int:
        for arr in (self.tracks_ref0, self.flow_ref0):
            if arr is not None:
                return int(arr.shape[1])
        return 0

    def to_npz_dict(self) -> Dict[str, np.ndarray]:
        payload: Dict[str, np.ndarray] = {"sequence_name": np.asarray(self.sequence_name)}
        if self.tracks_ref0 is not None:
            payload[TRACKS_KEY] = self.tracks_ref0
        if self.flow_ref0 is not None:
            payload[FLOW_KEY] = self.flow_ref0
        if self.query_indices is not None:
            payload["query_indices"] = self.query_indices
        return payload


def save_prediction(path: Any, prediction: SequencePrediction) -> Path:
    """Write one prediction to a compressed ``.npz`` archive."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **prediction.to_npz_dict())
    return out


def load_prediction(path: Any) -> SequencePrediction:
    """Read one prediction from a ``.npz`` archive."""
    src = Path(path)
    with np.load(src, allow_pickle=False) as pack:
        name = str(pack["sequence_name"]) if "sequence_name" in pack.files else src.stem
        tracks = pack[TRACKS_KEY] if TRACKS_KEY in pack.files else None
        flow = pack[FLOW_KEY] if FLOW_KEY in pack.files else None
        indices = pack["query_indices"] if "query_indices" in pack.files else None
    return SequencePrediction(
        sequence_name=name,
        tracks_ref0=tracks,
        flow_ref0=flow,
        query_indices=indices,
    )


def write_predictions(
    directory: Any,
    predictions: Mapping[str, SequencePrediction],
) -> Path:
    """Write a mapping ``sequence name -> prediction`` into ``directory``."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, pred in predictions.items():
        save_prediction(out_dir / f"{name}.npz", pred)
    return out_dir


def read_predictions(directory: Any) -> Dict[str, SequencePrediction]:
    """Read every ``.npz`` prediction in ``directory``."""
    out_dir = Path(directory)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"Not a prediction directory: {out_dir}")
    preds: Dict[str, SequencePrediction] = {}
    for path in sorted(out_dir.glob("*.npz")):
        preds[path.stem] = load_prediction(path)
    return preds


def score_prediction(
    prediction: SequencePrediction,
    gt_tracks_world: np.ndarray,
    *,
    task: str = "dynamic-points",
    dynamic_threshold: Optional[float] = None,
    dynamic_only: bool = False,
) -> Dict[str, Any]:
    """Score one prediction against the GT trajectories.

    Args:
        task: ``dynamic-points`` or ``scene-flow``.
        gt_tracks_world: ``[T, Q, 3]`` ground truth in the frame-0 reference frame.
        dynamic_threshold: Motion threshold (m) separating dynamic from static
            queries. Defaults to the protocol value of 0.01 m.
        dynamic_only: Score only the dynamic queries (the query set must already
            be filtered for this to be meaningful).
    """
    task_key = str(task).strip().lower().replace("_", "-")
    if task_key not in ("dynamic-points", "scene-flow"):
        raise ValueError(f"Unknown task {task!r}; use 'dynamic-points' or 'scene-flow'.")

    from ..metrics.constants import DEFAULT_DYNAMIC_THRESHOLD_M

    thr = DEFAULT_DYNAMIC_THRESHOLD_M if dynamic_threshold is None else float(dynamic_threshold)
    gt = np.asarray(gt_tracks_world, dtype=np.float64)

    if task_key == "dynamic-points":
        from ..metrics.dynamic_points import metrics_for_sequence

        if prediction.tracks_ref0 is None:
            raise ValueError("The prediction has no 'tracks_ref0' array.")
        return metrics_for_sequence(
            gt_tracks_world=gt,
            pred_tracks_ref0=prediction.tracks_ref0,
            compute_dyn=True,
            dynamic_threshold=thr,
            eval_dynamic_only=dynamic_only,
        )

    from ..metrics.scene_flow import metrics_for_sequence_scene_flow

    if prediction.flow_ref0 is None:
        raise ValueError("The prediction has no 'flow_ref0' array.")
    return metrics_for_sequence_scene_flow(
        gt_tracks_world=gt,
        pred_flow_ref0=prediction.flow_ref0,
        pred_tracks_ref0=prediction.tracks_ref0,
        compute_dyn=True,
        dynamic_threshold=thr,
    )


def write_summary_json(path: Any, payload: Any) -> Path:
    """Write a summary payload as JSON.

    ``NaN`` values are emitted as bare ``NaN`` tokens so that existing
    downstream parsers keep working.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def iter_prediction_files(directory: Any) -> Iterable[Path]:
    """Iterate over the prediction archives in ``directory``."""
    out_dir = Path(directory)
    for path in sorted(out_dir.glob("*.npz")):
        yield path
