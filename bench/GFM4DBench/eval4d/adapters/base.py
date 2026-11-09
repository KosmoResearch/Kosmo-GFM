"""Model adapters.

An adapter bridges a specific model implementation to the evaluation protocol.
The protocol itself never imports a model framework: it consumes predictions in
the frame-0 reference frame and compares them against the ground truth.

Implement :class:`PredictionAdapter` to plug a model into the runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..data.sequences import WorldTrackSequence

__all__ = ["PredictionAdapter", "AdapterPrediction", "align_batch_queries_with_sequence"]


class AdapterPrediction:
    """The output of an adapter for a single sequence.

    ``tracks_ref0`` holds absolute 3D positions in the frame-0 reference frame
    and ``flow_ref0`` holds per-frame displacements relative to frame 0. An
    adapter may fill either or both; the task decides which one it consumes.
    """

    __slots__ = ("tracks_ref0", "flow_ref0", "extra")

    def __init__(
        self,
        tracks_ref0: Optional[np.ndarray] = None,
        flow_ref0: Optional[np.ndarray] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.tracks_ref0 = None if tracks_ref0 is None else np.asarray(tracks_ref0, dtype=np.float64)
        self.flow_ref0 = None if flow_ref0 is None else np.asarray(flow_ref0, dtype=np.float64)
        self.extra: Dict[str, Any] = dict(extra or {})


@runtime_checkable
class PredictionAdapter(Protocol):
    """The interface the runner expects from a model adapter.

    Implementations own everything model-specific: building the input dataset,
    loading the checkpoint, running inference, and turning the raw model output
    into frame-0 reference-frame predictions.
    """

    name: str

    def build_dataset(
        self,
        json_path: Any,
        data_root: Any,
        config: Dict[str, Any],
        num_frames: int,
        sequence_names: Optional[list] = None,
        coord_convention: str = "opend4rt",
        dynamic_query_threshold: Optional[float] = None,
    ) -> Any:
        """Build the input dataset for one subset."""
        ...

    def load_model(self, config: Dict[str, Any], checkpoint_path: Any, device: Any) -> Any:
        """Instantiate the model and load the checkpoint."""
        ...

    def load_config(self, config_path: Any) -> Dict[str, Any]:
        """Read the model configuration file."""
        ...

    def infer(self, model: Any, batch: Dict[str, Any], mixed_precision: str = "fp16") -> Any:
        """Run the model on one batch."""
        ...

    def extract(
        self,
        outputs: Any,
        *,
        num_frames: int,
        num_queries: int,
    ) -> AdapterPrediction:
        """Turn the raw model output into reference-frame predictions."""
        ...


def align_batch_queries_with_sequence(
    batch: Dict[str, Any],
    sequence: "WorldTrackSequence",
    *,
    sequence_name: str = "",
) -> np.ndarray:
    """Make the model input use exactly the evaluation query set.

    - The batch query indices must match ``sequence.query_indices``; otherwise
      the model would be evaluated on a different query set than the metrics.
    - The frame-0 query UVs are injected into the batch so that inference does
      not depend on the on-disk ``trajs_2d`` files.

    The batch is mutated in place; it is expected to be a plain ``dict`` that
    may hold torch tensors (they are converted with ``.detach().cpu().numpy()``
    when needed).
    """
    from ..data.queries import assert_query_uv_in_image_bounds

    raw = batch.get("worldtrack_query_indices")
    if raw is None:
        query_idx = np.asarray(sequence.query_indices, dtype=np.int64)
    else:
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        query_idx = np.asarray(raw, dtype=np.int64)
        if query_idx.ndim == 2:
            query_idx = query_idx[0]
        query_idx = query_idx.reshape(-1)

    if not np.array_equal(query_idx, sequence.query_indices):
        label = sequence_name or sequence.video_name
        raise RuntimeError(
            f"{label}: dataset query_indices ({query_idx.size}) != "
            f"JSON loader ({sequence.num_queries}); the model and the metrics "
            f"would be evaluated on different query sets."
        )

    height, width = sequence.image_hw
    assert_query_uv_in_image_bounds(sequence.query_uv, (height, width))
    batch["worldtrack_query_uv"] = np.asarray(sequence.query_uv, dtype=np.float32)
    batch["worldtrack_query_indices"] = query_idx.copy()
    batch["worldtrack_num_queries"] = int(query_idx.size)
    batch["worldtrack_num_frames"] = int(sequence.num_frames)
    return query_idx
