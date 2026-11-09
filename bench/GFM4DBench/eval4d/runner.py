"""The evaluation loop.

The runner owns the parts that are identical for every task and every model:
building the input dataset, running inference, extracting predictions, writing
per-sequence results, and aggregating them.

It deliberately imports **no metric module**. Scoring is delegated to a task
object (see :mod:`eval4d.protocol`), which keeps the task-specific semantics
(including the differences between Dynamic Points and Scene Flow) in one
place.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .adapters.base import PredictionAdapter, align_batch_queries_with_sequence
from .data.sequences import (
    filter_sequences_with_frame0_queries,
    list_sequences_in_json,
    load_sequence_from_json,
)
from .io.predictions import SequencePrediction, save_prediction, write_summary_json
from .protocol.base import BaseTask
from .registry import SubsetSpec

__all__ = [
    "SubsetSpec",
    "RunConfig",
    "evaluate_subset",
    "run_evaluation",
    "score_offline_predictions",
]


@dataclass
class RunConfig:
    """Options that control how a run is executed."""

    output_dir: Path = Path("tmp/eval4d")
    num_frames: int = 64
    limit_seqs: int = 0
    seq_name: str = ""
    mixed_precision: str = "fp16"
    device: str = "cuda"
    save_per_sequence: bool = True
    save_predictions: bool = True
    resume: bool = False
    #: Render an overlay video of the ground truth and the prediction.
    visualize: bool = False
    vis_fps: float = 15.0
    vis_max_points: int = 300
    #: Score only the dynamic queries (and fit the scale on them). Defaults to
    #: ``True`` when the subset filters its queries by a dynamic threshold.
    eval_dynamic_only: Optional[bool] = None
    #: Motion threshold used by the metrics (metres).
    dynamic_threshold: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _sequence_filter(
    json_path: Path,
    *,
    limit_seqs: int,
    seq_name: str,
) -> List[str]:
    """The sequences of a subset that this run should touch."""
    names = list_sequences_in_json(json_path)
    if seq_name:
        if seq_name not in names:
            raise ValueError(f"Sequence {seq_name!r} not in {json_path}")
        return [seq_name]
    if limit_seqs and int(limit_seqs) > 0:
        return names[: int(limit_seqs)]
    return names


def _metric_dynamic_threshold(spec: SubsetSpec, config: RunConfig) -> float:
    """The motion threshold the metrics use (metres)."""
    if config.dynamic_threshold is not None:
        return float(config.dynamic_threshold)
    if spec.dynamic_query_threshold is not None:
        return float(spec.dynamic_query_threshold)
    return 0.01


def _uses_dynamic_only(spec: SubsetSpec, config: RunConfig) -> bool:
    """Whether only the dynamic queries are scored.

    A subset that filters its queries by a dynamic threshold is scored on that
    filtered set, and the scale is re-fitted on it.
    """
    if config.eval_dynamic_only is not None:
        return bool(config.eval_dynamic_only)
    return spec.dynamic_query_threshold is not None


def _load_ground_truth(
    spec: SubsetSpec,
    sequence_name: str,
    config: RunConfig,
    *,
    load_rgb: bool = False,
) -> Any:
    """Load one sequence's ground truth and its frame-0 query set."""
    return load_sequence_from_json(
        spec.json_path,
        sequence_name,
        data_root=spec.resolved_data_root(),
        num_frames=config.num_frames,
        load_rgb=load_rgb,
        coord_convention=spec.coord_convention,
        dynamic_query_threshold=spec.dynamic_query_threshold,
    )


def _write_overlay(
    *,
    spec: SubsetSpec,
    sequence_name: str,
    config: RunConfig,
    query_idx: np.ndarray,
    num_frames_eff: int,
    pred_ref0: Optional[np.ndarray],
    subset_out: Path,
) -> Optional[Path]:
    """Render the ground truth and the prediction into an overlay video."""
    from .viz import GT_COLOR, PRED_COLOR, project_ref0_to_uv, save_overlay_video

    sequence = _load_ground_truth(spec, sequence_name, config, load_rgb=True)
    positions = np.arange(len(query_idx))
    keep = query_idx
    if config.vis_max_points and len(query_idx) > int(config.vis_max_points):
        step = int(np.ceil(len(query_idx) / float(config.vis_max_points)))
        positions = positions[::step]
        keep = query_idx[positions]

    gt_uv = sequence.tracks_uv[:num_frames_eff, keep]
    layers: List[Dict[str, Any]] = [
        {
            "uv": gt_uv,
            "visibility": sequence.visibility[:num_frames_eff, keep],
            "color": GT_COLOR,
        }
    ]
    if pred_ref0 is not None:
        layers.append(
            {
                "uv": project_ref0_to_uv(
                    np.asarray(pred_ref0, dtype=np.float64)[:, positions],
                    sequence.extrinsics_w2c[:num_frames_eff],
                    sequence.intrinsics,
                ),
                "color": PRED_COLOR,
            }
        )

    out_dir = subset_out / sequence_name / "vis"
    return save_overlay_video(
        sequence.video_rgb[:num_frames_eff],
        layers,
        out_dir / "tracks.mp4",
        fps=config.vis_fps,
    )


def _ref0_epe(pred_ref0: np.ndarray, gt_ref0: np.ndarray) -> float:
    """Mean 3D error in the reference frame, before the scale alignment."""
    fin = np.isfinite(pred_ref0).all(axis=-1) & np.isfinite(gt_ref0).all(axis=-1)
    if not np.any(fin):
        return float("nan")
    return float(np.linalg.norm(pred_ref0[fin] - gt_ref0[fin], axis=-1).mean())


def evaluate_subset(
    *,
    task: BaseTask,
    adapter: PredictionAdapter,
    model: Any,
    dataset: Any,
    spec: SubsetSpec,
    config: RunConfig,
) -> Dict[str, Any]:
    """Evaluate one subset and return its summary.

    The dataset is built by the caller so that a checkpoint is loaded only
    once for all subsets.
    """
    from torch.utils.data import DataLoader

    collate_fn = dataset.get_collate_fn() if hasattr(dataset, "get_collate_fn") else None
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    subset_out = Path(config.output_dir) / spec.name
    subset_out.mkdir(parents=True, exist_ok=True)

    seq_names = _sequence_filter(
        spec.json_path,
        limit_seqs=config.limit_seqs,
        seq_name=config.seq_name,
    )

    valid_names, skipped_no_query = filter_sequences_with_frame0_queries(
        spec.json_path,
        seq_names,
        data_root=spec.resolved_data_root(),
        num_frames=config.num_frames,
        coord_convention=spec.coord_convention,
        dynamic_query_threshold=spec.dynamic_query_threshold,
    )
    if skipped_no_query:
        logging.warning(
            "Skipping %d sequences with no frame-0 visible queries: %s",
            len(skipped_no_query),
            ", ".join(skipped_no_query),
        )
        write_summary_json(
            subset_out / "skipped_no_frame0_queries.json",
            {"sequences": skipped_no_query},
        )
    seq_names = valid_names

    results: List[Dict[str, Any]] = []
    if config.resume:
        pending = []
        for name in seq_names:
            result_path = subset_out / f"{name}.json"
            prediction_path = subset_out / "predictions" / f"{name}.npz"
            if result_path.is_file() and (not config.save_predictions or prediction_path.is_file()):
                results.append(json.loads(result_path.read_text(encoding="utf-8")))
            else:
                pending.append(name)
        logging.info("Resume: reusing %d completed sequences", len(results))
        seq_names = pending

    if hasattr(dataset, "sequence_names") and dataset.sequence_names is not None:
        dataset.sequence_names = list(seq_names)
        if hasattr(dataset, "get_data_infos"):
            dataset.data_infos = dataset.get_data_infos()

    dynamic_only = _uses_dynamic_only(spec, config)
    threshold = _metric_dynamic_threshold(spec, config)
    pred_dir = subset_out / "predictions"

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= len(seq_names):
            break
        sequence_name = seq_names[batch_idx]

        sequence = _load_ground_truth(spec, sequence_name, config)
        num_frames_eff = int(sequence.num_frames)
        query_idx = align_batch_queries_with_sequence(
            batch, sequence, sequence_name=sequence_name
        )
        q_count = int(query_idx.shape[0])

        outputs = adapter.infer(model, batch, config.mixed_precision)
        if (
            isinstance(outputs, list)
            and len(outputs) > 0
            and isinstance(outputs[-1], dict)
            and outputs[-1].get("type") is not None
        ):
            batch = outputs.pop()

        prediction = adapter.extract(
            outputs,
            num_frames=num_frames_eff,
            num_queries=q_count,
        )

        gt_tracks = sequence.gt_tracks_world[:num_frames_eff, query_idx]
        for field in ("tracks_ref0", "flow_ref0"):
            value = getattr(prediction, field, None)
            if value is not None and np.asarray(value).shape != (num_frames_eff, q_count, 3):
                raise ValueError(f"{sequence_name}: {field} shape does not match the evaluated queries")

        metrics = task.score_prediction(
            gt_tracks,
            prediction,
            sequence_name=sequence_name,
            num_queries=q_count,
            num_queries_json=int(sequence.num_queries),
            num_frames=num_frames_eff,
            dynamic_threshold=threshold,
            **({"eval_dynamic_only": dynamic_only} if task.accepts_dynamic_only else {}),
        )
        if prediction.tracks_ref0 is not None:
            metrics["sanity_epe_ref0_vs_gt"] = _ref0_epe(prediction.tracks_ref0, gt_tracks)

        results.append(metrics)

        primary = task.primary_values(metrics)
        logging.info(
            "%s | %s | T=%d Q=%d | %s",
            spec.name,
            sequence_name,
            num_frames_eff,
            q_count,
            " ".join(
                f"{key}={value:.4f}" for key, value in zip(task.primary_keys, primary)
            ),
        )

        if config.save_per_sequence:
            write_summary_json(subset_out / f"{sequence_name}.json", metrics)
        if config.save_predictions:
            save_prediction(
                pred_dir / f"{sequence_name}.npz",
                SequencePrediction(
                    sequence_name=sequence_name,
                    tracks_ref0=prediction.tracks_ref0,
                    flow_ref0=prediction.flow_ref0,
                    query_indices=query_idx,
                ),
            )

        if config.visualize:
            overlay = _write_overlay(
                spec=spec,
                sequence_name=sequence_name,
                config=config,
                query_idx=query_idx,
                num_frames_eff=num_frames_eff,
                pred_ref0=prediction.tracks_ref0,
                subset_out=subset_out,
            )
            if overlay is not None:
                logging.info("Overlay: %s", overlay)

    summary = task.aggregate(results)
    summary["sequences"] = results

    if skipped_no_query:
        summary["skipped_no_frame0_queries"] = skipped_no_query

    _check_reference_queries(spec, summary)
    write_summary_json(subset_out / "summary.json", summary)
    return summary


def _check_reference_queries(spec: SubsetSpec, summary: Dict[str, Any]) -> None:
    """Warn when the number of scored queries differs from the reference."""
    expected = spec.reference_queries
    if not expected:
        return
    actual = int(summary.get("total_queries", 0) or 0)
    if actual != int(expected):
        logging.warning(
            "%s: scored %d queries, expected %d -- check the query selection "
            "and the dynamic-query threshold.",
            spec.name,
            actual,
            int(expected),
        )


def run_evaluation(
    *,
    task: BaseTask,
    adapter: PredictionAdapter,
    model: Any,
    specs: Sequence[SubsetSpec],
    config: RunConfig,
    model_config: Optional[Dict[str, Any]] = None,
    inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate several subsets and write the global summary.

    Args:
        task: The task that scores the predictions.
        adapter: The model adapter.
        model: The loaded model (already on its device).
        specs: The subsets to evaluate.
        config: Run options.
        model_config: The model configuration; used to build the input dataset.
        inputs: Extra fields recorded in the global summary.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import hashlib

    manifest = dict(task=task.name, num_frames=config.num_frames,
                    model_config=model_config, inputs=inputs or {},
                    dynamic_threshold=config.dynamic_threshold,
                    eval_dynamic_only=config.eval_dynamic_only,
                    subsets=[dict(name=spec.name, json_path=str(spec.json_path.resolve()),
                                  json_sha256=hashlib.sha256(spec.json_path.read_bytes()).hexdigest(),
                                  data_root=str(spec.resolved_data_root().resolve()),
                                  coord_convention=spec.coord_convention,
                                  dynamic_query_threshold=spec.dynamic_query_threshold)
                             for spec in specs])
    checkpoint = (inputs or {}).get("load_from")
    if checkpoint and Path(checkpoint).is_file():
        stat = Path(checkpoint).stat()
        manifest["checkpoint_stat"] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous != manifest:
            raise ValueError("Output directory belongs to a different evaluation; choose a new --output-dir")
    elif config.resume and any(output_dir.glob("*/*.json")):
        raise ValueError("Cannot resume results without run_manifest.json; choose a new --output-dir")
    write_summary_json(manifest_path, manifest)

    all_summary: Dict[str, Any] = {
        "task": task.name,
        "inputs": dict(inputs or {}),
        "subsets": {},
    }
    lines = [f"eval4d {task.name} summary"]

    for spec in specs:
        dataset = adapter.build_dataset(
            spec.json_path,
            spec.resolved_data_root(),
            model_config or {},
            config.num_frames,
            sequence_names=_sequence_filter(
                spec.json_path,
                limit_seqs=config.limit_seqs,
                seq_name=config.seq_name,
            ),
            coord_convention=spec.coord_convention,
            dynamic_query_threshold=spec.dynamic_query_threshold,
        )
        summary = evaluate_subset(
            task=task,
            adapter=adapter,
            model=model,
            dataset=dataset,
            spec=spec,
            config=config,
        )
        all_summary["subsets"][spec.name] = summary
        lines.append(task.format_summary(spec.name, summary))

    write_summary_json(output_dir / "summary.json", all_summary)
    text = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    return all_summary


def score_offline_predictions(
    *,
    task: BaseTask,
    specs: Sequence[SubsetSpec],
    config: RunConfig,
    predictions_dir: Any,
    inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score saved predictions against the ground truth -- no GPU required.

    The predictions are expected in ``<predictions_dir>/<subset>/*.npz``; a
    flat ``<predictions_dir>/*.npz`` layout is accepted as well.

    The ground-truth query set is rebuilt from the subset JSON with the same
    dynamic-query threshold, so the offline numbers match the online ones.
    """
    from .io.predictions import read_predictions

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(predictions_dir)

    all_summary: Dict[str, Any] = {
        "task": task.name,
        "inputs": dict(inputs or {}),
        "subsets": {},
    }
    lines = [f"eval4d {task.name} summary (offline)"]

    for spec in specs:
        subset_dir = root / spec.name
        source = subset_dir if subset_dir.is_dir() else root
        if (source / "predictions").is_dir():
            source = source / "predictions"
        predictions = read_predictions(source)
        if not predictions:
            raise ValueError(f"No prediction NPZ files found in {source}")
        selected = set(_sequence_filter(spec.json_path, limit_seqs=config.limit_seqs,
                                        seq_name=config.seq_name))

        results: List[Dict[str, Any]] = []
        for sequence_name, prediction in predictions.items():
            if sequence_name not in selected:
                continue
            sequence = _load_ground_truth(spec, sequence_name, config)
            query_idx = np.asarray(sequence.query_indices, dtype=np.int64)
            if prediction.query_indices is not None and not np.array_equal(prediction.query_indices, query_idx):
                raise ValueError(f"{sequence_name}: saved query indices differ from the current evaluation")
            for field in ("tracks_ref0", "flow_ref0"):
                array = getattr(prediction, field, None)
                if array is None:
                    continue
                if int(array.shape[1]) != int(query_idx.size):
                    raise ValueError(
                        f"{spec.name}/{sequence_name}: the prediction has "
                        f"{int(array.shape[1])} queries but the ground truth has "
                        f"{int(query_idx.size)}; the dynamic-query threshold of "
                        f"the scoring run must match the one of the prediction run."
                    )
            num_frames = int(sequence.num_frames)
            if prediction.num_frames != num_frames:
                raise ValueError(f"{sequence_name}: saved clip length differs; use the same --num-frames")
            gt_tracks = sequence.gt_tracks_world[:num_frames, query_idx]
            metrics = task.score_prediction(
                gt_tracks,
                prediction,
                sequence_name=sequence_name,
                num_queries=int(query_idx.size),
                num_queries_json=int(sequence.num_queries),
                num_frames=num_frames,
                dynamic_threshold=_metric_dynamic_threshold(spec, config),
                **(
                    {"eval_dynamic_only": _uses_dynamic_only(spec, config)}
                    if task.accepts_dynamic_only
                    else {}
                ),
            )
            results.append(metrics)

        summary = task.aggregate(results)
        summary["sequences"] = results
        all_summary["subsets"][spec.name] = summary
        _check_reference_queries(spec, summary)
        write_summary_json(output_dir / spec.name / "summary.json", summary)
        lines.append(task.format_summary(spec.name, summary))

    write_summary_json(output_dir / "summary.json", all_summary)
    text = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    return all_summary
