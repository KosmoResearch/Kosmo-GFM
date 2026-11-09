#!/usr/bin/env python3
"""Evaluate a model on the WorldTrack / TapVid3D benchmarks.

Two modes:

* **Live** -- run a model over the subsets and score the predictions. This
  needs a model adapter (the bundled one is ``gfm4d``).
* **Offline** -- score predictions that were saved earlier, with
  ``--predictions-dir``. No GPU and no model are needed.

Examples::

    gfm-4d-evaluate --task dynamic-points --subset adt --subset pstudio \\
        --data-root /path/to/4dbenchmark_data --adapter gfm4d \\
        --config configs/gfm4d.yaml --load-from checkpoint.pth --output-dir tmp/dp

    gfm-4d-evaluate --task scene-flow --subset adt --data-root /path/to/4dbenchmark_data \\
        --predictions-dir tmp/dp --output-dir tmp/sf_offline
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..runner import RunConfig, run_evaluation, score_offline_predictions
from .common import TASKS, add_subset_arguments, make_task, resolve_specs, validate_specs

__all__ = ["main", "build_parser"]



def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of ``gfm-4d-evaluate``."""
    parser = argparse.ArgumentParser(
        prog="gfm-4d-evaluate",
        description="Evaluate a model on WorldTrack / TapVid3D (Dynamic Points or Scene Flow). "
        "Defaults to adt, pstudio, po and ds when no --subset or --json is given.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=TASKS,
        help="Which protocol to score with.",
    )
    add_subset_arguments(parser)

    parser.add_argument(
        "--output-dir",
        default="tmp/eval4d",
        help="Where the summaries and the predictions are written.",
    )
    parser.add_argument("--num-frames", type=int, default=64, help="Clip length.")
    parser.add_argument(
        "--limit-seqs",
        type=int,
        default=0,
        help="Number of sequences per subset; 0 or less means all of them.",
    )
    parser.add_argument(
        "--seq-name",
        default="",
        help="Evaluate only this sequence (overrides --limit-seqs).",
    )
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument(
        "--mixed-precision",
        default="fp16",
        choices=("fp16",),
        help="Inference precision.",
    )

    parser.add_argument(
        "--adapter",
        default="gfm4d",
        choices=("gfm4d",),
        help="Model adapter used for the live run.",
    )
    parser.add_argument("--config", default=None, help="Model configuration file.")
    parser.add_argument("--load-from", default=None, help="Model checkpoint.")
    parser.add_argument(
        "--pred-3d-source",
        default="warp3d",
        choices=("warp3d", "warp3d_delta"),
        help=(
            "How the GFM4D adapter reads the per-pair 3D prediction: 'warp3d' is the "
            "absolute reference-frame position, 'warp3d_delta' is src_points + warp3d_delta."
        ),
    )

    parser.add_argument(
        "--predictions-dir",
        default=None,
        help="Score the predictions in this directory instead of running a model.",
    )
    parser.add_argument(
        "--no-save-per-sequence",
        dest="save_per_sequence",
        action="store_false",
        help="Do not write one JSON file per sequence.",
    )
    parser.add_argument(
        "--no-save-predictions",
        dest="save_predictions",
        action="store_false",
        help="Do not save the predictions as npz files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip the sequences that already have a per-sequence JSON.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Render an overlay video of the ground truth and the prediction.",
    )
    parser.add_argument("--vis-fps", type=float, default=15.0)
    parser.add_argument(
        "--vis-max-points",
        type=int,
        default=300,
        help="Maximum number of tracks drawn in the overlay.",
    )
    return parser


def _adapter_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    kwargs["pred_3d_source"] = args.pred_3d_source
    return kwargs


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point of ``gfm-4d-evaluate``."""
    args = build_parser().parse_args(argv)
    if not args.subsets and not args.json_paths:
        args.subsets = ["adt", "pstudio", "po", "ds"]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.num_frames < 1:
        raise ValueError("--num-frames must be positive")
    if args.resume and not args.save_per_sequence:
        raise ValueError("--resume requires per-sequence results")
    task = make_task(args.task)
    specs = resolve_specs(args)
    try:
        validate_specs(specs)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("%s", exc)
        return 2

    config = RunConfig(
        output_dir=Path(args.output_dir),
        num_frames=int(args.num_frames),
        limit_seqs=int(args.limit_seqs),
        seq_name=str(args.seq_name),
        mixed_precision=str(args.mixed_precision),
        device=str(args.device),
        save_per_sequence=bool(args.save_per_sequence),
        save_predictions=bool(args.save_predictions),
        resume=bool(args.resume),
        visualize=bool(args.visualize),
        vis_fps=float(args.vis_fps),
        vis_max_points=int(args.vis_max_points),
    )

    if args.predictions_dir:
        score_offline_predictions(
            task=task,
            specs=specs,
            config=config,
            predictions_dir=Path(args.predictions_dir),
            inputs={
                "mode": "offline",
                "predictions_dir": str(args.predictions_dir),
                "num_frames": int(args.num_frames),
            },
        )
        return 0

    if not args.load_from:
        logging.error("--load-from is required for a live run.")
        return 2

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Live evaluation requires PyTorch. Install the GFM extra with "
            "'pip install -e \"bench/GFM4DBench[gfm]\"' plus the PyTorch and "
            "torchvision builds for your CUDA version."
        ) from exc

    from ..adapters import get_adapter

    adapter = get_adapter(args.adapter, **_adapter_kwargs(args))
    model_config = adapter.load_config(args.config)
    device = torch.device(args.device)
    model = adapter.load_model(model_config, Path(args.load_from), device)

    run_evaluation(
        task=task,
        adapter=adapter,
        model=model,
        specs=specs,
        config=config,
        model_config=model_config,
        inputs={
            "mode": "live",
            "adapter": args.adapter,
            "config": str(args.config),
            "load_from": str(args.load_from),
            "num_frames": int(args.num_frames),
            "mixed_precision": args.mixed_precision,
            "device": str(device),
            "pred_3d_source": args.pred_3d_source,
            "model_config": model_config,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
