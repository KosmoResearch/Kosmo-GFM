#!/usr/bin/env python3
"""Score saved predictions against the ground truth.

This is the offline counterpart of ``gfm-4d-evaluate``: it needs neither a GPU
nor a model, only the predictions written by a previous run and the subset
JSON with the ground truth.

Example::

    gfm-4d-score --task dynamic-points --subset adt --subset pstudio \\
        --data-root /path/to/4dbenchmark_data --predictions-dir tmp/dp --output-dir tmp/dp_rescored
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from ..runner import RunConfig, score_offline_predictions
from .common import TASKS, add_subset_arguments, make_task, resolve_specs, validate_specs

__all__ = ["main", "build_parser"]



def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of ``gfm-4d-score``."""
    parser = argparse.ArgumentParser(
        prog="gfm-4d-score",
        description="Score saved WorldTrack predictions against the ground truth.",
    )
    parser.add_argument("--task", required=True, choices=TASKS, help="Which protocol to score with.")
    add_subset_arguments(parser)
    parser.add_argument(
        "--predictions-dir",
        required=True,
        help="Directory with the saved predictions (<subset>/*.npz, or a flat layout).",
    )
    parser.add_argument(
        "--output-dir",
        default="tmp/eval4d_score",
        help="Where the summaries are written.",
    )
    parser.add_argument("--num-frames", type=int, default=64, help="Clip length.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point of ``gfm-4d-score``."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    )

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


if __name__ == "__main__":
    raise SystemExit(main())
