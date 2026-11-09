"""Command-line entry points.

* ``gfm-4d-evaluate`` -- run a model over the benchmark subsets, or score
  saved predictions.
* ``gfm-4d-score`` -- score saved predictions against the ground truth.
* ``gfm-4d-report`` -- build a benchmark table from evaluation summaries.
"""

from __future__ import annotations

from .common import TASKS, make_task, parse_dynamic_threshold, resolve_specs

__all__ = ["TASKS", "make_task", "parse_dynamic_threshold", "resolve_specs"]
