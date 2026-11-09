"""Evaluation tasks.

Each task bundles the scoring, aggregation and reporting rules of one
evaluation flavour. The tasks are independent: the Dynamic Points task never
touches the Scene Flow metrics and vice versa.
"""

from __future__ import annotations

from .base import BaseTask, finalize_sequence_metrics, resolve_dynamic_threshold
from .dynamic_points import DYNAMIC_POINTS_PRIMARY_KEYS, DynamicPointsTask
from .scene_flow import SCENE_FLOW_PRIMARY_KEYS, SceneFlowTask

#: All available tasks, keyed by their command-line name.
TASKS = {
    DynamicPointsTask.name: DynamicPointsTask,
    SceneFlowTask.name: SceneFlowTask,
}


def get_task(name: str) -> BaseTask:
    """Instantiate a task by name (``dynamic-points`` / ``scene-flow``)."""
    key = str(name).strip().lower().replace("_", "-")
    if key not in TASKS:
        raise ValueError(
            f"Unknown task {name!r}; available tasks: {', '.join(sorted(TASKS))}."
        )
    return TASKS[key]()


__all__ = [
    "TASKS",
    "BaseTask",
    "DYNAMIC_POINTS_PRIMARY_KEYS",
    "DynamicPointsTask",
    "SCENE_FLOW_PRIMARY_KEYS",
    "SceneFlowTask",
    "finalize_sequence_metrics",
    "get_task",
    "resolve_dynamic_threshold",
]
