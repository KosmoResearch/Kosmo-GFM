"""Markdown benchmark tables.

Two tables are supported:

* **Dynamic Points** -- ``APD (%)``, ``tau@0.1m (%)`` and ``EPE (m)`` per
  subset. The TapVid3D average is **query-weighted** over the TapVid3D subsets.
* **Scene Flow** -- ``tau@0.1m (%)`` and ``EPE (m)`` per subset, plus the
  macro averages over all subsets and over the GT-dynamic subsets.

The metric keys are read from the summary payloads, so the tables work on the
output of any run of this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..metrics.constants import (
    DP_APD_GLOBAL_KEY,
    DP_EPE_GLOBAL_KEY,
    DP_TAU_GLOBAL_KEY,
    SF_APD_DYN_KEY,
    SF_APD_GLOBAL_KEY,
    SF_EPE_DYN_KEY,
    SF_EPE_GLOBAL_KEY,
)
from .summary import format_value, macro_average, query_weighted_average

__all__ = [
    "SubsetColumn",
    "ModelRow",
    "dynamic_points_markdown",
    "scene_flow_markdown",
]


@dataclass
class SubsetColumn:
    """One column of a benchmark table."""

    key: str
    label: str


@dataclass
class ModelRow:
    """One row of a benchmark table: a model and its per-subset summaries."""

    name: str
    subsets: Dict[str, Mapping[str, Any]] = field(default_factory=dict)

    def summary(self, column: SubsetColumn) -> Optional[Mapping[str, Any]]:
        return self.subsets.get(column.key)


#: The TapVid3D subsets, whose average is query-weighted.
TAPVID3D_SUBSETS = ("adt", "pstudio", "po", "ds")


def _dp_cells(
    row: ModelRow,
    columns: Sequence[SubsetColumn],
    metrics: Sequence[str],
) -> List[str]:
    cells: List[str] = []
    for column in columns:
        summary = row.summary(column)
        if summary is None:
            cells.extend("-" for _ in metrics)
            continue
        for key in metrics:
            scale = 100.0 if key in (DP_APD_GLOBAL_KEY, DP_TAU_GLOBAL_KEY) else 1.0
            digits = 2 if key != DP_EPE_GLOBAL_KEY else 4
            cells.append(format_value(summary.get(key, float("nan")), scale=scale, digits=digits))
    return cells


def dynamic_points_markdown(
    rows: Sequence[ModelRow],
    columns: Sequence[SubsetColumn],
    *,
    title: str = "Dynamic Points benchmark",
    tapvid3d_subsets: Sequence[str] = TAPVID3D_SUBSETS,
    extra_average_column: Optional[SubsetColumn] = None,
) -> str:
    """Build the Dynamic Points markdown table.

    Args:
        rows: One row per model.
        columns: The subset columns, in display order.
        title: The table title.
        tapvid3d_subsets: The subsets whose average is query-weighted.
        extra_average_column: An optional extra subset column, averaged with
            the macro policy for that group.
    """
    metrics = (DP_APD_GLOBAL_KEY, DP_TAU_GLOBAL_KEY, DP_EPE_GLOBAL_KEY)
    metric_titles = ("APD (%)", "tau@0.1m (%)", "EPE (m)")

    group_header = ["Model"]
    metric_header = [""]
    for column in columns:
        group_header.extend([column.label, "", ""])
        metric_header.extend(metric_titles)
    if tapvid3d_subsets:
        group_header.extend(["TapVid3D Avg", "", ""])
        metric_header.extend(metric_titles)
    if extra_average_column is not None:
        group_header.extend([extra_average_column.label, "", ""])
        metric_header.extend(metric_titles)

    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(group_header) + " |")
    lines.append("| " + " | ".join(metric_header) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(group_header) - 1)) + " |")
    for row in rows:
        cells = [row.name]
        cells.extend(_dp_cells(row, columns, metrics))
        if tapvid3d_subsets:
            cells.extend(_tapvid3d_average_cells(row, metrics, tapvid3d_subsets))
        if extra_average_column is not None:
            cells.extend(_extra_average_cells(row, metrics, extra_average_column))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    return "\n".join(lines)


def _tapvid3d_average_cells(
    row: ModelRow,
    metrics: Sequence[str],
    tapvid3d_subsets: Sequence[str],
) -> List[str]:
    subsets = {
        name: summary
        for name, summary in row.subsets.items()
        if name in tapvid3d_subsets
    }
    if not subsets:
        return ["-" for _ in metrics]
    averaged = query_weighted_average(subsets, metrics)
    cells: List[str] = []
    for key in metrics:
        scale = 100.0 if key in (DP_APD_GLOBAL_KEY, DP_TAU_GLOBAL_KEY) else 1.0
        digits = 2 if key != DP_EPE_GLOBAL_KEY else 4
        cells.append(format_value(averaged.get(key, float("nan")), scale=scale, digits=digits))
    return cells


def _extra_average_cells(
    row: ModelRow,
    metrics: Sequence[str],
    column: SubsetColumn,
) -> List[str]:
    summary = row.summary(column)
    if summary is None:
        return ["-" for _ in metrics]
    cells: List[str] = []
    for key in metrics:
        scale = 100.0 if key in (DP_APD_GLOBAL_KEY, DP_TAU_GLOBAL_KEY) else 1.0
        digits = 2 if key != DP_EPE_GLOBAL_KEY else 4
        cells.append(format_value(summary.get(key, float("nan")), scale=scale, digits=digits))
    return cells


def scene_flow_markdown(
    rows: Sequence[ModelRow],
    columns: Sequence[SubsetColumn],
    *,
    title: str = "Scene Flow benchmark (tau@0.1m)",
    include_dynamic: bool = True,
) -> str:
    """Build the Scene Flow markdown table.

    The ``Avg`` column is the **macro** average over the subsets; ``Avg(dyn)``
    is the macro average of the GT-dynamic subset metrics.
    """
    metrics = (SF_APD_GLOBAL_KEY, SF_EPE_GLOBAL_KEY)
    dyn_metrics = (SF_APD_DYN_KEY, SF_EPE_DYN_KEY)
    metric_titles = ("tau@0.1m (%)", "EPE (m)")

    group_header = ["Model"]
    metric_header = [""]
    for column in columns:
        group_header.extend([column.label, ""])
        metric_header.extend(metric_titles)
    group_header.extend(["Avg", ""])
    metric_header.extend(metric_titles)
    if include_dynamic:
        group_header.extend(["Avg(dyn)", ""])
        metric_header.extend(metric_titles)

    lines = [
        f"# {title}",
        "",
        "Protocol: frame-0 queries with a global median-norm scale. "
        "``tau`` is the scene-flow hit rate at 0.1 m; ``Avg`` is the macro "
        "average over the subsets.",
        "",
    ]
    if include_dynamic:
        lines.append(
            "**Avg(dyn)** is the macro average of the metrics restricted to the "
            "GT-dynamic queries; it reuses the global scale."
        )
        lines.append("")

    lines.append("| " + " | ".join(group_header) + " |")
    lines.append("| " + " | ".join(metric_header) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(group_header) - 1)) + " |")

    for row in rows:
        cells = [row.name]
        for column in columns:
            summary = row.summary(column)
            if summary is None:
                cells.extend(["-", "-"])
                continue
            cells.append(
                format_value(summary.get(SF_APD_GLOBAL_KEY, float("nan")), scale=100.0, digits=2)
            )
            cells.append(
                format_value(summary.get(SF_EPE_GLOBAL_KEY, float("nan")), digits=4)
            )
        averaged = macro_average(row.subsets, metrics)
        cells.append(format_value(averaged.get(SF_APD_GLOBAL_KEY, float("nan")), scale=100.0, digits=2))
        cells.append(format_value(averaged.get(SF_EPE_GLOBAL_KEY, float("nan")), digits=4))
        if include_dynamic:
            dyn = macro_average(row.subsets, dyn_metrics)
            cells.append(format_value(dyn.get(SF_APD_DYN_KEY, float("nan")), scale=100.0, digits=2))
            cells.append(format_value(dyn.get(SF_EPE_DYN_KEY, float("nan")), digits=4))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)
