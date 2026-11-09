"""Aggregation helpers for benchmark reports.

Two cross-subset policies exist, and they are deliberately different:

* **Query-weighted** -- each subset is weighted by the number of scored
  queries. This is the policy used for the Dynamic Points tables.
* **Macro** -- every subset counts the same, no matter how many queries it
  contains. This is the policy used for the Scene Flow tables.

Within a subset the aggregation is always the macro mean over sequences; see
:func:`eval4d.metrics.dynamic_points.aggregate_results`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "load_summary",
    "collect_subset_summaries",
    "macro_average",
    "query_weighted_average",
    "mean_of_finite",
    "format_value",
]


def load_summary(path: Any) -> Dict[str, Any]:
    """Read a ``summary.json`` file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect_subset_summaries(
    root: Any,
    subsets: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Load ``<root>/<subset>/summary.json`` for every subset that exists."""
    base = Path(root)
    out: Dict[str, Dict[str, Any]] = {}
    for name in subsets:
        path = base / name / "summary.json"
        if path.is_file():
            out[name] = load_summary(path)
    return out


def mean_of_finite(values: Iterable[float]) -> float:
    """Mean of the finite values; ``NaN`` when there is none."""
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.mean(arr[np.isfinite(arr)]))


def macro_average(
    summaries: Mapping[str, Mapping[str, Any]],
    keys: Sequence[str],
) -> Dict[str, float]:
    """Mean over the subsets, every subset weighted equally."""
    out: Dict[str, float] = {}
    for key in keys:
        out[key] = mean_of_finite(
            summary.get(key, float("nan")) for summary in summaries.values()
        )
    return out


def query_weighted_average(
    summaries: Mapping[str, Mapping[str, Any]],
    keys: Sequence[str],
    *,
    weight_key: str = "total_queries",
) -> Dict[str, float]:
    """Mean over the subsets, weighted by the number of scored queries."""
    weights = {
        name: float(summary.get(weight_key, 0) or 0)
        for name, summary in summaries.items()
    }
    total = sum(weights.values())
    out: Dict[str, float] = {}
    for key in keys:
        if total <= 0:
            out[key] = float("nan")
            continue
        acc = 0.0
        for name, summary in summaries.items():
            value = float(summary.get(key, float("nan")))
            if not np.isfinite(value):
                continue
            acc += value * weights[name]
        out[key] = acc / total
    return out


def format_value(value: Any, *, scale: float = 1.0, digits: int = 2) -> str:
    """Format a metric value for a markdown table."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    return f"{number * scale:.{digits}f}"


def format_row(
    summary: Mapping[str, Any],
    keys: Sequence[str],
    *,
    scales: Optional[Mapping[str, float]] = None,
    digits: int = 2,
) -> List[str]:
    """Format the given summary keys as markdown cells."""
    scale_map = dict(scales or {})
    return [
        format_value(summary.get(key, float("nan")), scale=scale_map.get(key, 1.0), digits=digits)
        for key in keys
    ]


def summary_has_subset(summaries: Mapping[str, Mapping[str, Any]], names: Iterable[str]) -> bool:
    """Whether any of the named subsets is present."""
    return any(name in summaries for name in names)


def subset_query_counts(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    weight_key: str = "total_queries",
) -> Dict[str, int]:
    """The number of scored queries per subset."""
    return {
        name: int(summary.get(weight_key, 0) or 0)
        for name, summary in summaries.items()
    }


def format_summary_line(
    subset: str,
    summary: Mapping[str, Any],
    primary_keys: Sequence[str],
) -> str:
    """A single-line, human-readable rendering of a subset summary."""
    parts = [f"{subset}:"]
    for key in primary_keys:
        value = float(summary.get(key, float("nan")))
        parts.append(f"{key}={value:.4f}")
    return " ".join(parts)


def as_list(payload: Any) -> List[Any]:
    """Coerce a JSON payload into a list."""
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    return [payload]
