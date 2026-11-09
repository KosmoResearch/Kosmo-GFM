"""Dynamic Points metrics: 3D tracking of frame-0 queries in the ref0 world frame.

The model predicts, for every frame-0 query, its 3D position in the **frame-0
reference frame** for each subsequent frame. The prediction is compared against
the ground-truth ``trajs_3d`` after a single global median-norm scale
alignment (see :mod:`eval4d.metrics.alignment`).

Reported metrics
----------------
``avg_pts_global`` (APD)
    Mean of the hit rates at 0.1 / 0.3 / 0.5 / 1.0 m.
``tau_global`` (tau@0.1m)
    Hit rate at 0.1 m only (``fractions_global[1]``).
``epe_global`` (EPE)
    Mean 3D Euclidean error after the global scale alignment, in metres.

Secondary metrics (GT-dynamic query subset)
--------------------------------------------
``avg_pts_global_dyn`` / ``tau_global_dyn`` / ``epe_global_dyn`` are recomputed
**on the dynamic subset with its own re-estimated scale**. This asymmetry with
the Scene Flow task (which reuses the global scale on its dynamic subset) is
intentional and part of the protocol.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from ..data.queries import max_displacement_mask
from .alignment import compute_average_pts_within_thresh
from .constants import (
    DEFAULT_DYNAMIC_THRESHOLD_M,
    DP_APD_GLOBAL_KEY,
    DP_EPE_GLOBAL_KEY,
    DP_TAU_GLOBAL_KEY,
    DP_TAU_GLOBAL_DYN_KEY,
)

__all__ = [
    "APD_GLOBAL_KEY",
    "EPE_GLOBAL_KEY",
    "TAU_GLOBAL_KEY",
    "TAU_GLOBAL_DYN_KEY",
    "DYNAMIC_POINTS_PRIMARY_KEYS",
    "dynamic_point_mask_ref0",
    "metrics_for_sequence",
    "aggregate_results",
    "format_subset_summary",
]

APD_GLOBAL_KEY = DP_APD_GLOBAL_KEY
EPE_GLOBAL_KEY = DP_EPE_GLOBAL_KEY
TAU_GLOBAL_KEY = DP_TAU_GLOBAL_KEY
TAU_GLOBAL_DYN_KEY = DP_TAU_GLOBAL_DYN_KEY

#: The metrics that define the reported Dynamic Points result.
DYNAMIC_POINTS_PRIMARY_KEYS = (APD_GLOBAL_KEY, TAU_GLOBAL_KEY, EPE_GLOBAL_KEY)


def dynamic_point_mask_ref0(
    tracks_tq3: np.ndarray,
    threshold: float = DEFAULT_DYNAMIC_THRESHOLD_M,
) -> np.ndarray:
    """True for queries whose GT displacement from frame 0 exceeds ``threshold`` (m)."""
    return max_displacement_mask(tracks_tq3, threshold)


def _empty_dynamic_only_result(dynamic_threshold: float) -> Dict[str, Any]:
    return {
        APD_GLOBAL_KEY: float("nan"),
        TAU_GLOBAL_KEY: float("nan"),
        "avg_pts_pertraj": float("nan"),
        "avg_pts_sim3": float("nan"),
        "avg_pts_sim3_closed": float("nan"),
        EPE_GLOBAL_KEY: float("nan"),
        "epe_pertraj": float("nan"),
        "epe_sim3": float("nan"),
        "epe_sim3_closed": float("nan"),
        "avg_pts_global_dyn": float("nan"),
        TAU_GLOBAL_DYN_KEY: float("nan"),
        "epe_global_dyn": float("nan"),
        "avg_pts_sim3_closed_dyn": float("nan"),
        "epe_sim3_closed_dyn": float("nan"),
        "fractions_global": {},
        "fractions_pertraj": {},
        "fractions_sim3": {},
        "fractions_sim3_closed": {},
        "fractions_global_dyn": {},
        "fractions_sim3_closed_dyn": {},
        "dyn_fraction": float("nan"),
        "dyn_count": 0,
        "ref0_dyn_count": 0,
        "num_queries": 0,
        "eval_dynamic_only": True,
        "dynamic_threshold": float(dynamic_threshold),
    }


def metrics_for_sequence(
    gt_tracks_world: np.ndarray,
    pred_tracks_ref0: np.ndarray,
    compute_dyn: bool = True,
    dynamic_threshold: float = DEFAULT_DYNAMIC_THRESHOLD_M,
    eval_dynamic_only: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Per-sequence Dynamic Points metrics for trajectories in the ref0 world frame.

    Args:
        gt_tracks_world: ``[T, Q, 3]`` ground truth.
        pred_tracks_ref0: ``[T, Q, 3]`` predictions in the same frame.
        compute_dyn: Whether to compute the dynamic-query subset metrics.
        dynamic_threshold: Ref0 max displacement (m) above which a query is dynamic.
        eval_dynamic_only: If True, score only dynamic queries and fit the global
            scale on those queries only.
        rng: RNG for the Sim(3) RANSAC subsampling (seed 0 when None).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    gt_full = np.asarray(gt_tracks_world, dtype=np.float64)
    pred_full = np.asarray(pred_tracks_ref0, dtype=np.float64)
    ref0_dyn_mask = dynamic_point_mask_ref0(gt_full, dynamic_threshold)
    ref0_dyn_count = int(ref0_dyn_mask.sum())

    gt_use = gt_full
    pred_use = pred_full
    align_mask: Optional[np.ndarray] = None
    if eval_dynamic_only:
        if ref0_dyn_count <= 0:
            return _empty_dynamic_only_result(dynamic_threshold)
        gt_use = gt_full[:, ref0_dyn_mask]
        pred_use = pred_full[:, ref0_dyn_mask]

    avg_pts_global, _, fractions_global, _, epe_global = compute_average_pts_within_thresh(
        gt_use,
        pred_use,
        scaling="global",
        compute_epe=True,
        rng=rng,
        align_query_mask=align_mask,
    )
    avg_pts_pertraj, _, fractions_pertraj, _, epe_pertraj = compute_average_pts_within_thresh(
        gt_use,
        pred_use,
        scaling="per_traj",
        compute_epe=True,
        rng=rng,
    )
    avg_pts_sim3, _, fractions_sim3, _, epe_sim3 = compute_average_pts_within_thresh(
        gt_use,
        pred_use,
        scaling="sim3",
        compute_epe=True,
        rng=rng,
    )
    avg_pts_sim3_closed, _, fractions_sim3_closed, _, epe_sim3_closed = compute_average_pts_within_thresh(
        gt_use,
        pred_use,
        scaling="sim3_closed",
        compute_epe=True,
        rng=rng,
    )

    avg_pts_global_dyn = float("nan")
    epe_global_dyn = float("nan")
    avg_pts_sim3_closed_dyn = float("nan")
    epe_sim3_closed_dyn = float("nan")
    fractions_global_dyn: Dict[int, float] = {}
    fractions_sim3_closed_dyn: Dict[int, float] = {}
    dyn_fraction = float("nan")
    dyn_count = 0

    if compute_dyn and gt_use.shape[0] >= 2:
        total_motion = gt_use[1:] - gt_use[:-1]
        total_motion_norm = np.linalg.norm(total_motion, axis=-1).sum(axis=0)
        dyn_mask = total_motion_norm > float(dynamic_threshold)
        dyn_count = int(dyn_mask.sum())
        dyn_fraction = float(dyn_mask.mean()) if dyn_mask.size > 0 else float("nan")
        if dyn_count > 0:
            avg_pts_global_dyn, _, fractions_global_dyn, _, epe_global_dyn = compute_average_pts_within_thresh(
                gt_use[:, dyn_mask],
                pred_use[:, dyn_mask],
                scaling="global",
                compute_epe=True,
                rng=rng,
            )
            avg_pts_sim3_closed_dyn, _, fractions_sim3_closed_dyn, _, epe_sim3_closed_dyn = compute_average_pts_within_thresh(
                gt_use[:, dyn_mask],
                pred_use[:, dyn_mask],
                scaling="sim3_closed",
                compute_epe=True,
                rng=rng,
            )

    tau_global = float(fractions_global.get(1, float("nan")))
    tau_global_dyn = float(fractions_global_dyn.get(1, float("nan")))

    return {
        APD_GLOBAL_KEY: float(avg_pts_global),
        TAU_GLOBAL_KEY: tau_global,
        "avg_pts_pertraj": float(avg_pts_pertraj),
        "avg_pts_sim3": float(avg_pts_sim3),
        "avg_pts_sim3_closed": float(avg_pts_sim3_closed),
        EPE_GLOBAL_KEY: float(epe_global),
        "epe_pertraj": float(epe_pertraj),
        "epe_sim3": float(epe_sim3),
        "epe_sim3_closed": float(epe_sim3_closed),
        "avg_pts_global_dyn": float(avg_pts_global_dyn),
        TAU_GLOBAL_DYN_KEY: tau_global_dyn,
        "epe_global_dyn": float(epe_global_dyn),
        "avg_pts_sim3_closed_dyn": float(avg_pts_sim3_closed_dyn),
        "epe_sim3_closed_dyn": float(epe_sim3_closed_dyn),
        "fractions_global": fractions_global,
        "fractions_pertraj": fractions_pertraj,
        "fractions_sim3": fractions_sim3,
        "fractions_sim3_closed": fractions_sim3_closed,
        "fractions_global_dyn": fractions_global_dyn,
        "fractions_sim3_closed_dyn": fractions_sim3_closed_dyn,
        "dyn_fraction": float(dyn_fraction),
        "dyn_count": int(dyn_count),
        "ref0_dyn_count": int(ref0_dyn_count),
        "num_queries": int(gt_use.shape[1]),
        "eval_dynamic_only": bool(eval_dynamic_only),
        "dynamic_threshold": float(dynamic_threshold),
    }


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean-pool per-sequence metrics into a subset summary (macro over sequences)."""
    scalar_keys = [
        APD_GLOBAL_KEY,
        TAU_GLOBAL_KEY,
        "avg_pts_pertraj",
        "avg_pts_sim3",
        "avg_pts_sim3_closed",
        EPE_GLOBAL_KEY,
        "epe_pertraj",
        "epe_sim3",
        "epe_sim3_closed",
        "avg_pts_global_dyn",
        TAU_GLOBAL_DYN_KEY,
        "epe_global_dyn",
        "avg_pts_sim3_closed_dyn",
        "epe_sim3_closed_dyn",
        "dyn_fraction",
    ]
    fraction_keys = [
        "fractions_global",
        "fractions_pertraj",
        "fractions_sim3",
        "fractions_sim3_closed",
        "fractions_global_dyn",
        "fractions_sim3_closed_dyn",
    ]
    summary: Dict[str, Any] = {"num_sequences": int(len(results))}
    for key in scalar_keys:
        values = [
            float(item[key])
            for item in results
            if key in item and np.isfinite(float(item[key]))
        ]
        summary[key] = float(np.mean(values)) if values else float("nan")
    summary["total_queries"] = int(sum(int(item.get("num_queries", 0)) for item in results))
    summary["total_dynamic_queries"] = int(sum(int(item.get("dyn_count", 0)) for item in results))

    for frac_key in fraction_keys:
        agg: Dict[int, List[float]] = defaultdict(list)
        for item in results:
            payload = item.get(frac_key, {})
            if not isinstance(payload, dict):
                continue
            for thr, value in payload.items():
                if np.isfinite(float(value)):
                    agg[int(thr)].append(float(value))
        summary[frac_key] = {int(thr): float(np.mean(vals)) for thr, vals in sorted(agg.items())}
    return summary


def format_subset_summary(subset: str, summary: Dict[str, Any]) -> str:
    """One-line summary of a Dynamic Points subset."""
    return (
        f"{subset}: "
        f"APD(global)={summary.get(APD_GLOBAL_KEY, float('nan')):.4f} "
        f"tau(global)={summary.get(TAU_GLOBAL_KEY, float('nan')):.4f} "
        f"EPE(global)={summary.get(EPE_GLOBAL_KEY, float('nan')):.4f} "
        f"APD(global,dyn)={summary.get('avg_pts_global_dyn', float('nan')):.4f} "
        f"tau(global,dyn)={summary.get(TAU_GLOBAL_DYN_KEY, float('nan')):.4f} "
        f"EPE(global,dyn)={summary.get('epe_global_dyn', float('nan')):.4f} "
        f"queries={int(summary.get('total_queries', 0))}"
    )
