"""Scale alignment between predictions and ground truth.

The WorldTrack protocol aligns a prediction to the ground truth with a single
**global median-norm scale** estimated over all finite ``(t, q)`` 3D points::

    scale = median(||GT||) / median(||Pred||)

Alternative alignments (per-trajectory scale, Sim(3)) are retained as
diagnostics and are not part of the reported protocol.

This module contains only mechanics; which alignment is *reported* is decided by
the task modules in :mod:`eval4d.protocol`.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .constants import APD_THRESHOLDS

__all__ = [
    "compute_scale_factor_global",
    "scale_per_trajectory",
    "estimate_sim3_closed_form",
    "estimate_sim3_ransac",
    "finite_correspondence_pairs",
    "compute_average_pts_within_thresh",
]


def compute_scale_factor_global(
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    query_mask: Optional[np.ndarray] = None,
) -> float:
    """Median-norm scale: ``median(||GT||) / median(||Pred||)`` over finite points."""
    gt = np.asarray(gt_points, dtype=np.float64)
    pred = np.asarray(pred_points, dtype=np.float64)
    gt_flat = gt.reshape(-1, 3)
    pred_flat = pred.reshape(-1, 3)
    finite = np.isfinite(gt_flat).all(axis=-1) & np.isfinite(pred_flat).all(axis=-1)
    if query_mask is not None:
        qmask = np.asarray(query_mask, dtype=bool).reshape(-1)
        if qmask.size != gt.shape[1]:
            raise ValueError(f"query_mask length {qmask.size} != Q={gt.shape[1]}")
        frame_q = np.broadcast_to(qmask.reshape(1, -1), gt.shape[:2]).reshape(-1)
        finite = finite & frame_q
    if not np.any(finite):
        return 1.0
    gt_norm = np.linalg.norm(gt_flat[finite], axis=-1)
    pred_norm = np.linalg.norm(pred_flat[finite], axis=-1)
    eps = 1e-12
    if gt_norm.size <= 0 or pred_norm.size <= 0:
        return 1.0
    gt_norm = np.maximum(gt_norm, eps)
    pred_norm = np.maximum(pred_norm, eps)
    return float(np.median(gt_norm) / max(float(np.median(pred_norm)), eps))


def scale_per_trajectory(
    gt_points: np.ndarray,
    pred_points: np.ndarray,
) -> np.ndarray:
    """Per-trajectory median scale alignment (diagnostic variant)."""
    gt = np.asarray(gt_points, dtype=np.float64)
    pred = np.asarray(pred_points, dtype=np.float64)
    out = pred.copy()
    eps = 1e-12
    for idx in range(gt.shape[1]):
        finite = np.isfinite(gt[:, idx]).all(axis=-1) & np.isfinite(pred[:, idx]).all(axis=-1)
        if not np.any(finite):
            continue
        gt_norm = np.linalg.norm(gt[finite, idx], axis=-1)
        pred_norm = np.linalg.norm(pred[finite, idx], axis=-1)
        if gt_norm.size <= 0 or pred_norm.size <= 0:
            continue
        gt_norm = np.maximum(gt_norm, eps)
        pred_norm = np.maximum(pred_norm, eps)
        scale = float(np.median(gt_norm) / max(float(np.median(pred_norm)), eps))
        out[:, idx] = pred[:, idx] * scale
    return out


def estimate_sim3_closed_form(
    src: np.ndarray,
    dst: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Closed-form Sim(3) (scale, rotation, translation) mapping ``src`` onto ``dst``."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    centroid_src = src.mean(axis=0, keepdims=True)
    centroid_dst = dst.mean(axis=0, keepdims=True)
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst
    h = src_centered.T @ dst_centered
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    var_src = float((src_centered**2).sum())
    scale = float(np.sum(s) / max(var_src, 1e-12))
    t = centroid_dst[0] - scale * (r @ centroid_src[0])
    return scale, r, t


def estimate_sim3_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    iterations: int = 1000,
    inlier_threshold: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """RANSAC Sim(3) estimate (diagnostic variant)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 points for Sim3 estimation.")
    if rng is None:
        rng = np.random.default_rng(0)
    best_count = -1
    best_model: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
    best_mask: Optional[np.ndarray] = None
    for _ in range(int(iterations)):
        subset_idx = rng.choice(src.shape[0], size=3, replace=False)
        try:
            scale, rot, trans = estimate_sim3_closed_form(src[subset_idx], dst[subset_idx])
        except np.linalg.LinAlgError:
            continue
        transformed = scale * (rot @ src.T).T + trans
        dists = np.linalg.norm(transformed - dst, axis=1)
        mask = dists < float(inlier_threshold)
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_model = (scale, rot, trans)
            best_mask = mask
    if best_model is None:
        return estimate_sim3_closed_form(src, dst)
    if best_mask is not None and int(best_mask.sum()) >= 3:
        return estimate_sim3_closed_form(src[best_mask], dst[best_mask])
    return best_model


def finite_correspondence_pairs(
    src: np.ndarray,
    dst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten ``[T, Q, 3]`` correspondences and drop non-finite pairs."""
    src_flat = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst_flat = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(src_flat).all(axis=-1) & np.isfinite(dst_flat).all(axis=-1)
    return src_flat[finite], dst_flat[finite]


def compute_average_pts_within_thresh(
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    scaling: str = "global",
    compute_epe: bool = True,
    apd_thresholds: Optional[Dict[int, float]] = None,
    rng: Optional[np.random.Generator] = None,
    align_query_mask: Optional[np.ndarray] = None,
) -> Tuple[
    float,
    np.ndarray,
    Dict[int, float],
    Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]],
    float,
]:
    """Compute APD, per-threshold fractions, and EPE after the chosen alignment.

    APD = ``mean(fractions)`` over :data:`eval4d.metrics.constants.APD_THRESHOLDS`
    (0.1 / 0.3 / 0.5 / 1.0 m). tau@0.1m = ``fractions[1]``.

    Returns:
        apd, pred_aligned, fractions, alignment_params, epe
    """
    if apd_thresholds is None:
        apd_thresholds = APD_THRESHOLDS
    if rng is None:
        rng = np.random.default_rng(0)

    gt = np.asarray(gt_points, dtype=np.float64)
    pred = np.asarray(pred_points, dtype=np.float64)
    params: Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]

    if scaling == "global":
        scale = compute_scale_factor_global(gt, pred, query_mask=align_query_mask)
        pred_aligned = pred * scale
        params = (scale, np.eye(3, dtype=np.float64), np.zeros((3,), dtype=np.float64))
    elif scaling == "per_traj":
        pred_aligned = scale_per_trajectory(gt, pred)
        params = (1.0, np.eye(3, dtype=np.float64), np.zeros((3,), dtype=np.float64))
    elif scaling == "sim3_closed":
        src_fit, dst_fit = finite_correspondence_pairs(pred, gt)
        if src_fit.shape[0] < 3:
            pred_aligned = np.full_like(pred, np.nan, dtype=np.float64)
            params = (None, None, None)
        else:
            scale, rot, trans = estimate_sim3_closed_form(src_fit, dst_fit)
            src = pred.reshape(-1, 3)
            finite = np.isfinite(src).all(axis=-1)
            pred_aligned = np.full_like(src, np.nan, dtype=np.float64)
            pred_aligned[finite] = (scale * (rot @ src[finite].T)).T + trans
            pred_aligned = pred_aligned.reshape(gt.shape)
            params = (scale, rot, trans)
    elif scaling == "sim3":
        src_fit, dst_fit = finite_correspondence_pairs(pred, gt)
        if src_fit.shape[0] < 3:
            pred_aligned = np.full_like(pred, np.nan, dtype=np.float64)
            params = (None, None, None)
        else:
            if src_fit.shape[0] > 16384:
                pick = rng.choice(src_fit.shape[0], size=16384, replace=False)
                src_sample = src_fit[pick]
                dst_sample = dst_fit[pick]
            else:
                src_sample = src_fit
                dst_sample = dst_fit
            scale, rot, trans = estimate_sim3_ransac(src_sample, dst_sample, rng=rng)
            src = pred.reshape(-1, 3)
            finite = np.isfinite(src).all(axis=-1)
            pred_aligned = np.full_like(src, np.nan, dtype=np.float64)
            pred_aligned[finite] = (scale * (rot @ src[finite].T)).T + trans
            pred_aligned = pred_aligned.reshape(gt.shape)
            params = (scale, rot, trans)
    else:
        raise ValueError(f"Unknown scaling: {scaling}")

    dists = np.linalg.norm(pred_aligned - gt, axis=-1)
    total_points = int(np.isfinite(dists).sum())
    fractions: Dict[int, float] = {}
    for thr_key, fixed_threshold in apd_thresholds.items():
        within_dist = np.isfinite(dists) & (dists <= float(fixed_threshold))
        fractions[int(thr_key)] = float(np.sum(within_dist) / max(total_points, 1))
    apd = float(np.mean(list(fractions.values()))) if fractions else float("nan")
    if compute_epe and np.any(np.isfinite(dists)):
        epe = float(np.mean(dists[np.isfinite(dists)]))
    else:
        epe = float("inf")
    return apd, pred_aligned, fractions, params, epe
