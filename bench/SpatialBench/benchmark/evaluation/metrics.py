"""Reference SpatialBench depth, pairwise camera, and trajectory metrics."""
import numpy as np
from itertools import combinations


def compute_depth_metrics(pred, gt, valid_mask):
    """Compute the standard depth evaluation metrics.

    Args:
        pred: (H, W) predicted depth (already aligned)
        gt: (H, W) GT depth
        valid_mask: (H, W) bool

    Returns:
        dict: abs_rel, sq_rel, rmse, log_rmse,
              delta_1.25, delta_1.25^2, delta_1.25^3
    """
    pred = np.asanyarray(pred)
    gt = np.asanyarray(gt)
    vm = np.asarray(valid_mask, dtype=bool)
    print(
        f"[metric:depth] pred shape={pred.shape} dtype={pred.dtype} | "
        f"gt shape={gt.shape} dtype={gt.dtype} | "
        f"valid_mask shape={vm.shape} valid={int(vm.sum())}/{vm.size}"
    )
    if pred.shape == vm.shape and gt.shape == vm.shape and vm.any():
        pv, gv = pred[vm], gt[vm]
        print(
            f"[metric:depth] masked pixels n={pv.size} | "
            f"pred min/max/mean={np.nanmin(pv):.6g}/{np.nanmax(pv):.6g}/{np.nanmean(pv):.6g} | "
            f"gt min/max/mean={np.nanmin(gv):.6g}/{np.nanmax(gv):.6g}/{np.nanmean(gv):.6g}"
        )

    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]

    if len(pred_valid) == 0:
        return {k: float('nan') for k in [
            'abs_rel', 'sq_rel', 'rmse', 'log_rmse',
            'delta_1.25', 'delta_1.25^2', 'delta_1.25^3',
            'inlier_1.03', 'inlier_1.05', 'inlier_1.10',
        ]}

    # Filter out zero and negative values
    mask = (pred_valid > 1e-6) & (gt_valid > 1e-6)
    pred_valid = pred_valid[mask]
    gt_valid = gt_valid[mask]

    if len(pred_valid) == 0:
        return {k: float('nan') for k in [
            'abs_rel', 'sq_rel', 'rmse', 'log_rmse',
            'delta_1.25', 'delta_1.25^2', 'delta_1.25^3',
            'inlier_1.03', 'inlier_1.05', 'inlier_1.10',
        ]}

    # AbsRel
    abs_rel = float(np.mean(np.abs(pred_valid - gt_valid) / gt_valid))

    # SqRel
    sq_rel = float(np.mean(((pred_valid - gt_valid) ** 2) / gt_valid))

    # RMSE
    rmse = float(np.sqrt(np.mean((pred_valid - gt_valid) ** 2)))

    # Log RMSE
    log_rmse = float(np.sqrt(np.mean((np.log(pred_valid) - np.log(gt_valid)) ** 2)))

    # Delta thresholds
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    delta_1 = float(np.mean(ratio < 1.25))
    delta_2 = float(np.mean(ratio < 1.25 ** 2))
    delta_3 = float(np.mean(ratio < 1.25 ** 3))

    # Inlier Ratio (RobustMVD): max(pred/gt, gt/pred) < τ
    inlier_1_03 = float(np.mean(ratio < 1.03))
    inlier_1_05 = float(np.mean(ratio < 1.05))
    inlier_1_10 = float(np.mean(ratio < 1.10))

    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'log_rmse': log_rmse,
        'delta_1.25': delta_1,
        'delta_1.25^2': delta_2,
        'delta_1.25^3': delta_3,
        'inlier_1_03': inlier_1_03,
        'inlier_1_05': inlier_1_05,
        'inlier_1_10': inlier_1_10,
    }


def _invert_se3_batch(se3_44):
    """Invert (N, 4, 4) SE3 numpy batch."""
    out = se3_44.copy()
    R = se3_44[:, :3, :3]
    t = se3_44[:, :3, 3:]
    R_inv = R.transpose(0, 2, 1)             # (N, 3, 3)
    t_inv = -np.matmul(R_inv, t)              # (N, 3, 1)
    out[:, :3, :3] = R_inv
    out[:, :3, 3:] = t_inv
    return out


def _mat_to_quat(R):
    """Rotation matrix (N, 3, 3) -> unit quaternion (N, 4).  Shepperd method."""
    N = R.shape[0]
    q = np.zeros((N, 4), dtype=np.float64)

    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    trace = m00 + m11 + m22

    # case 0: trace > 0
    c0 = trace > 0
    if c0.any():
        s = np.sqrt(np.maximum(trace[c0] + 1.0, 1e-10)) * 2
        q[c0, 0] = 0.25 * s
        q[c0, 1] = (R[c0, 2, 1] - R[c0, 1, 2]) / s
        q[c0, 2] = (R[c0, 0, 2] - R[c0, 2, 0]) / s
        q[c0, 3] = (R[c0, 1, 0] - R[c0, 0, 1]) / s

    # case 1: m00 largest diagonal
    c1 = ~c0 & (m00 > m11) & (m00 > m22)
    if c1.any():
        s = np.sqrt(np.maximum(1.0 + m00[c1] - m11[c1] - m22[c1], 1e-10)) * 2
        q[c1, 0] = (R[c1, 2, 1] - R[c1, 1, 2]) / s
        q[c1, 1] = 0.25 * s
        q[c1, 2] = (R[c1, 0, 1] + R[c1, 1, 0]) / s
        q[c1, 3] = (R[c1, 0, 2] + R[c1, 2, 0]) / s

    # case 2: m11 largest diagonal
    c2 = ~c0 & ~c1 & (m11 > m22)
    if c2.any():
        s = np.sqrt(np.maximum(1.0 + m11[c2] - m00[c2] - m22[c2], 1e-10)) * 2
        q[c2, 0] = (R[c2, 0, 2] - R[c2, 2, 0]) / s
        q[c2, 1] = (R[c2, 0, 1] + R[c2, 1, 0]) / s
        q[c2, 2] = 0.25 * s
        q[c2, 3] = (R[c2, 1, 2] + R[c2, 2, 1]) / s

    # case 3: m22 largest diagonal
    c3 = ~c0 & ~c1 & ~c2
    if c3.any():
        s = np.sqrt(np.maximum(1.0 + m22[c3] - m00[c3] - m11[c3], 1e-10)) * 2
        q[c3, 0] = (R[c3, 1, 0] - R[c3, 0, 1]) / s
        q[c3, 1] = (R[c3, 0, 2] + R[c3, 2, 0]) / s
        q[c3, 2] = (R[c3, 1, 2] + R[c3, 2, 1]) / s
        q[c3, 3] = 0.25 * s

    norms = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norms, 1e-10)
    return q


def _rotation_angle_quat(R_gt, R_pred, eps=1e-15):
    """Quaternion-based rotation error (degrees). Matches DA3 rotation_angle."""
    q_pred = _mat_to_quat(R_pred)
    q_gt = _mat_to_quat(R_gt)
    dot2 = np.sum(q_pred * q_gt, axis=1) ** 2
    loss_q = np.clip(1.0 - dot2, eps, None)
    err_q = np.arccos(np.clip(1.0 - 2.0 * loss_q, -1.0, 1.0))
    return np.degrees(err_q)


def _translation_angle(t_gt, t_pred, eps=1e-15):
    """Translation angle error (degrees) with ambiguity handling. Matches DA3."""
    t_pred_n = t_pred / (np.linalg.norm(t_pred, axis=1, keepdims=True) + eps)
    t_gt_n = t_gt / (np.linalg.norm(t_gt, axis=1, keepdims=True) + eps)
    dot2 = np.sum(t_pred_n * t_gt_n, axis=1) ** 2
    loss_t = np.clip(1.0 - dot2, eps, None)
    err_t = np.degrees(np.arccos(np.clip(np.sqrt(1.0 - loss_t), -1.0, 1.0)))
    err_t[~np.isfinite(err_t)] = 1e6
    # ambiguity: min(angle, 180 - angle)
    err_t = np.minimum(err_t, np.abs(180.0 - err_t))
    return err_t


def _calculate_auc_np(r_error, t_error, max_threshold=30):
    """AUC via histogram bins. Matches DA3 calculate_auc_np."""
    error_matrix = np.stack([r_error, t_error], axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return float(np.mean(np.cumsum(normalized_histogram)))


def cameras_evaluation(gt_w2c, pred_w2c, num_frames):
    """Pose evaluation fully aligned with DA3 cameras_evaluation (pure numpy).

    Args:
        gt_w2c: (N, 3, 4) w2c numpy
        pred_w2c: (N, 3, 4) w2c numpy
        num_frames: int

    Returns:
        Racc_5, Tacc_5, Racc_3, Tacc_3, rError (np), tError (np)
    """
    # (N, 3, 4) -> (N, 4, 4)
    def to_44(m34):
        N = m34.shape[0]
        m44 = np.zeros((N, 4, 4), dtype=np.float64)
        m44[:, :3, :] = m34
        m44[:, 3, 3] = 1.0
        return m44

    pred_se3 = to_44(pred_w2c.astype(np.float64))
    gt_se3 = to_44(gt_w2c.astype(np.float64))

    # build all pair indices
    pairs = np.array(list(combinations(range(num_frames), 2)))
    i1, i2 = pairs[:, 0], pairs[:, 1]

    # relative poses: inv(T_i) @ T_j
    rel_gt = np.matmul(_invert_se3_batch(gt_se3[i1]), gt_se3[i2])
    rel_pred = np.matmul(_invert_se3_batch(pred_se3[i1]), pred_se3[i2])

    rel_rangle_deg = _rotation_angle_quat(rel_gt[:, :3, :3], rel_pred[:, :3, :3])
    rel_tangle_deg = _translation_angle(rel_gt[:, :3, 3], rel_pred[:, :3, 3])

    Racc_5 = float(np.mean(rel_rangle_deg < 5))
    Tacc_5 = float(np.mean(rel_tangle_deg < 5))
    Racc_3 = float(np.mean(rel_rangle_deg < 3))
    Tacc_3 = float(np.mean(rel_tangle_deg < 3))

    return (Racc_5, Tacc_5, Racc_3, Tacc_3,
            rel_rangle_deg.astype(np.float32),
            rel_tangle_deg.astype(np.float32))


def compute_pose_metrics(pred_w2c, gt_w2c):
    """Compute camera pose evaluation metrics (aligned with DA3 cameras_evaluation).

    Args:
        pred_w2c: (N, 3, 4) world-to-camera numpy
        gt_w2c: (N, 3, 4) world-to-camera numpy (already normalized by normalize_gt_poses)

    Returns:
        dict: racc_3, racc_5, tacc_3, tacc_5, auc_3, auc_5, auc_15, auc_30
    """
    pw = np.asarray(pred_w2c)
    gw = np.asarray(gt_w2c)
    N = len(pred_w2c)
    print(
        f"[metric:pose] pred_w2c shape={pw.shape} dtype={pw.dtype} | "
        f"gt_w2c shape={gw.shape} dtype={gw.dtype} | N={N}"
    )

    nan_result = {k: float('nan') for k in [
        'racc_3', 'racc_5', 'tacc_3', 'tacc_5',
        'auc_3', 'auc_5', 'auc_15', 'auc_30'
    ]}
    if N < 2:
        return nan_result

    try:
        Racc_5, Tacc_5, Racc_3, Tacc_3, rError, tError = cameras_evaluation(
            gt_w2c, pred_w2c, num_frames=N
        )
    except Exception:
        return nan_result

    return {
        'racc_3': float(Racc_3),
        'racc_5': float(Racc_5),
        'tacc_3': float(Tacc_3),
        'tacc_5': float(Tacc_5),
        'auc_3': _calculate_auc_np(rError, tError, max_threshold=3),
        'auc_5': _calculate_auc_np(rError, tError, max_threshold=5),
        'auc_15': _calculate_auc_np(rError, tError, max_threshold=15),
        'auc_30': _calculate_auc_np(rError, tError, max_threshold=30),
    }


def _c2w_to_se3_list(c2w):
    """(N, 3, 4) c2w -> list of (4, 4) SE3 numpy matrices."""
    poses = []
    for i in range(len(c2w)):
        m = np.eye(4, dtype=np.float64)
        m[:3, :] = c2w[i]
        poses.append(m)
    return poses


def compute_trajectory_metrics(pred_c2w, gt_c2w):
    """Compute trajectory metrics after Sim(3) alignment: ATE, RPEt, RPEr.

    Uses the evo library (DROID-SLAM evaluation standard):
    - ATE: APE translation_part (RMSE), Sim(3) aligned
    - RPE_t: RPE translation_part (mean), delta=1 frame
    - RPE_r: RPE rotation_angle_deg (mean), delta=1 frame
      (cf. DROID-SLAM issue #156, using evo's rotation_angle_deg)

    Args:
        pred_c2w: (N, 3, 4) cam-to-world numpy
        gt_c2w: (N, 3, 4) cam-to-world numpy

    Returns:
        dict: ate, rpe_t, rpe_r
    """
    nan_result = {k: float('nan') for k in ['ate', 'rpe_t', 'rpe_r']}
    pp = np.asarray(pred_c2w)
    gg = np.asarray(gt_c2w)
    N = len(pred_c2w)
    if N >= 2:
        p_t = pp[:, :3, 3].astype(np.float64)
        g_t = gg[:, :3, 3].astype(np.float64)
        print(
            f"[metric:trajectory] pred_c2w shape={pp.shape} dtype={pp.dtype} | "
            f"gt_c2w shape={gg.shape} dtype={gg.dtype} | N={N} | "
            f"pred t min/max/mean={p_t.min():.6g}/{p_t.max():.6g}/{p_t.mean():.6g} | "
            f"gt t min/max/mean={g_t.min():.6g}/{g_t.max():.6g}/{g_t.mean():.6g}"
        )
    else:
        print(
            f"[metric:trajectory] pred_c2w shape={pp.shape} dtype={pp.dtype} | "
            f"gt_c2w shape={gg.shape} dtype={gg.dtype} | N={N} (skip, N<2)"
        )

    if N < 2:
        return nan_result

    try:
        import copy
        from evo.core.trajectory import PosePath3D
        from evo.core import metrics
        from evo.core.metrics import PoseRelation, StatisticsType, Unit

        traj_pred = PosePath3D(poses_se3=_c2w_to_se3_list(pred_c2w))
        traj_gt = PosePath3D(poses_se3=_c2w_to_se3_list(gt_c2w))

        # Sim(3) alignment: Umeyama align + scale correction
        traj_pred_aligned = copy.deepcopy(traj_pred)
        traj_pred_aligned.align(traj_gt, correct_scale=True)

        # ATE: APE translation_part, RMSE
        ape_metric = metrics.APE(PoseRelation.translation_part)
        ape_metric.process_data((traj_gt, traj_pred_aligned))
        ate = ape_metric.get_statistic(StatisticsType.rmse)

        # RPE translation: delta=1 frame, mean
        rpe_t_metric = metrics.RPE(
            PoseRelation.translation_part,
            delta=1, delta_unit=Unit.frames, all_pairs=False)
        rpe_t_metric.process_data((traj_gt, traj_pred_aligned))
        rpe_t = rpe_t_metric.get_statistic(StatisticsType.mean)

        # RPE rotation: rotation_angle_deg, delta=1 frame, mean
        rpe_r_metric = metrics.RPE(
            PoseRelation.rotation_angle_deg,
            delta=1, delta_unit=Unit.frames, all_pairs=False)
        rpe_r_metric.process_data((traj_gt, traj_pred_aligned))
        rpe_r = rpe_r_metric.get_statistic(StatisticsType.mean)

        return {
            'ate': float(ate),
            'rpe_t': float(rpe_t),
            'rpe_r': float(rpe_r),
        }
    except ImportError:
        # Fall back to the numpy implementation when evo is not installed
        return _compute_trajectory_metrics_numpy(pred_c2w, gt_c2w)
    except Exception:
        # evo's Umeyama alignment throws GeometryException on degenerate trajectories
        # (coplanar/colinear); fall back to the numpy implementation.
        return _compute_trajectory_metrics_numpy(pred_c2w, gt_c2w)


def _compute_trajectory_metrics_numpy(pred_c2w, gt_c2w):
    """numpy fallback (used when evo is not installed)."""
    nan_result = {k: float('nan') for k in ['ate', 'rpe_t', 'rpe_r']}
    N = len(pred_c2w)
    if N < 2:
        return nan_result

    try:
        pred_pos = pred_c2w[:, :3, 3].astype(np.float64)
        gt_pos = gt_c2w[:, :3, 3].astype(np.float64)

        # Umeyama Sim(3) alignment
        pred, gt = pred_pos, gt_pos
        mu_p, mu_g = pred.mean(0), gt.mean(0)
        pc, gc = pred - mu_p, gt - mu_g
        H = pc.T @ gc / N
        U, D, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        S = np.diag([1.0, 1.0, np.sign(d)])
        R_align = Vt.T @ S @ U.T
        var_p = np.sum(pc ** 2) / N
        s = np.trace(np.diag(D) @ S) / max(var_p, 1e-10)
        t_align = mu_g - s * R_align @ mu_p

        # Aligned c2w
        pred_64 = pred_c2w.astype(np.float64)
        aligned = np.zeros_like(pred_64)
        for i in range(N):
            aligned[i, :3, :3] = R_align @ pred_64[i, :3, :3]
            aligned[i, :3, 3] = s * R_align @ pred_64[i, :3, 3] + t_align

        # ATE
        ate = float(np.sqrt(np.mean(np.sum(
            (aligned[:, :3, 3] - gt_pos) ** 2, axis=1))))

        # RPE (consecutive frames)
        def to_44(m34):
            m44 = np.zeros((m34.shape[0], 4, 4), dtype=np.float64)
            m44[:, :3, :] = m34
            m44[:, 3, 3] = 1.0
            return m44

        a44 = to_44(aligned)
        g44 = to_44(gt_c2w.astype(np.float64))
        rpe_t_list, rpe_r_list = [], []
        for i in range(N - 1):
            rel_p = np.linalg.inv(a44[i]) @ a44[i + 1]
            rel_g = np.linalg.inv(g44[i]) @ g44[i + 1]
            rpe = np.linalg.inv(rel_g) @ rel_p
            rpe_t_list.append(np.linalg.norm(rpe[:3, 3]))
            cos_a = np.clip((np.trace(rpe[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            rpe_r_list.append(float(np.degrees(np.arccos(cos_a))))

        return {
            'ate': ate,
            'rpe_t': float(np.mean(rpe_t_list)),
            'rpe_r': float(np.mean(rpe_r_list)),
        }
    except Exception:
        return nan_result
