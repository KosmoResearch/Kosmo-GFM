"""Common camera conventions for GFM evaluation."""
import numpy as np


class ModelAdapter:


    def supports_metric_depth(self):
        """Whether the model outputs metric-scale depth (no alignment required).
        Default False (relative depth, requires median-scale alignment).
        """
        return False


    def get_model_params(self):
        """Compute model parameter counts and return as a dict.

        Returns:
            dict: {
                "total_params": int,        # total parameter count
                "trainable_params": int,     # trainable parameter count
                "frozen_params": int,        # frozen parameter count
                "total_params_M": float,     # total parameter count (millions)
            }
            Returns an empty dict if the model is not loaded or is not an nn.Module.
        """
        model = getattr(self, 'model', None)
        if model is None:
            return {}
        try:
            import torch.nn as nn
            if not isinstance(model, nn.Module):
                return {}
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen = total - trainable
            return {
                "total_params": total,
                "trainable_params": trainable,
                "frozen_params": frozen,
                "total_params_M": round(total / 1e6, 2),
            }
        except Exception:
            return {}


    def normalize_gt_poses(self, scene):
        """Normalize GT poses to the same coordinate frame as the model prediction and return w2c.

        Different models may use different normalization strategies (align first, scale by points, etc.).
        The default does no processing and simply converts GT c2w to w2c and returns it.
        Subclasses may override as needed.

        Args:
            scene: dict from BenchmarkDataset, containing extrinsic, depth, intrinsic, valid_mask, world_points, etc.

        Returns:
            np.ndarray (N, 3, 4) normalized GT world-to-camera poses
        """
        return self._invert_se3(scene["extrinsic"])


    @staticmethod
    def _invert_se3(se3_34):
        """Invert (N, 3, 4) SE3 matrices → (N, 3, 4).  R' = R^T, t' = -R^T @ t."""
        N = se3_34.shape[0]
        out = np.zeros_like(se3_34)
        for i in range(N):
            R, t = se3_34[i, :3, :3], se3_34[i, :3, 3]
            out[i, :3, :3] = R.T
            out[i, :3, 3] = -R.T @ t
        return out


    @staticmethod
    def normalize_camera_extrinsics_and_points(gt_c2w, world_points, valid_mask):
        """Normalization aligned with DA3's training-time normalize_camera_extrinsics_and_points_batch.

        Align to first camera + scale by avg point distance.
        Input is in c2w format (benchmark standard); internally converts to w2c, computes, and returns normalized w2c.

        Args:
            gt_c2w: (N, 3, 4) cam2world
            world_points: (N, H, W, 3) world-coordinate points
            valid_mask: (N, H, W) bool

        Returns:
            gt_w2c_norm: (N, 3, 4) normalized world-to-camera
        """
        N = gt_c2w.shape[0]

        # c2w -> w2c
        gt_w2c = np.zeros((N, 3, 4), dtype=np.float32)
        for i in range(N):
            R, t = gt_c2w[i, :3, :3], gt_c2w[i, :3, 3]
            gt_w2c[i, :3, :3] = R.T
            gt_w2c[i, :3, 3] = -R.T @ t

        # --- 1) align to first camera: T_i' = T_i @ T_0^{-1} ---
        # T_0^{-1} = c2w of first camera
        R0, t0 = gt_w2c[0, :3, :3], gt_w2c[0, :3, 3]

        w2c_44 = np.zeros((N, 4, 4), dtype=np.float32)
        w2c_44[:, :3, :] = gt_w2c
        w2c_44[:, 3, 3] = 1.0

        inv0 = np.eye(4, dtype=np.float32)
        inv0[:3, :3] = R0.T
        inv0[:3, 3] = -R0.T @ t0
        new_w2c_44 = w2c_44 @ inv0  # (N, 4, 4)

        # --- 2) world points -> first camera coord system ---
        # new_wp = R0 @ world_points + t0  (equivalent to DA3's world_points @ R.T + t)
        new_wp = np.einsum('ij,nhwj->nhwi', R0, world_points) + t0  # (N, H, W, 3)

        # --- 3) scale by avg point distance ---
        dist = np.linalg.norm(new_wp, axis=-1)  # (N, H, W)
        mask_f = valid_mask.astype(np.float32)
        dist_sum = (dist * mask_f).sum()
        valid_count = mask_f.sum()
        avg_scale = np.clip(dist_sum / (valid_count + 1e-3), 1e-6, 1e6)

        new_w2c_44[:, :3, 3] /= avg_scale

        # Return normalized w2c (N, 3, 4)
        return new_w2c_44[:, :3, :].copy()
