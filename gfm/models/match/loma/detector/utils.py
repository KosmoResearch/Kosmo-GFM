import torch
import torch.nn as nn
import torch.nn.functional as F
from ..geometry import kde, get_normalized_grid

def extract_patches_from_inds(x: torch.Tensor, inds: torch.Tensor, patch_size: int):
    B, H, W = x.shape
    B, N = inds.shape
    unfolder = nn.Unfold(kernel_size=patch_size, padding=patch_size // 2, stride=1)
    unfolded_x: torch.Tensor = unfolder(x[:, None])  # B x K_H * K_W x H * W
    patches = torch.gather(
        unfolded_x,
        dim=2,
        index=inds[:, None, :].expand(B, patch_size**2, N),
    )  # B x K_H * K_W x N
    return patches


def sample_keypoints(
    keypoint_probs: torch.Tensor,
    num_samples=8192,
    use_nms=True,
    nms_size=1,
    sample_topk=True,
    increase_coverage=True,
    remove_borders=False,
    return_probs=False,
    coverage_pow=1 / 2,
    coverage_size=51,
    subpixel=False,
    scoremap=None,  # required for subpixel
    subpixel_temp=0.5,
    coverage_from_sparse=False,
):
    device = keypoint_probs.device
    B, H, W = keypoint_probs.shape
    if coverage_from_sparse:
        non_coverage_points, non_coverage_probs = sample_keypoints(
            keypoint_probs,
            num_samples=4 * num_samples,
            use_nms=use_nms,
            nms_size=nms_size,
            sample_topk=sample_topk,
            remove_borders=remove_borders,
            coverage_pow=coverage_pow,
            coverage_size=coverage_size,
            subpixel=subpixel,
            scoremap=scoremap,
            subpixel_temp=subpixel_temp,
            increase_coverage=False,
            coverage_from_sparse=False,
            return_probs=True,
        )
        density = kde(non_coverage_points)
        balance = 1 / (density + 1)
        balance[density < 10] = 1e-7
        p: torch.Tensor = non_coverage_probs * balance
        inds = p.argsort(dim=-1, descending=True)[:, :num_samples]
        # print(non_coverage_points.shape, inds.shape)
        kps = torch.gather(
            non_coverage_points,
            dim=1,
            index=inds[..., None].expand(B, num_samples, 2),
        )

        if return_probs:
            return kps, torch.gather(non_coverage_probs, dim=1, index=inds)
        return kps
    if increase_coverage and not coverage_from_sparse:
        weights = (
            -(torch.linspace(-2, 2, steps=coverage_size, device=device) ** 2)
        ).exp()[None, None]
        # 10000 is just some number for maybe numerical stability, who knows. :), result is invariant anyway
        local_density_x = F.conv2d(
            (keypoint_probs[:, None] + 1e-6) * 10000,
            weights[..., None, :],
            padding=(0, coverage_size // 2),
        )
        local_density = F.conv2d(
            local_density_x, weights[..., None], padding=(coverage_size // 2, 0)
        )[:, 0]
        keypoint_probs = keypoint_probs * (local_density + 1e-8) ** (-coverage_pow)
    grid = get_normalized_grid(B, H, W, overload_device=device).reshape(B, H * W, 2)
    if use_nms:
        keypoint_probs = keypoint_probs * (
            keypoint_probs
            == F.max_pool2d(keypoint_probs, nms_size, stride=1, padding=nms_size // 2)
        )
    if remove_borders:
        frame = torch.zeros_like(keypoint_probs)
        # we hardcode 4px, could do it nicer, but whatever
        frame[..., 4:-4, 4:-4] = 1
        keypoint_probs = keypoint_probs * frame
    if sample_topk:
        inds = torch.topk(keypoint_probs.reshape(B, H * W), k=num_samples).indices
    else:
        inds = torch.multinomial(
            keypoint_probs.reshape(B, H * W), num_samples=num_samples, replacement=False
        )
    kps = torch.gather(grid, dim=1, index=inds[..., None].expand(B, num_samples, 2))
    if subpixel:
        if scoremap is None:
            raise ValueError("scoremap is required when subpixel=True")
        offsets = get_normalized_grid(
            B, nms_size, nms_size, overload_device=device
        ).reshape(B, nms_size**2, 2)  # B x K_H x K_W x 2
        offsets[..., 0] = offsets[..., 0] * nms_size / W
        offsets[..., 1] = offsets[..., 1] * nms_size / H
        keypoint_patch_scores = extract_patches_from_inds(scoremap, inds, nms_size)
        keypoint_patch_probs = (keypoint_patch_scores / subpixel_temp).softmax(
            dim=1
        )  # B x K_H * K_W x N
        keypoint_offsets = torch.einsum("bkn, bkd ->bnd", keypoint_patch_probs, offsets)
        kps = kps + keypoint_offsets
    if return_probs:
        return kps, torch.gather(keypoint_probs.reshape(B, H * W), dim=1, index=inds)
    return kps
