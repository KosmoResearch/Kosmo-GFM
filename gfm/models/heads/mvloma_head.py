"""MVLoMA shared-encoder inference decoder.

The checkpoint hierarchy is intentionally identical to the training head.
Only the final GFM feature level is consumed; no private DINO encoder or
training-framework import is needed. RGB for the VGG branch is raw [0, 1].
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from gfm.models.match.loma.descriptor.dedode import dedode_descriptor_G
from gfm.models.match.loma.device import amp_dtype, resolve_amp_dtype
from gfm.models.match.loma.detector.dad import DaD
from gfm.models.match.loma.loma import (
    LearnableFourierPositionalEncoding, MatchAssignment, TransformerLayer,
    apply_cached_rotary_emb,
)


class MVDescriptor(nn.Module):
    def __init__(self, fuse_dim: int, precision: str = "bf16"):
        super().__init__()
        self.amp_dtype = resolve_amp_dtype(precision)
        # auto preserves the historical GFM outer-FP16 projection exactly.
        self.projection_dtype = torch.float16 if precision == "auto" else self.amp_dtype
        self.encoder, self.decoder = dedode_descriptor_G(
            descriptor_dim=256,
            compute_dtype=self.amp_dtype,
        )
        self.mv_fuse_proj = nn.Conv2d(fuse_dim, 1024, 1)

    def forward(self, images, fuse_map, keypoints):
        features, sizes = self.encoder.vgg(images)
        return self.decode_features(features, fuse_map, keypoints)

    def decode_features(self, features, fuse_map, keypoints):
        """Single-view decoder; never mutate the cached VGG feature collection."""
        features = list(features)
        sizes = [feature.shape[-2:] for feature in features]
        with torch.autocast(device_type=fuse_map.device.type,
                            dtype=getattr(self, "projection_dtype", torch.float16),
                            enabled=fuse_map.is_cuda):
            projected = self.mv_fuse_proj(fuse_map)
        features.append(projected)
        sizes.append(projected.shape[-2:])
        descriptions, context = 0, None
        for index, (feature, scale) in enumerate(zip(reversed(features), self.decoder.scales)):
            delta, context = self.decoder(feature, scale=scale, context=context)
            descriptions = descriptions + delta
            if index < len(self.decoder.scales) - 1:
                size = sizes[-index - 2]
                descriptions = F.interpolate(descriptions, size=size, mode="bilinear", align_corners=False)
                context = F.interpolate(context, size=size, mode="bilinear", align_corners=False)
        return F.grid_sample(
            descriptions.float(), keypoints[:, None].float(),
            mode="bilinear", align_corners=False,
        )[:, :, 0].mT


class StarTransformerLayer(TransformerLayer):
    def forward(self, descs, encodings):
        # Same interleaved QKV layout as the training code. Contiguous V permits
        # fused SDPA rather than a quadratic-memory fallback at 2048 keypoints.
        views = []
        block = self.self_attn
        for desc, encoding in zip(descs, encodings):
            qkv = block.Wqkv(desc).unflatten(-1, (block.num_heads, -1, 3)).transpose(1, 2)
            q = apply_cached_rotary_emb(encoding, qkv[..., 0])
            k = apply_cached_rotary_emb(encoding, qkv[..., 1])
            context = F.scaled_dot_product_attention(q, k, qkv[..., 2].contiguous())
            message = block.out_proj(context.transpose(1, 2).flatten(-2))
            views.append(desc + block.ffn(torch.cat((desc, message), -1)))
        lengths = [view.shape[1] for view in views[1:]]
        anchor, neighbors = self.cross_attn(views[0], torch.cat(views[1:], dim=1))
        return [anchor, *neighbors.split(lengths, dim=1)]


class MVLoMa(nn.Module):
    def __init__(self, fuse_dim: int, precision: str = "bf16"):
        super().__init__()
        self._detector = DaD(DaD.Cfg(compile=False, precision=precision), init_device="cpu")
        self._descriptor = MVDescriptor(fuse_dim, precision=precision)
        self.input_proj = nn.Linear(256, 512)
        self.posenc = LearnableFourierPositionalEncoding(2, 64, gamma=1.0)
        self.transformers = nn.ModuleList(StarTransformerLayer(512, 8) for _ in range(9))
        self.log_assignment = nn.ModuleList(MatchAssignment(512) for _ in range(9))

    def match_descriptors(self, keypoints, descriptions):
        """Fixed-STAR tensor trunk; pair selection stays outside compilation."""
        descs = [self.input_proj(descriptions[:, i]) for i in range(keypoints.shape[1])]
        encodings = [self.posenc(keypoints[:, i]) for i in range(keypoints.shape[1])]
        for layer in self.transformers:
            descs = layer(descs, encodings)
        return descs


class MVLoMAHead(nn.Module):
    """Shared-encoder sparse decoder with strict training-checkpoint loading.

    The matcher consumes the complete STAR, in the encoder's view order.
    The supplied checkpoint uses no QK norm, view embeddings or token pruning.
    BF16 is the default for detector, descriptor, projection and matching AMP,
    independent of the shared GFM encoder's FP16 autocast.
    """
    def __init__(self, fuse_dim=3072, patch_size=14, precision="bf16"):
        super().__init__()
        self.amp_dtype = resolve_amp_dtype(precision)
        self.precision = precision
        self.fuse_dim = fuse_dim
        self.patch_size = patch_size
        # Preserve the checkpoint parameter prefix loma_match_head.loma.loma.*.
        self.loma = nn.Module()
        self.loma.loma = MVLoMa(fuse_dim, precision=precision)
        # Plain callable, not a registered module: checkpoint keys stay intact.
        self._compiled_matcher = None

    def load_checkpoint(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        for wrapper in ("model", "state_dict", "module"):
            if isinstance(state.get(wrapper), dict):
                state = state[wrapper]
        for prefix in ("module.", "_orig_mod.", "model."):
            while state and all(key.startswith(prefix) for key in state):
                state = {key[len(prefix):]: value for key, value in state.items()}
        prefix = "loma_match_head."
        if any(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        # Never silently accept partial heads or overwrite the GFM backbone.
        self.load_state_dict(state, strict=True)

    @torch.no_grad()
    def detect(self, images, num_keypoints=2048):
        return self.loma.loma._detector(images, num_keypoints=num_keypoints)["keypoints"]

    def describe(self, images, patch_features, patch_start_idx, keypoints,
                 descriptor_images=None, transforms=None):
        b, n, _, h, w = images.shape
        feature = patch_features[-1][:, :, patch_start_idx:]
        ph, pw = h // self.patch_size, w // self.patch_size
        if feature.shape != (b, n, ph * pw, self.fuse_dim):
            raise ValueError(f"Unexpected MVLoMA patch features: {tuple(feature.shape)}")
        if keypoints.ndim != 4 or keypoints.shape[:2] != (b, n) or keypoints.shape[-1] != 2:
            raise ValueError("MVLoMA keypoints must be [B,N,K,2]")
        descriptor_images = images if descriptor_images is None else descriptor_images
        if descriptor_images.shape[:3] != (b, n, 3):
            raise ValueError("MVLoMA descriptor images must be [B,N,3,H,W]")
        if transforms is not None and len(transforms) != n:
            raise ValueError("Expected one preprocessing transform per view")
        feature = feature.reshape(b, n, ph, pw, self.fuse_dim).permute(0, 1, 4, 2, 3).contiguous()
        descriptions = []
        # Decode one view at a time: the full 784-square, 256-channel grid is
        # large; only the sampled descriptors survive this loop.
        for view in range(n):
            fuse_map = feature[:, view]
            if transforms is not None:
                t = transforms[view]
                rh, rw = t.resized_hw
                if (rh, rw) != (h, w) or t.pad_left or t.pad_top:
                    # Undo centered padding, including sub-patch offsets, before
                    # fusing with the square descriptor of the original image.
                    theta = fuse_map.new_tensor([
                        [rw / w, 0, (2 * t.pad_left + rw) / w - 1],
                        [0, rh / h, (2 * t.pad_top + rh) / h - 1],
                    ]).unsqueeze(0).expand(b, -1, -1)
                    grid = F.affine_grid(theta, (b, self.fuse_dim, rh // self.patch_size,
                                                rw // self.patch_size), align_corners=False)
                    fuse_map = F.grid_sample(fuse_map, grid, align_corners=False, padding_mode="border")
            descriptions.append(self.loma.loma._descriptor(
                descriptor_images[:, view].to(images.device), fuse_map, keypoints[:, view],
            ))
        return torch.stack(descriptions, dim=1)

    def match(self, keypoints, descriptions, pair_idx=None):
        n = keypoints.shape[1]
        if n < 2:
            raise ValueError(f"MVLoMA matcher expects at least 2 views, got {n}")
        if keypoints.shape[:3] != descriptions.shape[:3]:
            raise ValueError("Keypoint and descriptor layouts differ")
        pairs = [(0, i) for i in range(1, n)] if pair_idx is None else [tuple(p) for p in pair_idx]
        if any(a == z or min(a, z) < 0 or max(a, z) >= n for a, z in pairs):
            raise ValueError("Invalid MVLoMA pair index")
        core = self.loma.loma
        with torch.autocast(device_type=descriptions.device.type, dtype=getattr(self, "amp_dtype", amp_dtype),
                            enabled=descriptions.is_cuda):
            matcher = self._compiled_matcher or core.match_descriptors
            descs = matcher(keypoints, descriptions)
            scores = {pair: core.log_assignment[-1](descs[pair[0]], descs[pair[1]])[0]
                      for pair in pairs}
        return {"scores_by_pair": scores, "pair_idx": pairs}

    def forward(self, images, patch_features, patch_start_idx, keypoints,
                descriptor_images=None, transforms=None, pair_idx=None, describe_only=False):
        descriptions = self.describe(images, patch_features, patch_start_idx, keypoints,
                                     descriptor_images, transforms)
        result = {"keypoints": keypoints, "descriptions": descriptions}
        if not describe_only:
            result.update(self.match(keypoints, descriptions, pair_idx))
        return result
