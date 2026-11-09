"""Frame and shared-chunk geometry scale decoders for GFM 1.1 checkpoints."""

import math

import torch
from torch import nn


class MetricCrossBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, query, context):
        context = self.context_norm(context)
        query = query + self.attn(
            self.query_norm(query), context, context, need_weights=False,
        )[0]
        return query + self.ffn(query)


class MetricHead(nn.Module):
    """Predict one positive global/pose ratio [B] over all views in a chunk."""

    def __init__(self, dim_in=3072, dim=512, num_heads=8, depth=5,
                 mlp_ratio=4, init_metric=1.0, max_abs_log_metric=15.0):
        super().__init__()
        if dim % num_heads or depth < 1 or init_metric <= 0 or max_abs_log_metric <= 0:
            raise ValueError("Invalid metric head dimensions or scale bounds")
        self.metric_token = nn.Parameter(torch.randn(1, 1, dim_in) * 1e-6)
        self.query_proj = nn.Linear(dim_in, dim)
        self.context_proj = nn.Linear(dim_in, dim)
        self.blocks = nn.ModuleList([
            MetricCrossBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.constant_(self.head.bias, math.log(init_metric))
        self.max_abs_log_metric = max_abs_log_metric

    def forward(self, features):
        features = features[-1] if isinstance(features, (list, tuple)) else features
        if features.ndim != 4:
            raise ValueError("MetricHead expects [B, views, tokens, channels]")
        b = features.shape[0]
        context = self.context_proj(features.flatten(1, 2))
        query = self.query_proj(self.metric_token.expand(b, -1, -1))
        for block in self.blocks:
            query = block(query, context)
        log_metric = self.head(self.norm(query)).float().reshape(b)
        log_metric = log_metric.clamp(-self.max_abs_log_metric, self.max_abs_log_metric)
        return {"log_metric": log_metric, "metric": log_metric.exp()}


class FrameMetricHead(MetricHead):
    """Apply the shared decoder independently per view; return [B,N] ratios."""

    def forward(self, features):
        features = features[-1] if isinstance(features, (list, tuple)) else features
        if features.ndim != 4:
            raise ValueError("FrameMetricHead expects [B, views, tokens, channels]")
        b, n, t, c = features.shape
        result = super().forward(features.reshape(b * n, 1, t, c))
        return {key: value.reshape(b, n) for key, value in result.items()}
