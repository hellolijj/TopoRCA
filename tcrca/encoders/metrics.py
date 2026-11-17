from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class MetricEncoder(nn.Module):
    def __init__(
        self,
        metric_dim: int,
        time_steps: int,
        out_dim: int,
        num_scales: int = 8,
        scale_factor: float = 2.0,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if time_steps < 1:
            raise ValueError("time_steps must be positive")
        self.num_scales = max(1, int(num_scales))
        self.time_steps = int(time_steps)
        self.out_dim = int(out_dim)
        self.configured_scale_factor = float(scale_factor)
        self.scale_factor, self.scale_lengths = self._resolve_scales()

        self.input_proj = nn.Linear(metric_dim, out_dim)
        self.input_norm = nn.LayerNorm(out_dim)
        self.scale_embed = nn.Parameter(torch.zeros(self.num_scales, out_dim))
        nn.init.normal_(self.scale_embed, std=0.02)
        self.max_len = int(time_steps)
        self.position = nn.Parameter(torch.zeros(1, self.max_len, out_dim))
        nn.init.normal_(self.position, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=heads,
            dim_feedforward=2 * out_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.fusion_logits = nn.Parameter(torch.zeros(self.num_scales))
        self.out_norm = nn.LayerNorm(out_dim)

    def _resolve_scales(self) -> Tuple[float, List[int]]:
        if self.num_scales == 1:
            return self.configured_scale_factor, [self.time_steps]
        if self.configured_scale_factor <= 1.0:
            raise ValueError("scale_factor must be greater than 1")
        if self.num_scales > self.time_steps:
            raise ValueError("num_scales cannot exceed time_steps")

        base = self.configured_scale_factor
        if base ** (self.num_scales - 1) > self.time_steps:
            base = float(self.time_steps) ** (1.0 / (self.num_scales - 1))
        lengths = [
            max(1, min(self.time_steps, int(round(self.time_steps / (base**k)))))
            for k in range(self.num_scales)
        ]
        lengths[0] = self.time_steps
        for k in range(self.num_scales - 2, -1, -1):
            lengths[k] = max(lengths[k], lengths[k + 1] + 1)
        if lengths[0] > self.time_steps:
            raise ValueError("num_scales cannot exceed time_steps")
        if any(lengths[k] <= lengths[k + 1] for k in range(self.num_scales - 1)):
            raise ValueError("metric scales must have distinct resolutions")
        return float(base), lengths

    def _pad_to(self, x: torch.Tensor, length: int) -> torch.Tensor:
        if x.shape[1] == length:
            return x
        if x.shape[1] > length:
            return x[:, :length]
        return F.pad(x.transpose(1, 2), (0, length - x.shape[1])).transpose(1, 2)

    def forward(self, metrics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        h = self.input_norm(self.input_proj(metrics))
        h = self._pad_to(h, self.max_len)

        scale_reps: List[torch.Tensor] = []
        for k, length in enumerate(self.scale_lengths):
            if length == h.shape[1]:
                pooled = h
            else:
                pooled = F.adaptive_avg_pool1d(h.transpose(1, 2), length).transpose(1, 2)
            pooled = pooled + self.position[:, : pooled.shape[1]]
            pooled = pooled + self.scale_embed[k].view(1, 1, -1)
            encoded = self.encoder(pooled)
            scale_reps.append(encoded.mean(dim=1))

        stacked = torch.stack(scale_reps, dim=1)
        weights = torch.softmax(self.fusion_logits, dim=0)
        fused = (stacked * weights.view(1, -1, 1)).sum(dim=1)
        return self.out_norm(fused), weights, scale_reps
