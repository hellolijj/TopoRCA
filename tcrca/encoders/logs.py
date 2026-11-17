from __future__ import annotations

import math

import torch
from torch import nn


class LogEncoder(nn.Module):
    def __init__(
        self,
        template_dim: int,
        hidden_dim: int,
        out_dim: int,
        max_events: int,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_events = int(max_events)
        self.hidden_dim = int(hidden_dim)
        self.input_proj = nn.Linear(1 + template_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, max_events, hidden_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(2 * hidden_dim, out_dim)

    def forward(
        self,
        hawkes: torch.Tensor,
        vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        n, length = hawkes.shape
        events = torch.cat([hawkes.unsqueeze(-1), vectors], dim=-1)
        h = self.input_proj(events)
        h = h + self.position[:, :length]
        h = h + self._sinusoidal(length, h.device, h.dtype).unsqueeze(0)

        valid_any = mask.any(dim=1, keepdim=True)
        safe_mask = mask | (~valid_any & (torch.arange(length, device=mask.device)[None, :] == 0))
        h = self.encoder(h, src_key_padding_mask=~safe_mask)
        h = self.norm(h)

        weights = safe_mask.to(h.dtype).unsqueeze(-1)
        mean = (h * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        neg_inf = torch.finfo(h.dtype).min
        maximum = h.masked_fill(~safe_mask.unsqueeze(-1), neg_inf).amax(dim=1)
        pooled = torch.cat([mean, maximum], dim=-1)
        out = self.out(pooled)
        return out * valid_any.to(out.dtype)

    def _sinusoidal(self, length: int, device, dtype) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.hidden_dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.hidden_dim)
        )
        pe = torch.zeros(length, self.hidden_dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div)
        if self.hidden_dim > 1:
            pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        return pe.to(dtype)
