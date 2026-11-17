from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if out_dim % heads != 0:
            raise ValueError("out_dim must be divisible by heads")
        self.heads = heads
        self.per_head = out_dim // heads
        self.out_dim = heads * self.per_head
        self.linear = nn.Linear(in_dim, self.out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, self.per_head))
        self.att_dst = nn.Parameter(torch.empty(heads, self.per_head))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        wh = self.linear(x).view(n, self.heads, self.per_head)
        src = (wh * self.att_src).sum(dim=-1).transpose(0, 1)
        dst = (wh * self.att_dst).sum(dim=-1).transpose(0, 1)
        logits = F.leaky_relu(src.unsqueeze(2) + dst.unsqueeze(1), negative_slope=0.2)
        logits = logits.masked_fill(~adjacency.unsqueeze(0), torch.finfo(logits.dtype).min)
        alpha = torch.softmax(logits, dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.einsum("hij,jhd->hid", alpha, wh)
        return out.permute(1, 0, 2).reshape(n, self.out_dim)


class GATEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int = 3, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        current = in_dim
        for _ in range(layers):
            self.layers.append(GATLayer(current, hidden_dim, heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))
            current = hidden_dim
        self.dropout = dropout
        self.pool_score = nn.Linear(hidden_dim, 1)

    def _mask(self, adjacency: torch.Tensor) -> torch.Tensor:
        n = adjacency.shape[0]
        eye = torch.eye(n, device=adjacency.device, dtype=torch.bool)
        mask = (adjacency > 0) | (adjacency.transpose(0, 1) > 0) | eye
        return mask

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor):
        mask = self._mask(adjacency)
        h = x
        for layer, norm in zip(self.layers, self.norms):
            out = layer(h, mask)
            if out.shape == h.shape:
                h = norm(h + F.dropout(out, p=self.dropout, training=self.training))
            else:
                h = norm(F.elu(out))
        scores = self.pool_score(torch.tanh(h)).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        h_g = (weights.unsqueeze(-1) * h).sum(dim=0)
        return h, h_g, weights
