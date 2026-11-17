from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


def normalized_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    n = adjacency.shape[0]
    eye = torch.eye(n, device=adjacency.device, dtype=adjacency.dtype)
    sym = ((adjacency > 0) | (adjacency.transpose(0, 1) > 0)).to(adjacency.dtype)
    a = sym + eye
    degree = a.sum(dim=1).clamp(min=1.0)
    inv_sqrt = degree.pow(-0.5)
    return inv_sqrt[:, None] * a * inv_sqrt[None, :]


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation

    def forward(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        out = a_norm @ self.linear(h)
        return F.relu(out) if self.activation else out


@dataclass
class GAELoss:
    total: torch.Tensor
    structure: torch.Tensor
    feature: torch.Tensor


class GraphAutoEncoder(nn.Module):
    def __init__(
        self,
        x_dim: int,
        hidden_dim: int = 128,
        layers: int = 2,
        lambda_s: float = 0.7,
        lambda_f: float = 0.3,
        anomaly_alpha: float = 0.6,
    ):
        super().__init__()
        self.lambda_s = float(lambda_s)
        self.lambda_f = float(lambda_f)
        self.alpha = float(anomaly_alpha)
        dims = [x_dim] + [hidden_dim] * int(layers)
        self.layers = nn.ModuleList(
            [
                GCNLayer(dims[i], dims[i + 1], activation=i < len(dims) - 2)
                for i in range(len(dims) - 1)
            ]
        )
        self.structure_weight = nn.Parameter(torch.eye(hidden_dim))
        self.attribute_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, x_unified: torch.Tensor, adjacency: torch.Tensor) -> Dict[str, torch.Tensor]:
        a_norm = normalized_adjacency(adjacency)
        z = x_unified
        for layer in self.layers:
            z = layer(z, a_norm)

        structure_logits = z @ self.structure_weight @ z.transpose(0, 1)
        adjacency_hat = torch.sigmoid(structure_logits)
        attributes_hat = self.attribute_decoder(z)

        target = (adjacency > 0).to(structure_logits.dtype)
        loss_structure = F.binary_cross_entropy_with_logits(structure_logits, target, reduction="sum")
        loss_feature = (x_unified - attributes_hat).pow(2).sum()
        total = self.lambda_s * loss_structure + self.lambda_f * loss_feature

        structure_error = (target - adjacency_hat).pow(2).sum(dim=1)
        feature_error = (x_unified - attributes_hat).pow(2).sum(dim=1)
        node_anomaly = self.alpha * structure_error + (1.0 - self.alpha) * feature_error
        trace_anomaly = node_anomaly.max()
        return {
            "z": z,
            "adjacency_hat": adjacency_hat,
            "attributes_hat": attributes_hat,
            "node_anomaly": node_anomaly,
            "trace_anomaly": trace_anomaly,
            "loss_structure": loss_structure,
            "loss_feature": loss_feature,
            "loss": total,
        }
