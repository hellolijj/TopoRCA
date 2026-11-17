from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn


@dataclass
class MSTGIOutput:
    a_causal: torch.Tensor
    root_scores: torch.Tensor
    scale_weights: torch.Tensor
    edge_logits: List[torch.Tensor]
    attention: List[torch.Tensor]
    loss: torch.Tensor
    loss_task: torch.Tensor
    loss_topo: torch.Tensor


class MSTGI(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_scales: int,
        mlp_hidden: int = 128,
        gamma_topo: float = 0.1,
        lambda_topo: float = 1e-2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gamma_topo = float(gamma_topo)
        self.lambda_topo = float(lambda_topo)
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )
        self.fusion_logits = nn.Parameter(torch.zeros(num_scales))

    def forward(
        self,
        scale_features: List[torch.Tensor],
        a_topo: torch.Tensor,
        anomaly_scores: torch.Tensor,
        anomaly_mask: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
    ) -> MSTGIOutput:
        if not scale_features:
            raise ValueError("MSTGI needs at least one scale representation")
        n = scale_features[0].shape[0]
        device, dtype = scale_features[0].device, scale_features[0].dtype
        if len(scale_features) != self.fusion_logits.numel():
            raise ValueError("MSTGI scale count does not match fusion weights")

        a_topo = a_topo.to(device=device, dtype=dtype)
        edge_logits: List[torch.Tensor] = []
        attention: List[torch.Tensor] = []
        for h in scale_features:
            hi = h.unsqueeze(1).expand(n, n, h.shape[-1])
            hj = h.unsqueeze(0).expand(n, n, h.shape[-1])
            e = self.pair_mlp(torch.cat([hi, hj], dim=-1)).squeeze(-1)
            biased = (1.0 + self.gamma_topo * a_topo) * e
            alpha = torch.softmax(biased, dim=1)
            edge_logits.append(e)
            attention.append(alpha)

        weights = torch.softmax(self.fusion_logits, dim=0)
        stacked = torch.stack(attention, dim=0)
        a_causal = (stacked * weights.view(-1, 1, 1)).sum(dim=0)

        root_scores = compute_root_scores(a_causal, anomaly_scores, anomaly_mask)

        if compute_loss:
            root_log = torch.log_softmax(root_scores, dim=0)
            root_prob = root_log.exp()
            anomaly_log = torch.log_softmax(anomaly_scores.detach(), dim=0)
            loss_task = (root_prob * (root_log - anomaly_log)).sum()
            loss_topo = (a_causal - a_topo).pow(2).sum()
            loss = loss_task + self.lambda_topo * loss_topo
        else:
            zero = torch.zeros((), device=device, dtype=dtype)
            loss_task = loss_topo = loss = zero

        return MSTGIOutput(
            a_causal=a_causal,
            root_scores=root_scores,
            scale_weights=weights,
            edge_logits=edge_logits,
            attention=attention,
            loss=loss,
            loss_task=loss_task,
            loss_topo=loss_topo,
        )


def compute_root_scores(
    a_causal: torch.Tensor,
    anomaly_scores: torch.Tensor,
    anomaly_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if anomaly_mask is None:
        return a_causal @ anomaly_scores
    mask = anomaly_mask.to(dtype=a_causal.dtype)
    return a_causal @ (anomaly_scores * mask)
