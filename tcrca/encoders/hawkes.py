from __future__ import annotations

import torch
from torch import nn


class HawkesEncoder(nn.Module):
    def __init__(self, num_services: int, max_events: int):
        super().__init__()
        self.num_services = int(num_services)
        self.max_events = int(max_events)
        self.mu_raw = nn.Parameter(torch.zeros(num_services))
        self.alpha_raw = nn.Parameter(torch.zeros(num_services))
        self.beta_raw = nn.Parameter(torch.full((num_services,), -1.0))

    def forward(self, times: torch.Tensor, mask: torch.Tensor, service_idx: torch.Tensor) -> torch.Tensor:
        mu = nn.functional.softplus(self.mu_raw[service_idx]) + 1e-4
        alpha = nn.functional.softplus(self.alpha_raw[service_idx]) + 1e-4
        beta = nn.functional.softplus(self.beta_raw[service_idx]) + 0.5

        safe_times = times.clamp(min=0.0)
        diff = safe_times.unsqueeze(2) - safe_times.unsqueeze(1)
        causal = (diff > 0) & mask.unsqueeze(1)
        excitation = torch.exp(-beta[:, None, None] * diff.clamp(min=0.0)) * causal
        lam = mu[:, None] + alpha[:, None] * excitation.sum(dim=2)
        return lam * mask.to(lam.dtype)
