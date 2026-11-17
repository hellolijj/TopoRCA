from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .hawkes import HawkesEncoder
from .logs import LogEncoder
from .metrics import MetricEncoder
from .spans import SpanEncoder


class MSCGEncoder(nn.Module):
    def __init__(
        self,
        num_services: int,
        metric_dim: int,
        span_feature_dim: int,
        template_dim: int,
        config,
    ):
        super().__init__()
        self.config = config
        self.hawkes = HawkesEncoder(num_services, config.max_log_events)
        self.metric = MetricEncoder(
            metric_dim=metric_dim,
            time_steps=config.metric_steps,
            out_dim=config.embedding_dim,
            num_scales=config.effective_num_scales,
            scale_factor=config.metric_scale_factor,
            layers=config.transformer_layers,
            heads=config.transformer_heads,
            dropout=config.dropout,
        )
        self.log = LogEncoder(
            template_dim=template_dim,
            hidden_dim=config.embedding_dim,
            out_dim=config.embedding_dim,
            max_events=config.max_log_events,
            layers=config.transformer_layers,
            heads=config.transformer_heads,
            dropout=config.dropout,
        )
        self.span = SpanEncoder(
            in_dim=span_feature_dim,
            hidden_dim=config.span_mlp_hidden,
            out_dim=config.embedding_dim,
            dropout=config.dropout,
        )

    def forward(self, graph) -> Dict[str, torch.Tensor]:
        h_metric, scale_weights, scale_reps = self.metric(graph.metrics)
        service_idx = graph.service_indices
        hawkes = self.hawkes(graph.log_event_times, graph.log_mask, service_idx)
        h_log = self.log(hawkes, graph.log_vectors, graph.log_mask)
        h_span = self.span(graph.span_features)

        if self.config.ablate_multimodal:
            h_log = torch.zeros_like(h_log)
            h_span = torch.zeros_like(h_span)

        x_unified = torch.cat([h_log, h_metric, h_span], dim=-1)
        return {
            "x_unified": x_unified,
            "h_metric": h_metric,
            "h_log": h_log,
            "h_span": h_span,
            "hawkes": hawkes,
            "scale_weights": scale_weights,
            "scale_reps": scale_reps,
        }
