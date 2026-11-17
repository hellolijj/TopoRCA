from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
from torch import nn

from ..config import TCRCAConfig
from ..schemas import FPGResult, TCRCAOutput
from ..encoders import GATEncoder, MSCGEncoder
from ..utils import move_graph_to_device
from .fpg import anomaly_mask, build_fpg, decode_paths, refine_fpg
from .gae import GraphAutoEncoder
from .mstgi import MSTGI

logger = logging.getLogger(__name__)


class TCRCAModel(nn.Module):
    def __init__(
        self,
        num_services: int,
        metric_dim: int,
        span_feature_dim: int,
        template_dim: int,
        config: TCRCAConfig,
    ):
        super().__init__()
        self.config = config
        self.mscg = MSCGEncoder(num_services, metric_dim, span_feature_dim, template_dim, config)
        self.gat = GATEncoder(
            in_dim=config.unified_dim,
            hidden_dim=config.embedding_dim,
            layers=3,
            heads=config.gat_heads,
            dropout=config.dropout,
        )
        self.gae = GraphAutoEncoder(
            x_dim=config.unified_dim,
            hidden_dim=config.gae_hidden,
            layers=config.gnn_layers,
            lambda_s=config.lambda_s,
            lambda_f=config.lambda_f,
            anomaly_alpha=config.anomaly_alpha,
        )
        self.mstgi = MSTGI(
            hidden_dim=config.embedding_dim,
            num_scales=config.metric_scale_count,
            mlp_hidden=config.granger_hidden,
            gamma_topo=config.effective_gamma,
            lambda_topo=config.effective_lambda_topo,
            dropout=config.dropout,
        )

    def forward_trace(
        self,
        graph,
        tau: Optional[float] = None,
        compute_causal_loss: bool = True,
        theta: Optional[float] = None,
        compute_fpg: bool = True,
    ) -> TCRCAOutput:
        device = next(self.parameters()).device
        graph = move_graph_to_device(graph, device)

        enc = self.mscg(graph)
        _, h_g, pool_weights = self.gat(enc["x_unified"], graph.adjacency)
        gae_out = self.gae(enc["x_unified"], graph.adjacency)
        anomaly = gae_out["node_anomaly"]

        threshold = self.config.tau if tau is None else tau
        threshold = None if threshold is None else float(threshold)
        if threshold is None:
            v_anom_mask = torch.zeros_like(anomaly)
        else:
            v_anom_mask = anomaly_mask(anomaly, threshold)
        causal = self.mstgi(
            enc["scale_reps"],
            graph.a_topo,
            anomaly,
            anomaly_mask=v_anom_mask,
            compute_loss=compute_causal_loss,
        )

        theta = float(self.config.theta if theta is None else theta)
        if compute_fpg and threshold is not None:
            fpg = build_fpg(
                causal.a_causal,
                anomaly,
                tau=threshold,
                theta=theta,
                root_top_k=self.config.root_top_k,
            )
            start = int(torch.argmax(anomaly).item()) if anomaly.numel() else 0
            if fpg.nodes:
                paths = decode_paths(
                    causal.a_causal,
                    anomaly,
                    tau=threshold,
                    theta=theta,
                    l_max=self.config.acps_l_max,
                    top_k=self.config.acps_top_k,
                    start_node=start,
                )
                refine_fpg(fpg, paths, causal.a_causal, anomaly)
        else:
            n = anomaly.numel()
            fpg = FPGResult(
                nodes=[],
                edges=[],
                adjacency=torch.zeros((n, n), device=anomaly.device),
                root_scores={},
                ranked_roots=[],
                acps_paths=[],
            )

        return TCRCAOutput(
            trace_id=graph.trace_id,
            node_anomaly=anomaly,
            trace_anomaly=float(gae_out["trace_anomaly"].item()),
            a_causal=causal.a_causal,
            fpg=fpg,
            h_g=h_g,
            x_unified=enc["x_unified"],
            extras={
                "loss_gae": gae_out["loss"],
                "loss_structure": gae_out["loss_structure"],
                "loss_feature": gae_out["loss_feature"],
                "loss_causal": causal.loss,
                "loss_task": causal.loss_task,
                "loss_topo": causal.loss_topo,
                "anomaly_mask": v_anom_mask,
                "anomaly_threshold": threshold,
                "scale_weights": causal.scale_weights,
                "metric_scale_weights": enc["scale_weights"],
                "hawkes": enc["hawkes"],
                "pool_weights": pool_weights,
                "z": gae_out["z"],
                "adjacency_hat": gae_out["adjacency_hat"],
                "attributes_hat": gae_out["attributes_hat"],
            },
        )
