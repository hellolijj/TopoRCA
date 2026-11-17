from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class TraceGraph:
    trace_id: str
    service_ids: List[str]
    service_indices: torch.LongTensor
    adjacency: torch.Tensor
    a_topo: torch.Tensor
    metrics: torch.Tensor
    log_event_times: torch.Tensor
    log_vectors: torch.Tensor
    log_mask: torch.BoolTensor
    span_features: torch.Tensor
    trace_start: float
    trace_end: float

    @property
    def num_nodes(self) -> int:
        return len(self.service_ids)


@dataclass
class FaultCase:
    case_id: str
    snapshot: str
    trace_id: str
    root_service: str
    fault_type: str
    fault_start: float
    fault_end: float
    is_faulty: bool = True


@dataclass
class FPGResult:
    nodes: List[int]
    edges: List[Tuple[int, int]]
    adjacency: torch.Tensor
    root_scores: Dict[int, float]
    ranked_roots: List[int]
    acps_paths: List[Tuple[float, List[int]]]
    refined_edges: Dict[Tuple[int, int], Dict[str, float]] = field(default_factory=dict)


@dataclass
class TCRCAOutput:
    trace_id: str
    node_anomaly: torch.Tensor
    trace_anomaly: float
    a_causal: torch.Tensor
    fpg: FPGResult
    h_g: torch.Tensor
    x_unified: torch.Tensor
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedData:
    normal_graphs: List[TraceGraph]
    test_graphs: List[TraceGraph]
    cases: List[FaultCase]
    topology: torch.Tensor
    service_vocab: Dict[str, int]
    metric_dim: int
    span_feature_dim: int
    log_template_dim: int
    normal_meta: Dict[str, Any] = field(default_factory=dict)
    case_stats: Dict[str, Any] = field(default_factory=dict)
    test_snapshot: str = ""
