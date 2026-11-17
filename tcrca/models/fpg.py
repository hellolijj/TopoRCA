from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..schemas import FPGResult


@dataclass
class ACPSPath:
    score: float
    nodes: List[int]


def anomalous_nodes(anomaly_scores: torch.Tensor, tau: float) -> List[int]:
    return [int(i) for i in torch.nonzero(anomaly_scores > float(tau), as_tuple=False).flatten().tolist()]


def build_fpg(
    a_causal: torch.Tensor,
    anomaly_scores: torch.Tensor,
    tau: float,
    theta: float,
    root_top_k: int = 5,
) -> FPGResult:
    n = a_causal.shape[0]
    values = anomaly_scores.detach().cpu()
    a = a_causal.detach().cpu()
    vanom = set(anomalous_nodes(values, tau))

    nodes = set(vanom)
    for j in vanom:
        for i in range(n):
            if i != j and float(a[i, j]) > theta:
                nodes.add(i)

    edges: List[Tuple[int, int]] = []
    for j in vanom:
        for i in range(n):
            if i != j and float(a[i, j]) > theta:
                edges.append((i, j))

    adjacency = torch.zeros_like(a)
    for i, j in edges:
        adjacency[i, j] = a[i, j]

    root_scores: Dict[int, float] = {}
    for i in sorted(nodes):
        root_scores[i] = float(sum(float(a[i, j]) * float(values[j]) for j in vanom))
    ranked = sorted(root_scores, key=lambda i: (-root_scores[i], i))
    ranked = ranked[: max(1, int(root_top_k))]

    return FPGResult(
        nodes=sorted(nodes),
        edges=edges,
        adjacency=adjacency,
        root_scores=root_scores,
        ranked_roots=ranked,
        acps_paths=[],
    )


def run_acps(
    a_causal: torch.Tensor,
    anomaly_scores: torch.Tensor,
    start_node: int,
    theta: float,
    tau: float,
    l_max: int = 6,
    top_k: int = 8,
) -> List[ACPSPath]:
    n = a_causal.shape[0]
    a = a_causal.detach().cpu()
    s = anomaly_scores.detach().cpu()

    counter = 0
    queue: List[Tuple[float, int, List[int]]] = [(-0.0, counter, [int(start_node)])]
    results: List[ACPSPath] = []
    seen: set[Tuple[int, ...]] = set()

    while queue and len(results) < top_k:
        neg_score, _, path = heapq.heappop(queue)
        score = -neg_score
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            results.append(ACPSPath(score=score, nodes=path))
            if len(results) >= top_k:
                break
        if len(path) >= l_max:
            continue
        head = path[0]
        for u in range(n):
            weight = float(a[u, head])
            if weight >= theta and float(s[u]) >= tau:
                new_path = [u] + path
                new_score = score + weight * float(s[head])
                counter += 1
                heapq.heappush(queue, (-new_score, counter, new_path))

    return results


def decode_paths(
    a_causal: torch.Tensor,
    anomaly_scores: torch.Tensor,
    tau: float,
    theta: float,
    l_max: int = 6,
    top_k: int = 8,
    start_node: Optional[int] = None,
) -> List[ACPSPath]:
    if start_node is None:
        start_node = int(torch.argmax(anomaly_scores).item())
    return run_acps(a_causal, anomaly_scores, start_node, theta, tau, l_max, top_k)


def refine_fpg(
    fpg: FPGResult,
    paths: Sequence[ACPSPath],
    a_causal: torch.Tensor,
    anomaly_scores: torch.Tensor,
) -> FPGResult:
    a = a_causal.detach().cpu()
    s = anomaly_scores.detach().cpu()
    refined: Dict[Tuple[int, int], Dict[str, float]] = {}
    for path in paths:
        nodes = path.nodes
        if len(nodes) < 2:
            continue
        for u, v in zip(nodes[:-1], nodes[1:]):
            info = refined.setdefault((int(u), int(v)), {"frequency": 0.0, "cumulative_influence": 0.0})
            info["frequency"] += 1.0
            info["cumulative_influence"] += float(a[u, v]) * float(s[v])
    fpg.refined_edges = refined
    fpg.acps_paths = [(p.score, p.nodes) for p in paths]
    return fpg


def anomaly_mask(scores: torch.Tensor, tau: float) -> torch.Tensor:
    return (scores > float(tau)).to(scores.dtype)
