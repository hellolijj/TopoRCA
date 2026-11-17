from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..config import TCRCAConfig
from ..pipeline import TCRCATrainer
from ..schemas import FaultCase, PreparedData, TCRCAOutput, TraceGraph
from .metrics import average_topk, rank_indices, summarize_rankings, topk_accuracy

logger = logging.getLogger(__name__)


def ranking_from_output(
    out: TCRCAOutput,
    graph: TraceGraph,
    config: TCRCAConfig,
    mode: str = "fpg",
) -> List[int]:
    n = graph.num_nodes
    if mode == "anomaly" or config.ablate_propagation:
        return rank_indices(out.node_anomaly.detach().cpu().tolist())

    return [i for i in out.fpg.ranked_roots if 0 <= i < n]


def evaluate_cases(
    trainer: TCRCATrainer,
    data: PreparedData,
    tau: float,
    cases: Optional[Sequence[FaultCase]] = None,
    mode: str = "fpg",
    k: int = 5,
) -> Dict[str, object]:
    cases = list(cases or data.cases)
    by_trace = {g.trace_id: g for g in data.test_graphs}
    rankings: List[List[int]] = []
    truths: List[int] = []
    details: List[Dict[str, object]] = []
    latencies: List[float] = []

    for case in cases:
        graph = by_trace.get(case.trace_id)
        if graph is None:
            continue
        started = time.perf_counter()
        out = trainer.predict(graph, tau=tau)
        latencies.append((time.perf_counter() - started) * 1000.0)
        ranking = ranking_from_output(out, graph, trainer.config, mode=mode)
        ranked_services = [graph.service_ids[i] for i in ranking]
        truth = graph.service_ids.index(case.root_service) if case.root_service in graph.service_ids else -1
        rankings.append(ranking)
        truths.append(truth)
        details.append(
            {
                "case_id": case.case_id,
                "trace_id": case.trace_id,
                "root": case.root_service,
                "root_present": truth >= 0,
                "fault": case.fault_type,
                "fault_start": case.fault_start,
                "fault_end": case.fault_end,
                "trace_start": graph.trace_start,
                "trace_end": graph.trace_end,
                "top5": ranked_services[:5],
                "hit@1": bool(truth >= 0 and ranking and ranking[0] == truth),
                "hit@3": bool(truth >= 0 and truth in ranking[:3]),
                "trace_anomaly": out.trace_anomaly,
                "fpg_nodes": [graph.service_ids[i] for i in out.fpg.nodes],
                "fpg_edges": [(graph.service_ids[i], graph.service_ids[j]) for i, j in out.fpg.edges],
                "paths": [
                    (score, [graph.service_ids[i] for i in nodes])
                    for score, nodes in out.fpg.acps_paths
                ],
            }
        )

    summary = summarize_rankings(rankings, truths, k=k)
    summary["avg_latency_ms"] = float(np.mean(latencies)) if latencies else 0.0
    summary["num_cases"] = len(details)
    summary["num_root_missing"] = sum(1 for truth in truths if truth < 0)
    stats = data.case_stats or {}
    summary["num_faults"] = int(stats.get("num_faults", 0))
    summary["num_covered_faults"] = int(stats.get("num_covered_faults", 0))
    summary["num_uncovered_faults"] = int(stats.get("num_uncovered_faults", 0))
    summary["fault_coverage"] = float(stats.get("fault_coverage", 0.0))
    summary["uncovered_faults"] = list(stats.get("uncovered", []))
    return {"summary": summary, "details": details, "case_stats": stats}
