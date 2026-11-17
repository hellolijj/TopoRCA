from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

from ..config import TCRCAConfig
from ..schemas import FaultCase, PreparedData, TraceGraph
from ..utils import read_json
from .logs import DrainFastTextModel
from .snapshot import Snapshot, SnapshotStore

logger = logging.getLogger(__name__)

SPAN_FEATURE_NAMES = [
    "log_mean_duration",
    "mean_nesting_depth",
    "error_rate",
]


class MetricNormalizer:
    def __init__(self, columns: Sequence[str], mean: np.ndarray, std: np.ndarray):
        self.columns = list(columns)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, frames: Iterable[pd.DataFrame], columns: Sequence[str]) -> "MetricNormalizer":
        cols = list(columns)
        chunks = []
        for frame in frames:
            if frame.empty:
                continue
            available = [c for c in cols if c in frame.columns]
            if not available:
                continue
            chunks.append(frame[available].to_numpy(dtype=np.float64))
        if not chunks:
            return cls(cols, np.zeros(len(cols), dtype=np.float32), np.ones(len(cols), dtype=np.float32))
        arr = np.concatenate(chunks, axis=0)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        mean = np.nan_to_num(mean, nan=0.0)
        std = np.nan_to_num(std, nan=1.0)
        std[std < 1e-8] = 1.0
        return cls(cols, mean.astype(np.float32), std.astype(np.float32))

    def transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.zeros((0, len(self.columns)), dtype=np.float32)
        arr = np.zeros((len(frame), len(self.columns)), dtype=np.float32)
        for j, col in enumerate(self.columns):
            if col in frame.columns:
                arr[:, j] = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float32)
        return (arr - self.mean[None, :]) / self.std[None, :]

    def transform_array(self, arr: np.ndarray) -> np.ndarray:
        return (arr.astype(np.float32) - self.mean[None, :]) / self.std[None, :]

    def state_dict(self) -> dict:
        return {"columns": self.columns, "mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: dict) -> "MetricNormalizer":
        return cls(state["columns"], state["mean"], state["std"])


def _timestamps(df: pd.DataFrame, col: str = "timestamp") -> pd.Series:
    if df.empty or col not in df.columns:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(df[col], utc=True, errors="coerce")


def build_normal_topology(
    snapshots: Sequence[Snapshot],
    topology_path: Optional[Path] = None,
) -> Tuple[Set[Tuple[str, str]], List[str]]:
    services: Set[str] = set()
    edges: Set[Tuple[str, str]] = set()
    for snap in snapshots:
        t = snap.traces
        if t.empty:
            continue
        services.update(str(s) for s in t["service_id"].dropna().unique())
        services.update(str(s) for s in t["parent_service_id"].dropna().unique() if s)
        for parent, child in zip(t["parent_service_id"], t["service_id"]):
            if parent and child and str(parent) != str(child):
                edges.add((str(parent), str(child)))

    if topology_path is not None and Path(topology_path).exists():
        raw = read_json(Path(topology_path))
        for item in raw.get("services", []):
            src = item.get("serviceId") or item.get("service_id")
            if not src:
                continue
            services.add(str(src))
            for dep in item.get("dependencies", []) or []:
                dst = dep.get("serviceId") if isinstance(dep, dict) else dep
                if dst and str(dst) != str(src):
                    edges.add((str(src), str(dst)))
                    services.add(str(dst))

    return edges, sorted(services)


def _window_bounds(trace_start: float, trace_end: float, minutes: float) -> Tuple[float, float]:
    window_end = float(trace_end)
    window_start = window_end - float(minutes) * 60.0
    window_start = min(window_start, float(trace_start))
    return window_start, window_end


def _fill_metric_tensor(
    frame: pd.DataFrame,
    start: float,
    end: float,
    steps: int,
    columns: Sequence[str],
    normalizer: Optional[MetricNormalizer],
    normalize: bool,
) -> np.ndarray:
    out = np.zeros((steps, len(columns)), dtype=np.float32)
    if frame.empty:
        return out
    ts = _timestamps(frame)
    mask = (ts >= pd.Timestamp(start, unit="s", tz="UTC")) & (
        ts <= pd.Timestamp(end, unit="s", tz="UTC")
    )
    part = frame.loc[mask].copy()
    if part.empty:
        return out
    part = part.sort_values("timestamp")
    raw = np.zeros((len(part), len(columns)), dtype=np.float32)
    for j, col in enumerate(columns):
        if col in part.columns:
            raw[:, j] = pd.to_numeric(part[col], errors="coerce").to_numpy(dtype=np.float32)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    span = max(float(end) - float(start), 1e-6)
    rel = (ts.loc[mask].astype("int64").to_numpy() / 1e9 - float(start)) / span
    bin_idx = np.clip((rel * steps).astype(np.int64), 0, steps - 1)
    sums = np.zeros((steps, len(columns)), dtype=np.float64)
    counts = np.zeros((steps,), dtype=np.float64)
    np.add.at(sums, bin_idx, raw)
    np.add.at(counts, bin_idx, 1.0)
    valid = counts > 0
    out[valid] = (sums[valid] / counts[valid, None]).astype(np.float32)
    if normalize and normalizer is not None:
        out = normalizer.transform_array(out)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _fill_log_tensors(
    frame: pd.DataFrame,
    start: float,
    end: float,
    max_events: int,
    log_model: Optional[DrainFastTextModel],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.full((max_events,), -1.0, dtype=np.float32)
    mask = np.zeros((max_events,), dtype=bool)
    dim = log_model.dim if log_model is not None else 0
    vectors = np.zeros((max_events, dim), dtype=np.float32)
    if frame.empty or max_events <= 0:
        return times, vectors, mask
    ts = _timestamps(frame)
    begin = pd.Timestamp(start, unit="s", tz="UTC")
    finish = pd.Timestamp(end, unit="s", tz="UTC")
    part = frame.loc[(ts >= begin) & (ts <= finish)].copy()
    if part.empty:
        return times, vectors, mask
    part = part.sort_values("timestamp").tail(max_events)
    t = _timestamps(part)
    span = max(end - start, 1e-6)
    rel = ((t - begin).dt.total_seconds() / span).to_numpy(dtype=np.float32)
    rel = np.clip(rel, 0.0, 1.0)
    n = len(part)
    times[:n] = rel
    mask[:n] = True
    if log_model is not None:
        vectors[:n] = log_model.transform(part["message"].astype(str).tolist())
    return times, vectors, mask


def _span_feature_block(tdf: pd.DataFrame, service: str) -> np.ndarray:
    sub = tdf.loc[tdf["service_id"] == service]
    if sub.empty:
        return np.zeros(len(SPAN_FEATURE_NAMES), dtype=np.float32)
    durations = pd.to_numeric(sub["duration"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    errors = sub["error"].astype(float).to_numpy(dtype=np.float64) if "error" in sub else np.zeros(len(sub))
    depths = pd.to_numeric(sub.get("depth", 0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return np.asarray(
        [
            np.log1p(durations.mean()) if durations.size else 0.0,
            depths.mean() if depths.size else 0.0,
            errors.mean() if errors.size else 0.0,
        ],
        dtype=np.float32,
    )


def build_trace_graphs(
    snapshot: Snapshot,
    topology_edges: Set[Tuple[str, str]],
    service_vocab: Dict[str, int],
    metric_columns: Sequence[str],
    metric_normalizer: Optional[MetricNormalizer],
    log_model: Optional[DrainFastTextModel],
    config: TCRCAConfig,
) -> List[TraceGraph]:
    tracetable = snapshot.traces
    if tracetable.empty:
        return []

    metric_frames = {
        str(sid): frame.sort_values("timestamp")
        for sid, frame in snapshot.metrics.groupby("service_id", sort=False)
    } if not snapshot.metrics.empty else {}
    log_frames = {
        str(sid): frame.sort_values("timestamp")
        for sid, frame in snapshot.logs.groupby("service_id", sort=False)
    } if not snapshot.logs.empty else {}

    graphs: List[TraceGraph] = []
    grouped = tracetable.groupby("trace_id", sort=False)
    for trace_id, tdf in grouped:
        tdf = tdf.sort_values("start_time").reset_index(drop=True)
        node_list: List[str] = []
        seen: Set[str] = set()
        for sid in tdf["service_id"].tolist():
            sid = str(sid)
            if sid and sid not in seen:
                seen.add(sid)
                node_list.append(sid)
        if not node_list:
            continue
        for sid in node_list:
            service_vocab.setdefault(sid, len(service_vocab))
        idx = {sid: i for i, sid in enumerate(node_list)}
        n = len(node_list)

        trace_start = float(pd.Timestamp(tdf["start_time"].min()).timestamp())
        trace_end = float(pd.Timestamp(tdf["end_time"].max()).timestamp())
        window_start, window_end = _window_bounds(
            trace_start, trace_end, config.feature_window_minutes
        )

        adjacency = torch.zeros((n, n), dtype=torch.float32)
        for child, parent in zip(tdf["service_id"], tdf["parent_service_id"]):
            child, parent = str(child), str(parent)
            if parent and child != parent and parent in idx and child in idx:
                adjacency[idx[parent], idx[child]] = 1.0

        a_topo = torch.zeros((n, n), dtype=torch.float32)
        for i, si in enumerate(node_list):
            for j, sj in enumerate(node_list):
                if i != j and (si, sj) in topology_edges:
                    a_topo[i, j] = 1.0

        metrics = np.zeros((n, config.metric_steps, len(metric_columns)), dtype=np.float32)
        for sid, i in idx.items():
            frame = metric_frames.get(sid, pd.DataFrame())
            metrics[i] = _fill_metric_tensor(
                frame, window_start, window_end, config.metric_steps,
                metric_columns, metric_normalizer, config.normalize_metrics,
            )

        log_times = np.full((n, config.max_log_events), -1.0, dtype=np.float32)
        log_vectors = np.zeros(
            (n, config.max_log_events, config.log_template_dim if log_model is None else log_model.dim),
            dtype=np.float32,
        )
        log_mask = np.zeros((n, config.max_log_events), dtype=bool)
        for sid, i in idx.items():
            frame = log_frames.get(sid, pd.DataFrame())
            times, vectors, mask = _fill_log_tensors(
                frame, window_start, window_end, config.max_log_events, log_model
            )
            log_times[i] = times
            log_vectors[i] = vectors
            log_mask[i] = mask

        span_features = np.stack([_span_feature_block(tdf, sid) for sid in node_list])
        if span_features.shape[1] != config.span_feature_dim:
            fixed = np.zeros((n, config.span_feature_dim), dtype=np.float32)
            m = min(fixed.shape[1], span_features.shape[1])
            fixed[:, :m] = span_features[:, :m]
            span_features = fixed

        graphs.append(
            TraceGraph(
                trace_id=str(trace_id),
                service_ids=node_list,
                service_indices=torch.tensor([service_vocab[s] for s in node_list], dtype=torch.long),
                adjacency=adjacency,
                a_topo=a_topo,
                metrics=torch.from_numpy(metrics),
                log_event_times=torch.from_numpy(log_times),
                log_vectors=torch.from_numpy(log_vectors),
                log_mask=torch.from_numpy(log_mask),
                span_features=torch.from_numpy(span_features),
                trace_start=trace_start,
                trace_end=trace_end,
            )
        )
        if len(graphs) % 200 == 0:
            logger.info("%s: built %d trace graphs", snapshot.name, len(graphs))
    logger.info("%s: %d trace graphs", snapshot.name, len(graphs))
    return graphs


def build_fault_cases(
    snapshot: Snapshot,
    graphs: Sequence[TraceGraph],
    max_traces_per_fault: int = 1,
) -> Tuple[List[FaultCase], Dict[str, object]]:
    faults = snapshot.meta.get("faults", []) or []
    cases: List[FaultCase] = []
    uncovered: List[Dict[str, object]] = []
    total_overlapping = 0

    for i, fault in enumerate(faults):
        service = str(fault.get("service", ""))
        start = float(fault.get("start", 0.0))
        end = start + float(fault.get("duration", 0.0))
        overlapping = [g for g in graphs if g.trace_end >= start and g.trace_start <= end]
        total_overlapping += len(overlapping)
        if not overlapping:
            uncovered.append(
                {
                    "fault_index": i,
                    "service": service,
                    "fault": str(fault.get("fault", "")),
                    "start": start,
                    "end": end,
                    "reason": "no_trace_overlaps_fault_interval",
                }
            )
            continue

        def overlap(g: TraceGraph) -> float:
            return max(0.0, min(g.trace_end, end) - max(g.trace_start, start))

        overlapping.sort(key=lambda g: (-overlap(g), g.trace_start, g.trace_id))
        limit = len(overlapping) if max_traces_per_fault <= 0 else int(max_traces_per_fault)
        for g in overlapping[:limit]:
            cases.append(
                FaultCase(
                    case_id=f"{snapshot.name}:{i}:{g.trace_id}",
                    snapshot=snapshot.name,
                    trace_id=g.trace_id,
                    root_service=service,
                    fault_type=str(fault.get("fault", "")),
                    fault_start=start,
                    fault_end=end,
                )
            )

    covered = len(faults) - len(uncovered)
    stats: Dict[str, object] = {
        "selection": "traces overlapping the fault interval; root service is ground truth only",
        "ranking": "largest temporal overlap, then earliest start",
        "max_traces_per_fault": int(max_traces_per_fault),
        "num_faults": len(faults),
        "num_covered_faults": covered,
        "num_uncovered_faults": len(uncovered),
        "fault_coverage": float(covered / len(faults)) if faults else 0.0,
        "num_overlapping_traces": total_overlapping,
        "num_cases": len(cases),
        "uncovered": uncovered,
    }
    logger.info(
        "%s: %d fault cases from %d faults (%d uncovered, %d overlapping traces)",
        snapshot.name, len(cases), len(faults), len(uncovered), total_overlapping,
    )
    return cases, stats


def prepare_trace_dataset(
    config: TCRCAConfig,
    test_snapshot: Optional[str] = None,
    normal_snapshots: Optional[Sequence[str]] = None,
) -> PreparedData:
    normal_names = list(normal_snapshots or config.normal_snapshots)
    test_name = test_snapshot or config.test_snapshot
    store = SnapshotStore(config.data_path, config.cache_path)

    normal_snaps = [store.load(name) for name in normal_names]
    test_snap = store.load(test_name)

    normal_traces = pd.concat([s.traces for s in normal_snaps], ignore_index=True)
    if normal_traces.empty:
        raise ValueError("No normal traces available for training")

    service_vocab: Dict[str, int] = {}
    for sid in sorted(set(normal_traces["service_id"].dropna().astype(str))):
        service_vocab.setdefault(sid, len(service_vocab))
    topology_path = Path(config.topology_path) if config.topology_path else None
    topology_edges, _ = build_normal_topology(normal_snaps, topology_path=topology_path)

    metric_cols: List[str] = []
    for snap in normal_snaps:
        for col in SnapshotStore.metric_columns(snap.metrics):
            if col not in metric_cols:
                metric_cols.append(col)
    metric_cols = sorted(metric_cols)
    normalizer = MetricNormalizer.fit([s.metrics for s in normal_snaps], metric_cols)

    log_model = DrainFastTextModel(
        config.cache_path / "log_model",
        dim=config.log_template_dim,
        strict=config.strict_log_embedding,
    )
    if log_model.is_fitted:
        log_model.load()
    else:
        messages: List[str] = []
        for snap in normal_snaps:
            if not snap.logs.empty:
                messages.extend(snap.logs["message"].astype(str).tolist())
        log_model.fit(
            messages,
            seed=config.seed,
            source=f"{config.data_path.name}:{','.join(normal_names)}",
        )

    normal_graphs: List[TraceGraph] = []
    for snap in normal_snaps:
        normal_graphs.extend(
            build_trace_graphs(
                snap, topology_edges, service_vocab, metric_cols, normalizer, log_model, config
            )
        )

    test_graphs = build_trace_graphs(
        test_snap, topology_edges, service_vocab, metric_cols, normalizer, log_model, config
    )
    cases, case_stats = build_fault_cases(
        test_snap, test_graphs, config.max_traces_per_fault
    )

    k = len(service_vocab)
    topology = torch.zeros((k, k), dtype=torch.float32)
    for src, dst in topology_edges:
        if src in service_vocab and dst in service_vocab:
            topology[service_vocab[src], service_vocab[dst]] = 1.0

    logger.info(
        "Prepared %d normal traces, %d test traces, %d cases, %d services, %d metric dims",
        len(normal_graphs), len(test_graphs), len(cases), k, len(metric_cols),
    )
    return PreparedData(
        normal_graphs=normal_graphs,
        test_graphs=test_graphs,
        cases=cases,
        topology=topology,
        service_vocab=service_vocab,
        metric_dim=len(metric_cols),
        span_feature_dim=config.span_feature_dim,
        log_template_dim=config.log_template_dim,
        normal_meta={
            "normal_snapshots": normal_names,
            "metric_columns": metric_cols,
            "metric_normalizer": normalizer.state_dict(),
            "num_topology_edges": len(topology_edges),
        },
        case_stats=case_stats,
        test_snapshot=test_name,
    )
