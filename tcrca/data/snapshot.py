from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils import ensure_dir, read_json

logger = logging.getLogger(__name__)

_LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)")


@dataclass
class Snapshot:
    name: str
    traces: pd.DataFrame
    logs: pd.DataFrame
    metrics: pd.DataFrame
    meta: Dict[str, Any]


class SnapshotStore:
    def __init__(self, data_root: Path, cache_dir: Path):
        self.data_root = Path(data_root)
        self.data_dir = self.data_root / "data"
        self.cache_dir = Path(cache_dir)
        ensure_dir(self.cache_dir / "traces")
        ensure_dir(self.cache_dir / "logs")
        ensure_dir(self.cache_dir / "metrics")

    def snapshot_dir(self, name: str) -> Path:
        return self.data_dir / name

    def load(self, name: str, use_cache: bool = True) -> Snapshot:
        snap_dir = self.snapshot_dir(name)
        if not snap_dir.is_dir():
            raise FileNotFoundError(f"Snapshot directory not found: {snap_dir}")

        meta_path = self.data_dir / f"{name}.json"
        meta = read_json(meta_path) if meta_path.exists() else {}

        traces = self._load_traces(name, snap_dir / "spans.json", use_cache)
        logs = self._load_logs(name, snap_dir / "logs.json", use_cache)
        metrics = self._load_metrics(name, snap_dir / "metrics", use_cache)
        logger.info(
            "%s: %d spans / %d trace ids, %d logs, %d metric rows",
            name, len(traces), traces["trace_id"].nunique() if not traces.empty else 0,
            len(logs), len(metrics),
        )
        return Snapshot(name=name, traces=traces, logs=logs, metrics=metrics, meta=meta)

    def _load_traces(self, name: str, spans_path: Path, use_cache: bool) -> pd.DataFrame:
        cache = self.cache_dir / "traces" / f"{name}.parquet"
        if use_cache and cache.exists():
            return pd.read_parquet(cache)
        if not spans_path.exists():
            raise FileNotFoundError(f"spans.json not found: {spans_path}")

        logger.info("Parsing Jaeger spans from %s", spans_path)
        with open(spans_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        rows: List[dict] = []
        for trace in raw or []:
            trace_id = trace.get("traceID")
            processes = trace.get("processes") or {}
            pid_to_service = {
                pid: (proc.get("serviceName") or "") for pid, proc in processes.items()
            }
            spans = trace.get("spans") or []
            span_to_service = {
                sp.get("spanID"): pid_to_service.get(sp.get("processID"), "")
                for sp in spans
            }
            span_to_parent = {}
            for sp in spans:
                refs = sp.get("references") or []
                span_to_parent[sp.get("spanID")] = refs[0].get("spanID") if refs else None

            depth_cache: Dict[str, int] = {}

            def depth_of(span_id: str) -> int:
                if span_id in depth_cache:
                    return depth_cache[span_id]
                parent = span_to_parent.get(span_id)
                if not parent or parent not in span_to_parent:
                    depth_cache[span_id] = 0
                else:
                    depth_cache[span_id] = depth_of(parent) + 1
                return depth_cache[span_id]

            for sp in spans:
                span_id = sp.get("spanID")
                parent_span = span_to_parent.get(span_id)
                start_us = sp.get("startTime")
                dur_us = sp.get("duration") or 0
                if start_us is None:
                    start = pd.NaT
                    end = pd.NaT
                else:
                    start = pd.to_datetime(start_us, unit="us", utc=True)
                    end = start + pd.to_timedelta(dur_us, unit="us")
                tags = {t.get("key"): t.get("value") for t in (sp.get("tags") or [])}
                error = str(tags.get("error", "")).lower() == "true"
                status = tags.get("http.status_code")
                if status is not None:
                    try:
                        error = error or int(status) >= 500
                    except (TypeError, ValueError):
                        pass
                rows.append(
                    {
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "parent_span_id": parent_span,
                        "service_id": span_to_service.get(span_id, ""),
                        "parent_service_id": span_to_service.get(parent_span, "")
                        if parent_span
                        else "",
                        "operation": sp.get("operationName", ""),
                        "start_time": start,
                        "end_time": end,
                        "duration": float(dur_us) / 1e6,
                        "error": bool(error),
                        "depth": depth_of(span_id) if span_id else 0,
                    }
                )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.dropna(subset=["trace_id", "start_time"])
            df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
            df["end_time"] = pd.to_datetime(df["end_time"], utc=True)
            df = df.sort_values(["trace_id", "start_time"]).reset_index(drop=True)
        ensure_dir(cache.parent)
        df.to_parquet(cache, index=False)
        logger.info("Cached traces -> %s (%d rows)", cache, len(df))
        return df

    def _load_logs(self, name: str, logs_path: Path, use_cache: bool) -> pd.DataFrame:
        cache = self.cache_dir / "logs" / f"{name}.parquet"
        if use_cache and cache.exists():
            return pd.read_parquet(cache)
        if not logs_path.exists():
            return pd.DataFrame(columns=["service_id", "timestamp", "message"])

        logger.info("Parsing logs from %s", logs_path)
        with open(logs_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        rows: List[dict] = []
        if isinstance(raw, dict):
            for service, lines in raw.items():
                last: Optional[dict] = None
                for line in lines or []:
                    text = str(line).rstrip("\n")
                    match = _LOG_TS_RE.match(text)
                    if match:
                        stamp = pd.to_datetime(match.group(1), utc=True)
                        last = {
                            "service_id": service,
                            "timestamp": stamp,
                            "message": text[match.end():].lstrip(),
                        }
                        rows.append(last)
                    elif last is not None:
                        last["message"] = f"{last['message']}\n{text}"
        else:
            for entry in raw or []:
                stamp = entry.get("timestamp", "")
                rows.append(
                    {
                        "service_id": entry.get("service_id", ""),
                        "timestamp": pd.to_datetime(stamp, utc=True, errors="coerce"),
                        "message": str(entry.get("message", "")),
                    }
                )

        df = pd.DataFrame(rows)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        ensure_dir(cache.parent)
        df.to_parquet(cache, index=False)
        logger.info("Cached logs -> %s (%d rows)", cache, len(df))
        return df

    def _load_metrics(self, name: str, metrics_dir: Path, use_cache: bool) -> pd.DataFrame:
        cache = self.cache_dir / "metrics" / f"{name}.parquet"
        if use_cache and cache.exists():
            return pd.read_parquet(cache)
        if not metrics_dir.is_dir():
            return pd.DataFrame()

        logger.info("Parsing metrics from %s", metrics_dir)
        frames: List[pd.DataFrame] = []
        for csv_path in sorted(metrics_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            if "timestamp" not in df.columns:
                continue
            raw_ts = df["timestamp"]
            if pd.api.types.is_numeric_dtype(raw_ts):
                df["timestamp"] = pd.to_datetime(raw_ts, unit="s", utc=True, errors="coerce")
            else:
                df["timestamp"] = pd.to_datetime(raw_ts, utc=True, errors="coerce")
            df["service_id"] = csv_path.stem
            for col in df.columns:
                if col not in {"timestamp", "service_id"}:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            frames.append(df)

        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out = out.dropna(subset=["timestamp"]).sort_values(["service_id", "timestamp"]).reset_index(drop=True)
        ensure_dir(cache.parent)
        out.to_parquet(cache, index=False)
        logger.info("Cached metrics -> %s (%d rows)", cache, len(out))
        return out

    @staticmethod
    def metric_columns(metrics: pd.DataFrame) -> List[str]:
        if metrics.empty:
            return []
        return [c for c in metrics.columns if c not in {"timestamp", "service_id"}]
