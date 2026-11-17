from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch


def resolve_device(device: Optional[str] = None) -> str:
    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(device)


def move_graph_to_device(graph: Any, device: Any) -> Any:
    target = torch.device(device)
    updates: Dict[str, Any] = {}
    for item in fields(graph):
        value = getattr(graph, item.name)
        if torch.is_tensor(value) and value.device != target:
            updates[item.name] = value.to(target)
    return replace(graph, **updates) if updates else graph


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        force=True,
    )


def topk_indices(scores: torch.Tensor, k: int) -> List[int]:
    if scores.numel() == 0 or k <= 0:
        return []
    k = min(k, scores.numel())
    values, indices = torch.topk(scores, k=k, largest=True, sorted=True)
    return [int(i) for i in indices.tolist()]


def normalize_rows(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(eps)


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def mean_std(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0)}


def maybe_mkdir_for_file(path: Path) -> None:
    if path.parent and not path.parent.exists():
        os.makedirs(path.parent, exist_ok=True)
