from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np


def rank_indices(scores: Sequence[float]) -> List[int]:
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


def topk_accuracy(rankings: Iterable[Sequence[int]], truths: Iterable[int], k: int) -> float:
    rankings = list(rankings)
    truths = list(truths)
    if not truths:
        return 0.0
    hits = sum(1 for ranking, truth in zip(rankings, truths) if truth in list(ranking)[:k])
    return hits / len(truths)


def average_topk(rankings: Iterable[Sequence[int]], truths: Iterable[int], k: int = 5) -> float:
    return float(np.mean([topk_accuracy(rankings, truths, i) for i in range(1, k + 1)]))


def summarize_rankings(rankings: Sequence[Sequence[int]], truths: Sequence[int], k: int = 5) -> Dict[str, float]:
    return {
        f"A@{i}": topk_accuracy(rankings, truths, i) for i in range(1, k + 1)
    } | {f"Avg@{k}": average_topk(rankings, truths, k)}
