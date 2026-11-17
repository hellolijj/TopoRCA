from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcrca.config import TCRCAConfig
from tcrca.data import prepare_trace_dataset
from tcrca.evaluation import evaluate_cases
from tcrca.pipeline import TCRCATrainer, build_model
from tcrca.utils import ensure_dir, resolve_device, setup_logging

VARIANTS = {
    "full": ({}, "fpg"),
    "wo_tg": ({"ablate_topology": True}, "fpg"),
    "wo_mf": ({"ablate_multimodal": True}, "fpg"),
    "wo_mt": ({"ablate_multiscale": True}, "fpg"),
    "wo_propagation": ({"ablate_propagation": True}, "anomaly"),
}

SWEEPS = {
    "Time Window": [
        ("TW=2", {"feature_window_minutes": 10.0}),
        ("TW=4", {"feature_window_minutes": 20.0}),
        ("TW=6", {"feature_window_minutes": 30.0}),
        ("TW=8", {"feature_window_minutes": 40.0}),
        ("TW=10", {"feature_window_minutes": 60.0}),
    ],
    "Scale |S|": [
        ("S=2", {"metric_num_scales": 2}),
        ("S=4", {"metric_num_scales": 4}),
        ("S=8", {"metric_num_scales": 8}),
        ("S=16", {"metric_num_scales": 16}),
    ],
    "GNN Layers": [
        ("L=2", {"gnn_layers": 2}),
        ("L=3", {"gnn_layers": 3}),
        ("L=4", {"gnn_layers": 4}),
    ],
    "Gamma": [
        ("gamma=0.05", {"gamma_topo": 0.05}),
        ("gamma=0.1", {"gamma_topo": 0.1}),
        ("gamma=0.2", {"gamma_topo": 0.2}),
    ],
}


def data_key(config: TCRCAConfig) -> Tuple[object, ...]:
    return (
        config.data_root,
        config.cache_dir,
        tuple(config.normal_snapshots),
        config.test_snapshot,
        config.max_traces_per_fault,
        config.metric_steps,
        config.feature_window_minutes,
        config.max_log_events,
        config.log_template_dim,
    )


def dataset_for(config: TCRCAConfig, cache: Dict[Tuple[object, ...], object]):
    key = data_key(config)
    if key not in cache:
        cache[key] = prepare_trace_dataset(config)
    return cache[key]


def run_entries(
    base: TCRCAConfig,
    seeds: List[int],
    entries: List[Tuple[str, Dict[str, object], str]],
    data_cache: Dict[Tuple[object, ...], object],
) -> Dict[str, List[Dict[str, float]]]:
    results: Dict[str, List[Dict[str, float]]] = {}
    for label, flags, mode in entries:
        runs = []
        for seed in seeds:
            config = replace(base, seed=seed, **flags)
            data = dataset_for(config, data_cache)
            model = build_model(data, config)
            trainer = TCRCATrainer(model, data, config)
            report = trainer.train()
            evaluation = evaluate_cases(trainer, data, report.tau, mode=mode)
            runs.append(evaluation["summary"])
            print(label, seed, json.dumps(evaluation["summary"]))
        results[label] = runs
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TCRCA ablations and RQ4 sweep")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--variants", type=str, default=",".join(VARIANTS))
    parser.add_argument("--sweeps", type=str, default=",".join(SWEEPS))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--allow-hash-fallback",
        action="store_true",
        help="Allow deterministic hash log embeddings when FastText is unavailable",
    )
    parser.add_argument("--output", type=str, default="results/experiments.json")
    args = parser.parse_args()

    setup_logging()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    sweeps = [s.strip() for s in args.sweeps.split(",") if s.strip()]
    base = TCRCAConfig(
        device=resolve_device(args.device),
        strict_log_embedding=not args.allow_hash_fallback,
    )
    if args.epochs is not None:
        base = replace(base, epochs=args.epochs)

    data_cache: Dict[Tuple[object, ...], object] = {}
    ablations = run_entries(
        base,
        seeds,
        [(name, VARIANTS[name][0], VARIANTS[name][1]) for name in variants],
        data_cache,
    )
    sweep_results = {
        group: run_entries(
            base,
            seeds,
            [(label, flags, "fpg") for label, flags in SWEEPS[group]],
            data_cache,
        )
        for group in sweeps
    }

    output = ROOT / args.output
    ensure_dir(output.parent)
    payload = {"ablations": ablations, "sweeps": sweep_results}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if ablations:
        print_table(ablations, "Ablation")
    for group, group_results in sweep_results.items():
        print_table(group_results, group)


def print_table(results: Dict[str, List[Dict[str, float]]], title: str = "Variant") -> None:
    keys = ["A@1", "A@2", "A@3", "A@4", "A@5", "Avg@5", "avg_latency_ms"]
    print(f"\n| {title} | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for name, runs in results.items():
        cells = []
        for key in keys:
            values = np.asarray([run.get(key, 0.0) for run in runs], dtype=float)
            cells.append(f"{values.mean():.3f}+/-{values.std():.3f}")
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
