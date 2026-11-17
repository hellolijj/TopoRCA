from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcrca.config import TCRCAConfig
from tcrca.data import prepare_trace_dataset
from tcrca.evaluation import evaluate_cases
from tcrca.pipeline import TCRCATrainer, build_model, save_checkpoint
from tcrca.utils import resolve_device, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TCRCA (trace-level)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-snapshot", type=str, default=None)
    parser.add_argument("--max-traces-per-fault", type=int, default=None)
    parser.add_argument("--train-tau", type=float, default=None)
    parser.add_argument("--train-tau-quantile", type=float, default=None)
    parser.add_argument("--gnn-layers", type=int, default=None, help="Eq. (18) GCN depth (RQ4 GNN Layers)")
    parser.add_argument("--windows", type=float, default=None, help="Metric feature window w in minutes")
    parser.add_argument("--scales", type=int, default=None, help="|S|: number of metric scales")
    parser.add_argument("--gamma", type=float, default=None, help="Eq. (26) topology-prior gamma")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--allow-hash-fallback",
        action="store_true",
        help="Allow deterministic hash log embeddings when FastText is unavailable",
    )
    parser.add_argument("--output", type=str, default="artifacts/tcrca.pt")
    args = parser.parse_args()

    setup_logging()
    config = TCRCAConfig(
        seed=args.seed,
        device=resolve_device(args.device),
        strict_log_embedding=not args.allow_hash_fallback,
    )
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.test_snapshot:
        config = replace(config, test_snapshot=args.test_snapshot)
    if args.max_traces_per_fault is not None:
        config = replace(config, max_traces_per_fault=args.max_traces_per_fault)
    if args.train_tau is not None:
        config = replace(config, train_tau=args.train_tau)
    if args.train_tau_quantile is not None:
        config = replace(config, train_tau_quantile=args.train_tau_quantile)
    if args.gnn_layers is not None:
        config = replace(config, gnn_layers=args.gnn_layers)
    if args.windows is not None:
        config = replace(config, feature_window_minutes=args.windows)
    if args.scales is not None:
        config = replace(config, metric_num_scales=args.scales)
    if args.gamma is not None:
        config = replace(config, gamma_topo=args.gamma)

    data = prepare_trace_dataset(config)
    model = build_model(data, config)
    trainer = TCRCATrainer(model, data, config)
    report = trainer.train()

    output = ROOT / args.output
    save_checkpoint(output, model, config, data, report.tau, report)
    evaluation = evaluate_cases(trainer, data, report.tau)
    print(json.dumps({"training": report.__dict__, "evaluation": evaluation["summary"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
