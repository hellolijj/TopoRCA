from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcrca.data import prepare_trace_dataset
from tcrca.evaluation import evaluate_cases
from tcrca.pipeline import TCRCATrainer, build_model, load_checkpoint
from tcrca.utils import resolve_device, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TCRCA")
    parser.add_argument("--checkpoint", type=str, default="artifacts/tcrca.pt")
    parser.add_argument("--mode", choices=["fpg", "anomaly"], default="fpg")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--allow-hash-fallback",
        action="store_true",
        help="Allow deterministic hash log embeddings when FastText is unavailable",
    )
    args = parser.parse_args()

    setup_logging()
    payload, config = load_checkpoint(ROOT / args.checkpoint, resolve_device(args.device))
    if args.allow_hash_fallback:
        config = replace(config, strict_log_embedding=False)
    data = prepare_trace_dataset(config)
    model = build_model(data, config)
    model.load_state_dict(payload["model"])
    trainer = TCRCATrainer(model, data, config)
    result = evaluate_cases(trainer, data, float(payload["tau"]), mode=args.mode)
    print(json.dumps(result["summary"], indent=2))
    for row in result["details"]:
        print(
            f"{row['case_id']} root={row['root']} top5={row['top5']} "
            f"hit@1={row['hit@1']} hit@3={row['hit@3']}"
        )


if __name__ == "__main__":
    main()
