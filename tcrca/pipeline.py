from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import TCRCAConfig
from .models import TCRCAModel
from .schemas import PreparedData, TCRCAOutput, TraceGraph
from .utils import ensure_dir, resolve_device, set_seed

logger = logging.getLogger(__name__)


def build_model(data: PreparedData, config: TCRCAConfig) -> TCRCAModel:
    set_seed(config.seed)
    return TCRCAModel(
        num_services=len(data.service_vocab),
        metric_dim=data.metric_dim,
        span_feature_dim=data.span_feature_dim,
        template_dim=data.log_template_dim,
        config=config,
    ).to(resolve_device(config.device))


def _split(items: Sequence, val_ratio: float, seed: int) -> Tuple[List, List]:
    items = list(items)
    random.Random(seed).shuffle(items)
    n_val = int(round(len(items) * max(0.0, min(1.0, val_ratio))))
    if len(items) > 1:
        n_val = max(1, min(len(items) - 1, n_val))
    return items[n_val:], items[:n_val]


@dataclass
class TrainReport:
    history: List[Dict[str, float]]
    best_epoch: int
    best_val: float
    tau: float
    train_tau: float


class TCRCATrainer:
    def __init__(self, model: TCRCAModel, data: PreparedData, config: TCRCAConfig):
        self.model = model
        self.data = data
        self.config = config
        self.tau: Optional[float] = None
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

    def _graph_terms(self, graph: TraceGraph, train_tau: float) -> torch.Tensor:
        out = self.model.forward_trace(
            graph,
            tau=train_tau,
            compute_causal_loss=True,
            compute_fpg=False,
        )
        return out.extras["loss_gae"] + self.config.lambda_causal * out.extras["loss_causal"]

    def _collect_node_scores(self, graphs: Sequence[TraceGraph]) -> List[float]:
        self.model.eval()
        scores: List[float] = []
        with torch.no_grad():
            for graph in graphs:
                out = self.model.forward_trace(
                    graph, tau=0.0, compute_causal_loss=False, compute_fpg=False
                )
                scores.extend(out.node_anomaly.detach().cpu().tolist())
        return scores

    def _resolve_train_tau(self, graphs: Sequence[TraceGraph]) -> float:
        if self.config.train_tau is not None:
            return float(self.config.train_tau)
        if self.config.tau is not None:
            return float(self.config.tau)
        scores = self._collect_node_scores(graphs)
        if not scores:
            return 0.0
        return float(
            np.quantile(np.asarray(scores, dtype=np.float64), self.config.train_tau_quantile)
        )

    def _validation(
        self, graphs: Sequence[TraceGraph], train_tau: float
    ) -> Tuple[float, List[float]]:
        if not graphs:
            return float("inf"), []
        self.model.eval()
        total = 0.0
        scores: List[float] = []
        with torch.no_grad():
            for graph in graphs:
                out = self.model.forward_trace(
                    graph, tau=train_tau, compute_causal_loss=True, compute_fpg=False
                )
                total += float(
                    (out.extras["loss_gae"] + self.config.lambda_causal * out.extras["loss_causal"]).item()
                )
                scores.extend(out.node_anomaly.detach().cpu().tolist())
        return total / len(graphs), scores

    def train(self) -> TrainReport:
        cfg = self.config
        normal_train, normal_val = _split(self.data.normal_graphs, cfg.val_ratio, cfg.seed)
        if not normal_train:
            raise ValueError("No fault-free traces available for training")

        calibration_graphs = normal_val or normal_train
        train_tau = self._resolve_train_tau(calibration_graphs)
        used_train_tau = train_tau
        logger.info(
            "training V_anom threshold tau_train=%.6f (source=%s)",
            train_tau,
            "explicit" if cfg.train_tau is not None or cfg.tau is not None else f"q{cfg.train_tau_quantile}",
        )

        history: List[Dict[str, float]] = []
        best_val = float("inf")
        best_state = None
        best_epoch = -1
        stale = 0
        rng = random.Random(cfg.seed)

        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            rng.shuffle(normal_train)
            used_train_tau = train_tau
            epoch_loss = 0.0
            batches = 0

            for start in range(0, len(normal_train), cfg.batch_size):
                self.optimizer.zero_grad(set_to_none=True)
                batch_loss = None
                batch = normal_train[start : start + cfg.batch_size]
                for graph in batch:
                    loss = self._graph_terms(graph, train_tau)
                    batch_loss = loss if batch_loss is None else batch_loss + loss
                if batch_loss is None:
                    continue
                batch_loss = batch_loss / max(len(batch), 1)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                epoch_loss += float(batch_loss.item())
                batches += 1

            val_loss, val_scores = self._validation(normal_val, train_tau)
            epoch_loss = epoch_loss / max(batches, 1)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "val_loss": val_loss,
                    "train_tau": used_train_tau,
                }
            )
            logger.info(
                "epoch %d/%d train=%.4f val=%.4f train_tau=%.6f",
                epoch, cfg.epochs, epoch_loss, val_loss, used_train_tau,
            )
            if val_loss < best_val - cfg.min_delta:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= cfg.patience:
                    logger.info("early stopping at epoch %d", epoch)
                    break

            if cfg.train_tau is None and cfg.tau is None and val_scores and epoch < cfg.epochs:
                train_tau = float(
                    np.quantile(
                        np.asarray(val_scores, dtype=np.float64),
                        cfg.train_tau_quantile,
                    )
                )
                logger.info("next-epoch V_anom threshold tau_train=%.6f", train_tau)

        if best_state is not None:
            self.model.load_state_dict(best_state)

        tau = self._calibrate_tau(normal_val or normal_train)
        self.tau = tau
        return TrainReport(
            history=history,
            best_epoch=best_epoch,
            best_val=best_val,
            tau=tau,
            train_tau=used_train_tau,
        )

    def _calibrate_tau(self, graphs: Sequence[TraceGraph]) -> float:
        if self.config.tau is not None:
            return float(self.config.tau)
        scores = self._collect_node_scores(graphs)
        if not scores:
            return 0.0
        return float(np.quantile(np.asarray(scores, dtype=np.float64), self.config.tau_quantile))

    def predict(self, graph: TraceGraph, tau: Optional[float] = None) -> TCRCAOutput:
        resolved = self.config.tau if tau is None else tau
        if resolved is None:
            resolved = self.tau
        self.model.eval()
        with torch.no_grad():
            return self.model.forward_trace(
                graph,
                tau=resolved,
                compute_causal_loss=False,
                compute_fpg=True,
            )


def save_checkpoint(
    path: Path,
    model: TCRCAModel,
    config: TCRCAConfig,
    data: PreparedData,
    tau: float,
    report: Optional[TrainReport] = None,
) -> None:
    ensure_dir(Path(path).parent)
    payload = {
        "model": model.state_dict(),
        "config": {k: v for k, v in asdict(config).items() if k != "project_root"},
        "service_vocab": data.service_vocab,
        "metric_dim": data.metric_dim,
        "span_feature_dim": data.span_feature_dim,
        "log_template_dim": data.log_template_dim,
        "normal_meta": data.normal_meta,
        "tau": tau,
        "train_tau": float(report.train_tau) if report is not None else None,
        "report": asdict(report) if report is not None else None,
    }
    torch.save(payload, path)
    logger.info("checkpoint saved to %s", path)


def load_checkpoint(path: Path, device: str = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    config = TCRCAConfig(**payload["config"])
    config.device = device
    return payload, config
