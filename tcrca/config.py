from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class TCRCAConfig:
    data_root: str = "../datasets/train_ticket"
    cache_dir: str = ".cache"
    normal_snapshots: Tuple[str, ...] = ("TT.N00", "TT.N01")
    test_snapshot: str = "TT.F01"
    max_traces_per_fault: int = 1
    metric_steps: int = 128
    feature_window_minutes: float = 30.0
    max_log_events: int = 64
    log_template_dim: int = 64
    strict_log_embedding: bool = True
    span_feature_dim: int = 3
    normalize_metrics: bool = True
    topology_path: Optional[str] = None

    embedding_dim: int = 128
    metric_num_scales: int = 8
    metric_scale_factor: float = 2.0
    transformer_layers: int = 2
    transformer_heads: int = 4
    span_mlp_hidden: int = 128
    dropout: float = 0.1

    gat_heads: int = 4

    gnn_layers: int = 3
    gae_hidden: int = 128
    lambda_s: float = 0.7
    lambda_f: float = 0.3
    anomaly_alpha: float = 0.6

    granger_hidden: int = 128
    gamma_topo: float = 0.1
    lambda_topo: float = 1e-2
    lambda_causal: float = 1.0

    train_tau: Optional[float] = None
    train_tau_quantile: float = 0.9
    tau: Optional[float] = None
    tau_quantile: float = 0.995
    theta: float = 0.15
    acps_l_max: int = 6
    acps_top_k: int = 8
    root_top_k: int = 5

    ablate_topology: bool = False
    ablate_multimodal: bool = False
    ablate_multiscale: bool = False
    ablate_propagation: bool = False

    seed: int = 42
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_ratio: float = 0.2
    patience: int = 8
    min_delta: float = 1e-5
    device: str = "cpu"

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    @property
    def data_path(self) -> Path:
        p = Path(self.data_root)
        return p if p.is_absolute() else (self.project_root / p).resolve()

    @property
    def cache_path(self) -> Path:
        p = Path(self.cache_dir)
        return p if p.is_absolute() else (self.project_root / p).resolve()

    @property
    def effective_num_scales(self) -> int:
        return 1 if self.ablate_multiscale else self.metric_num_scales

    @property
    def metric_scale_count(self) -> int:
        return max(1, self.effective_num_scales)

    @property
    def effective_gamma(self) -> float:
        return 0.0 if self.ablate_topology else self.gamma_topo

    @property
    def effective_lambda_topo(self) -> float:
        return 0.0 if self.ablate_topology else self.lambda_topo

    @property
    def hawkes_dim(self) -> int:
        return self.max_log_events

    @property
    def log_dim(self) -> int:
        return self.hawkes_dim + self.log_template_dim

    @property
    def unified_dim(self) -> int:
        return 3 * self.embedding_dim
