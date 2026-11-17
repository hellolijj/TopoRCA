from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..utils import ensure_dir, read_json, write_json

logger = logging.getLogger(__name__)

META_FORMAT = "tcrca-fasttext-meta-1"
FASTTEXT_PARAMS = {"window": 5, "min_count": 1, "sg": 1, "epochs": 10}


class DrainFastTextModel:
    def __init__(self, cache_dir: Path, dim: int = 64, strict: bool = True):
        self.cache_dir = Path(cache_dir)
        self.dim = int(dim)
        self.strict = bool(strict)
        self.state_path = self.cache_dir / "drain_state.bin"
        self.fasttext_path = self.cache_dir / "fasttext.model"
        self.meta_path = self.cache_dir / "fasttext.meta.json"
        self._miner = None
        self._fasttext = None
        self._meta: Optional[Dict[str, Any]] = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [token for token in str(text).split() if token]

    def _drain_config(self):
        from drain3.template_miner_config import TemplateMinerConfig

        cfg = TemplateMinerConfig()
        cfg.profiling_enabled = False
        cfg.drain_sim_th = 0.4
        cfg.drain_depth = 5
        cfg.parametrize_numeric_tokens = True
        return cfg

    def fit(self, messages: Sequence[str], seed: int = 42, source: str = "") -> None:
        messages = [str(m) for m in messages if str(m).strip()]
        if not messages:
            raise ValueError("Cannot fit log model on an empty message list")
        ensure_dir(self.cache_dir)
        for path in [self.state_path, self.meta_path, *self.cache_dir.glob("fasttext.model*")]:
            path.unlink(missing_ok=True)

        from drain3 import TemplateMiner
        from drain3.file_persistence import FilePersistence

        persistence = FilePersistence(str(self.state_path))
        miner = TemplateMiner(persistence_handler=persistence, config=self._drain_config())

        templates: List[str] = []
        for message in messages:
            result = miner.add_log_message(message)
            template = result.get("template_mined") if isinstance(result, dict) else None
            if template:
                templates.append(template)
        miner.save_state("fit")
        self._miner = miner

        unique_templates = list(dict.fromkeys(templates)) or messages
        sentences = [self.tokenize(template) for template in unique_templates]
        sentences = [tokens for tokens in sentences if tokens]
        token_count = sum(len(tokens) for tokens in sentences)

        self._fasttext = None
        backend = "hash"
        try:
            from gensim.models import FastText

            self._fasttext = FastText(
                sentences=sentences,
                vector_size=self.dim,
                workers=1,
                seed=seed,
                **FASTTEXT_PARAMS,
            )
            self._fasttext.save(str(self.fasttext_path))
            backend = "fasttext"
        except Exception as exc:
            if self.strict:
                raise RuntimeError(
                    "FastText training failed and the hash fallback is disabled "
                    f"(strict_log_embedding=True): {exc}"
                ) from exc
            logger.warning("FastText unavailable, using deterministic hash vectors: %s", exc)

        self._meta = {
            "format": META_FORMAT,
            "backend": backend,
            "dim": self.dim,
            "seed": int(seed),
            "source": str(source),
            "fit_messages": len(messages),
            "templates": len(unique_templates),
            "sentences": len(sentences),
            "tokens": token_count,
            "tokenizer": "whitespace",
            "aggregation": "mean",
            "fasttext_params": dict(FASTTEXT_PARAMS),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(self.meta_path, self._meta)
        logger.info(
            "Drain/FastText fitted on %d messages, %d templates, %d tokens; backend=%s; meta=%s",
            len(messages),
            len(unique_templates),
            token_count,
            backend,
            self.meta_path,
        )

    def load(self) -> None:
        from drain3 import TemplateMiner
        from drain3.file_persistence import FilePersistence

        if not self.state_path.exists():
            raise FileNotFoundError(f"Drain state not found: {self.state_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Log model metadata not found: {self.meta_path}")
        self._meta = read_json(self.meta_path)
        self._miner = TemplateMiner(
            persistence_handler=FilePersistence(str(self.state_path)), config=self._drain_config()
        )
        self._fasttext = None
        if self._meta.get("backend") == "fasttext":
            try:
                from gensim.models import FastText

                model = FastText.load(str(self.fasttext_path))
                if int(model.wv.vector_size) != self.dim:
                    raise ValueError(
                        f"cached FastText dim {model.wv.vector_size} != expected {self.dim}"
                    )
                self._fasttext = model
            except Exception as exc:
                if self.strict:
                    raise RuntimeError(
                        f"Failed to load cached FastText model at {self.fasttext_path}: {exc}"
                    ) from exc
                logger.warning("Could not load FastText model: %s", exc)

    def ensure_loaded(self) -> None:
        if self._miner is None:
            self.load()

    def _template_of(self, message: str) -> str:
        self.ensure_loaded()
        assert self._miner is not None
        cluster = self._miner.match(message, full_search_strategy="fallback")
        if cluster is not None:
            try:
                return cluster.get_template()
            except AttributeError:
                pass
        return message

    def _hash_vector(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim, dtype=np.float32)

    def _template_vector(self, text: str) -> np.ndarray:
        tokens = self.tokenize(text)
        if self._fasttext is not None and tokens:
            vectors = np.stack(
                [np.asarray(self._fasttext.wv[token], dtype=np.float32) for token in tokens]
            )
            return vectors.mean(axis=0)
        return self._hash_vector(text)

    def transform(self, messages: Sequence[str]) -> np.ndarray:
        if not messages:
            return np.zeros((0, self.dim), dtype=np.float32)
        self.ensure_loaded()
        out = np.zeros((len(messages), self.dim), dtype=np.float32)
        for i, message in enumerate(messages):
            text = self._template_of(str(message))
            out[i] = self._template_vector(text)
        return out

    @property
    def meta(self) -> Optional[Dict[str, Any]]:
        if self._meta is None and self.meta_path.exists():
            self._meta = read_json(self.meta_path)
        return self._meta

    @property
    def is_fitted(self) -> bool:
        if not (self.state_path.exists() and self.meta_path.exists()):
            return False
        try:
            meta = read_json(self.meta_path)
        except Exception:
            return False
        if meta.get("format") != META_FORMAT or int(meta.get("dim", -1)) != self.dim:
            return False
        backend = meta.get("backend")
        if backend == "fasttext":
            return self.fasttext_path.exists()
        return backend == "hash"
