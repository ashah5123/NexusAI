import os
import time
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
from fastembed import TextEmbedding

from .models import ReindexResult

if TYPE_CHECKING:
    from .repository import DocumentRepository

MODEL_NAME = os.getenv("NEXUSAI_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
MODEL_CACHE = Path(os.getenv("NEXUSAI_MODEL_CACHE", "./data/models"))


class EmbeddingService:
    """Lazily loads a compact ONNX embedding model only when semantic search is enabled."""

    def __init__(self) -> None:
        self._model: TextEmbedding | None = None
        self._lock = Lock()

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
                    self._model = TextEmbedding(
                        model_name=MODEL_NAME,
                        cache_dir=str(MODEL_CACHE),
                        threads=max(1, min(4, os.cpu_count() or 1)),
                    )
        return self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return MODEL_NAME

    def embed_query(self, query: str) -> np.ndarray:
        vector = next(iter(self._get_model().query_embed(query)))
        return np.asarray(vector, dtype=np.float32)

    def index_pending(self, repository: "DocumentRepository") -> ReindexResult:
        started = time.perf_counter()
        chunks = repository.chunks_pending_embedding(MODEL_NAME)
        if chunks:
            texts = [row["content"] for row in chunks]
            vectors = self._get_model().passage_embed(texts, batch_size=32)
            repository.save_embeddings(
                [(row["id"], np.asarray(vector, dtype=np.float32)) for row, vector in zip(chunks, vectors)],
                MODEL_NAME,
            )
        status = repository.embedding_status(MODEL_NAME, self.loaded)
        return ReindexResult(
            **status.model_dump(),
            indexed_now=len(chunks),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )
