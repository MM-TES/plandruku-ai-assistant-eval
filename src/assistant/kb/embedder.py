"""Fast multilingual embedder for the knowledge base (MiniLM, 384-dim), singleton."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.assistant import config

_DEFAULT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class KBEmbedder:
    def __init__(self, model_name: str | None = None) -> None:
        self._name = model_name or config.kb_embed_model() or _DEFAULT
        self._model: Any = None

    @property
    def name(self) -> str:
        """The resolved model id this embedder uses (e.g. for index provenance)."""
        return self._name

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._name)

    @property
    def dim(self) -> int:
        self._ensure()
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], *, batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        self._ensure()
        vecs = self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=show_progress,
        )
        return np.asarray(vecs, dtype=np.float32)


_SHARED: KBEmbedder | None = None


def get_kb_embedder() -> KBEmbedder:
    global _SHARED
    if _SHARED is None:
        _SHARED = KBEmbedder()
    return _SHARED


__all__ = ["KBEmbedder", "get_kb_embedder"]
