"""Persisted numpy cosine index over the knowledge corpus.

A deliberately small, dependency-light vector store (numpy + json) — the corpus
is a few hundred static chunks, read-only. The embedder is injectable so the
retrieval math is unit-testable without loading the real model.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.assistant import config
from src.assistant.rag.corpus import Chunk, build_chunks
from src.utils.logger import setup_logger

_logger = setup_logger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PERSIST_DIR = ROOT / "models" / "assistant_rag"
_EMB_FILE = "embeddings.npy"
_CHUNKS_FILE = "chunks.json"


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:  # returns (N, d) L2-normalized
        ...


class SentenceTransformerEmbedder:
    """Lazy wrapper around the configured multilingual model (BGE-M3)."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or config.model_for("embed")
        self._model: Any = None

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        self._ensure()
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _tokenize(text: str) -> list[str]:
    """Unicode-word tokenizer for BM25 (works for Ukrainian + codes)."""
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


def _rrf(rankings: list[list[int]], kconst: int = 60) -> list[int]:
    """Reciprocal Rank Fusion of several id-rankings (lesson-09)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (kconst + rank)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


class Reranker:
    """Cross-encoder reranker (lesson-09). Lazy + singleton via get_reranker()."""

    def __init__(self, model_name: str | None = None) -> None:
        self._name = model_name or config.model_for("reranker")
        self._model: Any = None

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._name)

    def rerank(self, query: str, candidates: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        self._ensure()
        scores = self._model.predict([[query, c.text] for c, _ in candidates])
        order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
        return [(candidates[i][0], float(scores[i])) for i in order]

    def order(self, query: str, texts: list[str]) -> list[int]:
        """Return the indices of *texts* sorted by cross-encoder relevance to *query*
        (best first). Index-addressed so an index-keyed candidate pool can be reranked
        without threading chunk objects through (KB reuse, T1.1)."""
        if not texts:
            return []
        self._ensure()
        scores = self._model.predict([[query, t] for t in texts])
        return sorted(range(len(texts)), key=lambda i: float(scores[i]), reverse=True)


def build_index(
    persist_dir: Path | str = DEFAULT_PERSIST_DIR,
    *,
    embedder: Embedder | None = None,
) -> int:
    """Build + persist the index. Returns the number of chunks indexed."""
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks()
    if not chunks:
        _logger.warning("RAG corpus is empty — nothing indexed")
        return 0
    emb = (embedder or SentenceTransformerEmbedder()).encode([c.text for c in chunks])
    emb = _normalize(emb)
    np.save(persist_dir / _EMB_FILE, emb)
    (persist_dir / _CHUNKS_FILE).write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False), encoding="utf-8"
    )
    _logger.info("Indexed %d chunks → %s", len(chunks), persist_dir)
    return len(chunks)


class Retriever:
    """Cosine top-k retriever over the persisted index."""

    def __init__(
        self,
        persist_dir: Path | str = DEFAULT_PERSIST_DIR,
        *,
        embedder: Embedder | None = None,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self._embedder = embedder
        self._emb: np.ndarray | None = None
        self._chunks: list[Chunk] = []
        self._bm25: Any = None
        self._bm25_ready = False

    @property
    def available(self) -> bool:
        return (self.persist_dir / _EMB_FILE).is_file() and (
            self.persist_dir / _CHUNKS_FILE
        ).is_file()

    def _load(self) -> None:
        if self._emb is not None:
            return
        self._emb = np.load(self.persist_dir / _EMB_FILE)
        raw = json.loads((self.persist_dir / _CHUNKS_FILE).read_text(encoding="utf-8"))
        self._chunks = [Chunk(**c) for c in raw]
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder()

    def _ensure_bm25(self) -> None:
        """Lazily build an in-memory BM25 index over the chunks (cheap)."""
        if self._bm25_ready:
            return
        self._bm25_ready = True
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([_tokenize(c.text) for c in self._chunks])
        except Exception as exc:  # noqa: BLE001 — degrade to dense-only
            _logger.info("BM25 unavailable (%s) — dense-only retrieval", exc)
            self._bm25 = None

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[Chunk, float]]:
        """Hybrid retrieval: dense + BM25 -> RRF -> (optional) cross-encoder rerank.

        Degrades gracefully: no rank_bm25 -> dense-only; rerank flag off or model
        missing -> RRF order. Scales recall as the corpus grows (lesson-09).
        """
        if not self.available:
            return []
        self._load()
        assert self._emb is not None and self._embedder is not None
        k = int(top_k if top_k is not None else config.threshold("rag_top_k", 4))
        n_dense = int(config.threshold("rag_dense_candidates", 25))
        n_bm25 = int(config.threshold("rag_bm25_candidates", 25))
        n_cand = int(config.threshold("rag_rerank_candidates", 12))

        qv = _normalize(self._embedder.encode([query]))[0]
        dsims = self._emb @ qv
        dense_ids = [int(i) for i in np.argsort(-dsims)[:n_dense]]
        rankings = [dense_ids]

        self._ensure_bm25()
        if self._bm25 is not None:
            bm_scores = self._bm25.get_scores(_tokenize(query))
            bm_ids = [int(i) for i in np.argsort(-bm_scores)[:n_bm25]]
            rankings.append(bm_ids)

        fused = _rrf(rankings)[: max(n_cand, k)]
        candidates = [(self._chunks[i], float(dsims[i])) for i in fused]

        if config.feature("rerank") and candidates:
            try:
                return get_reranker().rerank(query, candidates)[:k]
            except Exception as exc:  # noqa: BLE001 — reranker is best-effort
                _logger.info("rerank skipped (%s)", exc)
        return candidates[:k]


# --- process-wide singletons (load the 2.27 GB BGE-M3 model ONCE) -----------
# Without this, a fresh Retriever()/embedder per request re-loads the model into
# memory every time (~9-10 s, 82 s cold) — the dominant source of latency.
_SHARED_EMBEDDER: SentenceTransformerEmbedder | None = None
_SHARED_RETRIEVER: Retriever | None = None
_SHARED_RERANKER: Reranker | None = None


def get_embedder() -> SentenceTransformerEmbedder:
    """Return the shared embedder (model loaded lazily, once per process)."""
    global _SHARED_EMBEDDER
    if _SHARED_EMBEDDER is None:
        _SHARED_EMBEDDER = SentenceTransformerEmbedder()
    return _SHARED_EMBEDDER


def get_reranker() -> Reranker:
    """Return the shared cross-encoder reranker (model loaded lazily, once)."""
    global _SHARED_RERANKER
    if _SHARED_RERANKER is None:
        _SHARED_RERANKER = Reranker()
    return _SHARED_RERANKER


def get_retriever(persist_dir: Path | str = DEFAULT_PERSIST_DIR) -> Retriever:
    """Return the shared retriever (index + embedder cached for the process)."""
    global _SHARED_RETRIEVER
    if _SHARED_RETRIEVER is None:
        _SHARED_RETRIEVER = Retriever(persist_dir, embedder=get_embedder())
    return _SHARED_RETRIEVER


def warm() -> None:
    """Preload the embedder + index so the FIRST user query isn't slow.

    Safe to call in a background thread at server startup; never raises.
    """
    try:
        get_retriever().retrieve("розігрів", top_k=1)
        _logger.info("assistant RAG warmed up")
    except Exception as exc:  # noqa: BLE001
        _logger.info("assistant RAG warm-up skipped (%s)", exc)


__all__ = [
    "Embedder", "SentenceTransformerEmbedder", "Retriever", "Reranker", "build_index",
    "DEFAULT_PERSIST_DIR", "get_embedder", "get_retriever", "get_reranker", "warm",
]
