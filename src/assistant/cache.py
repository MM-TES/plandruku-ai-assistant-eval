"""Semantic response cache (lesson-10 pattern, in-memory).

Caches stable answers (how-to / out-of-scope) keyed by query+page embedding
similarity. A near-duplicate question (cosine >= threshold, not expired) returns
the cached answer instantly — skipping router + RAG + rerank + the answer LLM.

Deliberately NOT used for data/analysis/history routes (live data changes daily).
In-memory (per process); lost on restart — fine for a single-operator app.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.assistant import config


def _embed(text: str, embedder: Any) -> np.ndarray:
    emb = embedder.encode([text])
    vec = np.asarray(emb, dtype=np.float32)[0]
    norm = float(np.linalg.norm(vec)) or 1.0
    return vec / norm


class SemanticCache:
    """Tiny cosine cache over query+page-key embeddings with TTL + size cap."""

    def __init__(self, embedder: Any = None) -> None:
        self._embedder = embedder
        self._vecs: list[np.ndarray] = []
        self._payloads: list[dict] = []
        self._expire: list[float] = []

    def _emb(self) -> Any:
        if self._embedder is None:
            from src.assistant.rag.index import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    @staticmethod
    def _key(query: str, page_key: str) -> str:
        return f"{page_key} :: {query}"

    def lookup(self, query: str, page_key: str) -> dict | None:
        if not self._vecs:
            return None
        now = time.monotonic()
        qv = _embed(self._key(query, page_key), self._emb())
        best_i, best_sim = -1, -1.0
        for i, vec in enumerate(self._vecs):
            if self._expire[i] < now:
                continue
            sim = float(qv @ vec)
            if sim > best_sim:
                best_i, best_sim = i, sim
        threshold = float(config.threshold("cache_similarity", 0.92))
        if best_i >= 0 and best_sim >= threshold:
            return dict(self._payloads[best_i])
        return None

    def store(self, query: str, page_key: str, payload: dict, *, ttl: float | None = None) -> None:
        ttl = float(ttl if ttl is not None else config.threshold("cache_ttl_seconds", 3600))
        qv = _embed(self._key(query, page_key), self._emb())
        self._vecs.append(qv)
        self._payloads.append(dict(payload))
        self._expire.append(time.monotonic() + ttl)
        cap = int(config.threshold("cache_max_entries", 500))
        while len(self._vecs) > cap:
            self._vecs.pop(0)
            self._payloads.pop(0)
            self._expire.pop(0)

    def clear(self) -> None:
        self._vecs.clear()
        self._payloads.clear()
        self._expire.clear()


_SHARED_CACHE: SemanticCache | None = None


def get_cache() -> SemanticCache:
    global _SHARED_CACHE
    if _SHARED_CACHE is None:
        _SHARED_CACHE = SemanticCache()
    return _SHARED_CACHE


__all__ = ["SemanticCache", "get_cache"]
