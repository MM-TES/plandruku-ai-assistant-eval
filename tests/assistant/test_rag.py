"""Phase 2 gate: RAG corpus chunking + numpy retriever (offline)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from src.assistant.rag import corpus
from src.assistant.rag.corpus import Chunk
from src.assistant.rag.index import Retriever, build_index

_VOCAB = ["дефіцит", "розклад", "виробництво", "матеріал", "kpi"]


@pytest.fixture(autouse=True)
def _no_reranker(monkeypatch) -> None:
    # offline: never load the real cross-encoder reranker
    monkeypatch.setattr("src.assistant.config.feature", lambda name: False)


class FakeEmbedder:
    """Deterministic bag-of-words embedder over a tiny vocab (no model needed)."""

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            low = t.lower()
            rows.append([float(low.count(w)) for w in _VOCAB])
        return np.asarray(rows, dtype=np.float32)


def test_corpus_builds_from_operator_help() -> None:
    chunks = corpus.build_chunks()
    assert chunks, "corpus must not be empty (docs/operator_help/*.md present)"
    assert all(c.text and c.source for c in chunks)
    sources = {c.source for c in chunks}
    # sources are friendly H1 titles, NOT filenames or developer docs
    assert "Що зробити — пропозиції системи" in sources
    assert not any(s.endswith(".md") for s in sources)
    # developer-doc jargon files must NOT be indexed
    assert "schema_card.md" not in sources and "gotchas.md" not in sources


def test_retriever_offline_with_fake_embedder(tmp_path, monkeypatch) -> None:
    fixture = [
        Chunk(id="d#0", source="doc", url=None, text="Дефіцит — це нестача матеріалу під замовлення."),
        Chunk(id="d#1", source="doc", url=None, text="Розклад друку розподіляє замовлення по машинах."),
        Chunk(id="d#2", source="doc", url=None, text="Виробництво приймає забезпечені замовлення."),
    ]
    monkeypatch.setattr("src.assistant.rag.index.build_chunks", lambda: fixture)

    n = build_index(tmp_path, embedder=FakeEmbedder())
    assert n == 3

    retr = Retriever(tmp_path, embedder=FakeEmbedder())
    assert retr.available
    hits = retr.retrieve("що таке дефіцит матеріалу", top_k=2)
    assert hits
    top_chunk, score = hits[0]
    assert top_chunk.id == "d#0"
    assert score > 0


def test_retriever_empty_when_no_index(tmp_path) -> None:
    assert Retriever(tmp_path, embedder=FakeEmbedder()).retrieve("x") == []


def test_rrf_fuses_rankings() -> None:
    from src.assistant.rag.index import _rrf

    # id 2 appears high in both rankings -> should win the fusion
    fused = _rrf([[1, 2, 3], [2, 4, 1]])
    assert fused[0] == 2
    assert set(fused) == {1, 2, 3, 4}


def test_tokenize_handles_ukrainian() -> None:
    from src.assistant.rag.index import _tokenize

    assert _tokenize("Дефіцит матеріалу 12!") == ["дефіцит", "матеріалу", "12"]


def test_reranker_reorders_when_enabled(tmp_path, monkeypatch) -> None:
    fixture = [
        Chunk(id="d#0", source="doc", url=None, text="Дефіцит — це нестача матеріалу."),
        Chunk(id="d#1", source="doc", url=None, text="Розклад друку по машинах."),
        Chunk(id="d#2", source="doc", url=None, text="Виробництво замовлень."),
    ]
    monkeypatch.setattr("src.assistant.rag.index.build_chunks", lambda: fixture)
    build_index(tmp_path, embedder=FakeEmbedder())

    monkeypatch.setattr("src.assistant.config.feature", lambda name: name == "rerank")

    class _FakeReranker:
        def rerank(self, query, candidates):  # sort by id desc to prove rerank ran
            return sorted(candidates, key=lambda cs: cs[0].id, reverse=True)

    monkeypatch.setattr("src.assistant.rag.index.get_reranker", lambda: _FakeReranker())
    hits = Retriever(tmp_path, embedder=FakeEmbedder()).retrieve("дефіцит матеріал", top_k=3)
    assert [c.id for c, _ in hits] == ["d#2", "d#1", "d#0"]  # reranker order, not dense order


def test_get_retriever_is_a_singleton() -> None:
    # the model loads lazily on retrieve(), so building the singleton is cheap;
    # the same instance must be reused so BGE-M3 isn't reloaded per request.
    from src.assistant.rag import index

    index._SHARED_RETRIEVER = None
    try:
        assert index.get_retriever() is index.get_retriever()
        assert index.get_embedder() is index.get_embedder()
    finally:
        index._SHARED_RETRIEVER = None
        index._SHARED_EMBEDDER = None


@pytest.mark.skipif(
    os.getenv("ASSISTANT_RAG_TEST") != "1",
    reason="real BGE-M3 embedder test is opt-in (set ASSISTANT_RAG_TEST=1; downloads model)",
)
def test_real_embedder_roundtrip(tmp_path) -> None:
    from src.assistant.rag.index import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder()
    build_index(tmp_path, embedder=emb)
    hits = Retriever(tmp_path, embedder=emb).retrieve("як передати замовлення у виробництво", top_k=3)
    assert hits
