"""KB tests: loaders, corpus chunking/dedup, FAISS index/retrieve, escalation gate."""
from __future__ import annotations

import numpy as np

from src.assistant.kb import corpus, loaders
from src.assistant.kb.corpus import KBChunk
from src.assistant.kb.index import KBRetriever, build_index

_VOCAB = ["лезо", "рулон", "друк", "оснащення", "безпека"]


class FakeKBEmbedder:
    @property
    def dim(self) -> int:
        return len(_VOCAB)

    def encode(self, texts, **kw) -> np.ndarray:
        rows = [[float(t.lower().count(w)) for w in _VOCAB] for t in texts]
        a = np.asarray(rows, dtype=np.float32)
        n = np.linalg.norm(a, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return a / n


def test_loader_text_and_html(tmp_path) -> None:
    t = tmp_path / "a.txt"
    t.write_text("Привіт світ", encoding="utf-8")
    assert "Привіт" in loaders.extract(t)[0].text

    h = tmp_path / "a.html"
    h.write_text("<html><body><p>Текст тут</p><script>js</script></body></html>", encoding="utf-8")
    out = loaders.extract(h)[0].text
    assert "Текст тут" in out and "js" not in out  # script stripped


def test_supported_gating() -> None:
    assert loaders.supported(".docx", include_doc=False, ocr=False)
    assert not loaders.supported(".doc", include_doc=False, ocr=False)
    assert loaders.supported(".doc", include_doc=True, ocr=False)
    assert not loaders.supported(".jpg", include_doc=False, ocr=False)
    assert loaders.supported(".jpg", include_doc=False, ocr=True)


def test_corpus_builds_and_dedups(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("Установка леза на держатель. " * 40, encoding="utf-8")
    (tmp_path / "sub" / "b.md").write_text("Безпека при роботі з рулоном. " * 40, encoding="utf-8")
    (tmp_path / "dup.txt").write_text("Установка леза на держатель. " * 40, encoding="utf-8")

    chunks, stats = corpus.build_chunks(tmp_path)
    assert chunks and stats["files_ok"] >= 2
    texts = [c.text for c in chunks]
    assert len(texts) == len(set(texts))  # exact-dedup → all unique
    assert all(c.source for c in chunks)


def test_faiss_index_and_retrieve(tmp_path) -> None:
    chunks = [
        KBChunk("0", "doc1.txt", "p", "стор. 1", "Як встановити лезо на держатель оснащення"),
        KBChunk("1", "doc2.txt", "p", "", "Друк рулону на машині"),
        KBChunk("2", "doc3.txt", "p", "", "Безпека персоналу на ділянці"),
    ]
    assert build_index(chunks, tmp_path, embedder=FakeKBEmbedder(), show_progress=False) == 3
    retr = KBRetriever(tmp_path, embedder=FakeKBEmbedder())
    assert retr.available
    hits = retr.retrieve("як встановити лезо", top_k=2)
    assert hits and "лезо" in hits[0][0].text and hits[0][1] > 0


def test_search_kb_empty_without_index(tmp_path, monkeypatch) -> None:
    from src.assistant.kb import search
    from src.assistant.kb.index import KBRetriever as _KR

    monkeypatch.setattr("src.assistant.kb.index.get_kb_retriever", lambda: _KR(tmp_path))
    res = search.search_kb("будь-що")
    assert res.knowledge == "" and res.best_score == 0.0


def test_escalation_gate(monkeypatch) -> None:
    """Cosine pre-filter: below kb_min_score → no KB (relevance gate isolated)."""
    from src.assistant import orchestrator
    from src.assistant.kb.search import KBResult

    monkeypatch.setattr("src.assistant.config.kb_min_score", lambda: 0.3)
    monkeypatch.setattr("src.assistant.orchestrator._kb_relevant",
                        lambda q, res, usage=None: True)
    monkeypatch.setattr("src.assistant.kb.search.search_kb",
                        lambda q, usage=None, top_k=None, extra_variants=None: KBResult("знання", [], 0.6))
    assert orchestrator._search_external_kb("q") is not None
    monkeypatch.setattr("src.assistant.kb.search.search_kb",
                        lambda q, usage=None, top_k=None, extra_variants=None: KBResult("знання", [], 0.1))
    assert orchestrator._search_external_kb("q") is None


def test_relevance_gate_blocks_offtopic(monkeypatch) -> None:
    """Above the cosine pre-filter but LLM judges off-topic → no KB answer."""
    from src.assistant import orchestrator
    from src.assistant.kb.search import KBResult

    monkeypatch.setattr("src.assistant.config.kb_min_score", lambda: 0.3)
    monkeypatch.setattr("src.assistant.kb.search.search_kb",
                        lambda q, usage=None, top_k=None, extra_variants=None: KBResult("борщ з салом", [], 0.6))
    monkeypatch.setattr("src.assistant.orchestrator._kb_relevant",
                        lambda q, res, usage=None: False)
    assert orchestrator._search_external_kb("рецепт борщу") is None
    monkeypatch.setattr("src.assistant.orchestrator._kb_relevant",
                        lambda q, res, usage=None: True)
    assert orchestrator._search_external_kb("рецепт борщу") is not None
