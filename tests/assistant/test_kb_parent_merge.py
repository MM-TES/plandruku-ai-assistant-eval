"""INC-2 (P1.3) tests: whole-document assembly in search._assemble (offline)."""
from __future__ import annotations

import types

from src.assistant.kb import search as S


def _ch(cid, source, locator, text):
    return types.SimpleNamespace(id=str(cid), source=source, locator=locator, text=text)


def test_doc_loc_range_span_and_single():
    cs = [_ch(1, "a.pdf", "стор. 1", "x"), _ch(2, "a.pdf", "стор. 2", "y")]
    assert S._doc_loc_range(cs) == "стор. 1–2"
    assert S._doc_loc_range([_ch(1, "a.pdf", "стор. 3", "z")]) == "стор. 3"
    assert S._doc_loc_range([_ch(1, "a.pdf", "", "z")]) == ""


def test_assemble_groups_by_document(monkeypatch):
    monkeypatch.setattr(S.config, "kb_param", lambda name, default=None:
                        {"enabled": True, "max_parent_chars": 6000} if name == "parent_merge" else default)
    # Interleaved hits from two documents; A's pages out of order.
    hits = [
        (_ch(6, "A.pdf", "стор. 2", "A2"), 0.0),
        (_ch(10, "B.pdf", "стор. 1", "B1"), 0.8),
        (_ch(5, "A.pdf", "стор. 1", "A1"), 0.9),
    ]
    knowledge, citations = S._assemble(hits)
    # one block + one citation per document (not per chunk)
    assert len(citations) == 2
    assert knowledge.count("[A.pdf") == 1 and knowledge.count("[B.pdf") == 1
    # A's chunks ordered by id within a single coherent block, with page range
    a_block = next(b for b in knowledge.split("\n\n") if b.startswith("[A.pdf"))
    assert "A1\nA2" in a_block and "стор. 1–2" in a_block


def test_assemble_off_is_per_chunk(monkeypatch):
    monkeypatch.setattr(S.config, "kb_param", lambda name, default=None:
                        {"enabled": False} if name == "parent_merge" else default)
    hits = [(_ch(5, "A.pdf", "стор. 1", "A1"), 0.9), (_ch(6, "A.pdf", "стор. 2", "A2"), 0.5)]
    _, citations = S._assemble(hits)
    assert len(citations) == 2  # per-chunk listing, unchanged legacy behaviour


def test_assemble_truncates_to_budget(monkeypatch):
    monkeypatch.setattr(S.config, "kb_param", lambda name, default=None:
                        {"enabled": True, "max_parent_chars": 10} if name == "parent_merge" else default)
    hits = [(_ch(1, "A.pdf", "стор. 1", "0123456789ABCDEF"), 0.9)]
    knowledge, _ = S._assemble(hits)
    # block body is capped at max_parent_chars (header excluded)
    body = knowledge.split("\n", 1)[1]
    assert len(body) <= 10
