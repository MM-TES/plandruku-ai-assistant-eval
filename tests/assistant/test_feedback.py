"""Phase 6 gate: feedback recording (row build + vote validation, mocked DB)."""
from __future__ import annotations

from src.assistant import feedback


class _Session:
    session_id = "sid-1"
    assistant_messages = [
        {"role": "user", "text": "як це працює?"},
        {"role": "assistant", "text": "Ось як це працює.", "route": "instructions", "citations": []},
    ]


def test_bad_vote_rejected_without_db(monkeypatch) -> None:
    called = {"n": 0}
    monkeypatch.setattr(feedback, "_insert_row", lambda *a, **k: called.__setitem__("n", called["n"] + 1) or True)
    monkeypatch.setattr(feedback, "_send_langsmith_feedback", lambda *a, **k: None)
    assert feedback.record_feedback(_Session(), idx=1, vote="sideways") is False
    assert called["n"] == 0


def test_builds_row_and_persists(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(feedback, "_insert_row", lambda row, engine=None: captured.update(row) or True)
    monkeypatch.setattr(feedback, "_send_langsmith_feedback", lambda *a, **k: None)
    ok = feedback.record_feedback(_Session(), idx=1, vote="up", comment="дякую")
    assert ok is True
    assert captured["question"] == "як це працює?"
    assert captured["answer"] == "Ось як це працює."
    assert captured["route"] == "instructions"
    assert captured["vote"] == "up"
    assert captured["comment"] == "дякую"
    assert captured["session_id"] == "sid-1"


def test_persist_false_propagates(monkeypatch) -> None:
    monkeypatch.setattr(feedback, "_insert_row", lambda *a, **k: False)
    monkeypatch.setattr(feedback, "_send_langsmith_feedback", lambda *a, **k: None)
    assert feedback.record_feedback(_Session(), idx=1, vote="down") is False
