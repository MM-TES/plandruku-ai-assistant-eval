"""Record assistant feedback (👍/👎) to aps.assistant_feedback + LangSmith.

The DB write uses the MAIN engine (the assistant's only write; keeps the
read-only boundary intact). Everything is best-effort — feedback must never
break the UI, and it degrades gracefully if the table is absent.
"""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text

from src.assistant import config
from src.utils.logger import setup_logger

_logger = setup_logger(__name__)

_INSERT_SQL = text(
    'INSERT INTO aps.assistant_feedback '
    '("01_session_id","02_route","03_question","04_answer","05_vote","06_comment",'
    '"07_citations","08_langsmith_run_id") '
    "VALUES (:session_id,:route,:question,:answer,:vote,:comment,"
    "CAST(:citations AS JSONB),:run_id)"
)


def _build_row(session: Any, idx: int, vote: str, comment: str, run_id: str | None) -> dict[str, Any]:
    msgs = getattr(session, "assistant_messages", []) or []
    answer = route = question = ""
    citations: list = []
    if 0 <= idx < len(msgs):
        m = msgs[idx]
        answer = m.get("text", "")
        route = m.get("route", "")
        citations = m.get("citations", []) or []
    if 0 <= idx - 1 < len(msgs):
        question = msgs[idx - 1].get("text", "")
    return {
        "session_id": str(getattr(session, "session_id", ""))[:64],
        "route": route,
        "question": question,
        "answer": answer,
        "vote": vote,
        "comment": comment or None,
        "citations": json.dumps(citations, ensure_ascii=False),
        "run_id": run_id,
    }


def _insert_row(row: dict[str, Any], engine: Any = None) -> bool:
    try:
        from src.utils.db_connection import get_engine

        eng = engine or get_engine()
        with eng.begin() as conn:
            conn.execute(_INSERT_SQL, row)
        return True
    except Exception as exc:  # noqa: BLE001 — table may be absent; degrade gracefully
        _logger.info("assistant feedback not persisted (%s)", exc)
        return False


def _send_langsmith_feedback(run_id: str | None, vote: str, comment: str) -> None:
    if not run_id or not config.langsmith_tracing_enabled():
        return
    try:
        from langsmith import Client

        Client().create_feedback(
            run_id,
            key="user_vote",
            score=1.0 if vote == "up" else 0.0,
            comment=comment or None,
        )
    except Exception as exc:  # noqa: BLE001
        _logger.info("langsmith feedback skipped (%s)", exc)


def record_feedback(
    session: Any,
    *,
    idx: int,
    vote: str,
    comment: str = "",
    engine: Any = None,
    run_id: str | None = None,
) -> bool:
    """Persist a feedback vote. Returns True iff the DB row was written."""
    if vote not in {"up", "down"}:
        return False
    row = _build_row(session, idx, vote, comment, run_id)
    persisted = _insert_row(row, engine)
    _send_langsmith_feedback(run_id, vote, comment)
    return persisted


__all__ = ["record_feedback"]
