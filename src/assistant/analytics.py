"""Read-only analytics over assistant feedback (satisfaction, route mix)."""
from __future__ import annotations

from typing import Any

from sqlalchemy import text

from src.utils.logger import setup_logger

_logger = setup_logger(__name__)

_STATS_SQL = text(
    'SELECT "02_route" AS route, "05_vote" AS vote, count(*) AS n '
    "FROM aps.assistant_feedback GROUP BY 1, 2"
)


def feedback_stats(engine: Any = None) -> dict[str, Any]:
    """Aggregate feedback into headline + per-route satisfaction.

    Returns zeros (available=False) when the table is absent — never raises.
    """
    rows: list[dict] = []
    available = True
    try:
        from src.utils.db_connection import get_engine

        eng = engine or get_engine()
        with eng.connect() as conn:
            rows = [dict(r._mapping) for r in conn.execute(_STATS_SQL)]
    except Exception as exc:  # noqa: BLE001 — table may be absent
        _logger.info("assistant analytics unavailable (%s)", exc)
        available = False

    up = sum(int(r["n"]) for r in rows if r["vote"] == "up")
    down = sum(int(r["n"]) for r in rows if r["vote"] == "down")
    total = up + down
    by_route: dict[str, dict[str, int]] = {}
    for r in rows:
        bucket = by_route.setdefault(r["route"] or "?", {"up": 0, "down": 0})
        bucket[r["vote"]] = bucket.get(r["vote"], 0) + int(r["n"])
    return {
        "available": available,
        "up": up,
        "down": down,
        "total": total,
        "satisfaction": round(up / total, 2) if total else None,
        "by_route": by_route,
    }


__all__ = ["feedback_stats"]
