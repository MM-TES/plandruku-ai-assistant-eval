"""Phase 6 gate: feedback analytics aggregation + graceful degrade."""
from __future__ import annotations

from src.assistant.analytics import feedback_stats


class _FakeRow:
    def __init__(self, d: dict) -> None:
        self._mapping = d


class _FakeConn:
    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *a) -> bool:  # noqa: ANN002
        return False

    def execute(self, *_a, **_k):  # noqa: ANN002
        return [
            _FakeRow({"route": "instructions", "vote": "up", "n": 3}),
            _FakeRow({"route": "instructions", "vote": "down", "n": 1}),
            _FakeRow({"route": "data_query", "vote": "up", "n": 2}),
        ]


class _FakeEngine:
    def connect(self):  # noqa: ANN201
        return _FakeConn()


class _BrokenEngine:
    def connect(self):  # noqa: ANN201
        raise RuntimeError("no table")


def test_stats_aggregate() -> None:
    s = feedback_stats(engine=_FakeEngine())
    assert s["available"] is True
    assert s["up"] == 5 and s["down"] == 1 and s["total"] == 6
    assert s["satisfaction"] == round(5 / 6, 2)
    assert s["by_route"]["instructions"] == {"up": 3, "down": 1}
    assert s["by_route"]["data_query"]["up"] == 2


def test_stats_degrade_when_table_absent() -> None:
    s = feedback_stats(engine=_BrokenEngine())
    assert s["available"] is False
    assert s["total"] == 0
    assert s["satisfaction"] is None
