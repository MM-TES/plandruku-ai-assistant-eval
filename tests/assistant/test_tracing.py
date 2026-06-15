"""Phase 1 gate: traceable pass-through + JSON parsing."""
from __future__ import annotations

from src.assistant.tracing import parse_json_object, traceable


def test_traceable_passthrough_when_off() -> None:
    # conftest forces LANGSMITH_TRACING=false → no langsmith import, plain call
    @traceable
    def add(a: int, b: int) -> int:
        return a + b

    @traceable(name="mul", tags=["t"])
    def mul(a: int, b: int) -> int:
        return a * b

    assert add(2, 3) == 5
    assert mul(2, 3) == 6


def test_parse_json_object_handles_fences() -> None:
    assert parse_json_object('{"route": "instructions"}')["route"] == "instructions"
    fenced = '```json\n{"route": "data_query", "reason": "x"}\n```'
    assert parse_json_object(fenced)["route"] == "data_query"
    # embedded object in prose
    assert parse_json_object('blah {"a": 1} tail')["a"] == 1
    assert parse_json_object("not json at all") == {}
