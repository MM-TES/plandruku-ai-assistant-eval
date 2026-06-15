"""Phase 3 gate: read-only tool library — every tool is SELECT-only & parameterized."""
from __future__ import annotations

import pytest

from src.assistant.data import tools
from src.assistant.data.text2sql import validate_sql


@pytest.mark.parametrize("spec", tools.list_tools(), ids=lambda s: s.name)
def test_every_tool_sql_passes_the_ast_guard(spec) -> None:
    """Cross-check: each tool's SQL must satisfy the same safety guard as text2SQL."""
    v = validate_sql(spec.sql, max_rows=200)
    assert v.ok, f"tool {spec.name} SQL rejected by guard: {v.error}"


@pytest.mark.parametrize("spec", tools.list_tools(), ids=lambda s: s.name)
def test_tool_sql_has_no_format_injection(spec) -> None:
    assert "{" not in spec.sql and "%s" not in spec.sql and "%(" not in spec.sql
    for param in spec.params:
        assert f":{param}" in spec.sql, f"declared param {param} not bound in {spec.name}"


def test_unknown_tool_returns_error_without_db() -> None:
    res = tools.run_tool("does_not_exist")
    assert res["error"]
    assert res["rows"] == []


def test_catalogue_text_lists_all_tools() -> None:
    text = tools.tool_catalogue_text()
    for spec in tools.list_tools():
        assert spec.name in text


def test_get_order_tool_returns_allocated_kg_field() -> None:
    spec = tools.TOOLS["get_order"]
    assert spec.params == ["order_id"]
    assert "total_allocated_kg" in spec.sql   # рулони (allocated) surfaced
    assert tools._REQUIRED_PARAMS.get("get_order") == ["order_id"]


def test_tool_schemas_expose_function_calling_shape() -> None:
    schemas = tools.tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert {"get_material", "get_deficits_top", "order_risk", "get_order"} <= names
    gm = next(s for s in schemas if s["function"]["name"] == "get_material")
    props = gm["function"]["parameters"]["properties"]
    assert props["sku"]["type"] == "string"
    assert gm["function"]["parameters"]["required"] == ["sku"]   # essential param
    # every schema is a valid function object
    for s in schemas:
        assert s["type"] == "function" and s["function"]["name"]
        assert s["function"]["parameters"]["type"] == "object"


def test_python_backed_tools_in_schema() -> None:
    names = {s["function"]["name"] for s in tools.tool_schemas()}
    assert {"get_orders", "get_counters", "get_deficits"} <= names
    go = next(s for s in tools.tool_schemas() if s["function"]["name"] == "get_orders")
    assert go["function"]["parameters"]["required"] == ["stage"]
    assert "stage" in go["function"]["parameters"]["properties"]


def test_run_tool_dispatches_python_tool(monkeypatch) -> None:
    monkeypatch.setitem(tools._PY_TOOLS["get_counters"], "fn", lambda p: [{"n_sales": 5, "n_materials": 9}])
    res = tools.run_tool("get_counters", {})
    assert res["error"] is None
    assert res["rows"] == [{"n_sales": 5, "n_materials": 9}]
    assert res["columns"] == ["n_sales", "n_materials"]


def test_run_tool_python_tool_error_safe(monkeypatch) -> None:
    def _boom(_p):
        raise RuntimeError("db down")

    monkeypatch.setitem(tools._PY_TOOLS["get_orders"], "fn", _boom)
    res = tools.run_tool("get_orders", {"stage": "materialy"})
    assert res["rows"] == [] and res["error"]


def test_pending_proposed_actions_takes_optional_order_filter() -> None:
    spec = tools.TOOLS["pending_proposed_actions"]
    assert spec.params == ["order_id"]
    assert ":order_id IS NULL OR" in spec.sql  # backward-compatible: NULL → all orders


# --- integration (needs the read-only role + DB) ---------------------------
def test_tool_executes_against_db_if_available() -> None:
    from src.assistant.data.engine import ro_available

    if not ro_available():
        pytest.skip("read-only DB role not available")
    res = tools.run_tool("pending_orders_count")
    assert res["error"] is None
    assert res["columns"] == ["n_pending"]
