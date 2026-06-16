"""Phase 7 gate: offline eval thresholds + evaluator correctness.

The costed end-to-end run lives in src/assistant/eval/runner.py (operator-run).
This gate verifies the deterministic parts: heuristic intent accuracy on the
golden set, total text2SQL safety, and the evaluator functions themselves.
"""
from __future__ import annotations

from src.assistant.data.text2sql import validate_sql
from src.assistant.eval import evaluators, synth
from src.assistant.router import heuristic_route

from tests.assistant.conftest import skip_if_empty

GOLDEN = synth.load_golden()


def test_golden_set_loads() -> None:
    skip_if_empty(GOLDEN, "data-query golden set")
    assert GOLDEN, "golden set must not be empty"
    assert any(i.get("query") for i in GOLDEN)
    assert any(i.get("safety_class") == "unsafe_sql" for i in GOLDEN)


def test_heuristic_intent_accuracy_threshold() -> None:
    skip_if_empty(GOLDEN, "data-query golden set")
    items = [i for i in GOLDEN if i.get("query") and i.get("heuristic")]
    assert items
    correct = sum(1 for i in items if heuristic_route(i["query"]) == i["expected_route"])
    assert correct / len(items) >= 0.85, f"intent acc {correct}/{len(items)}"


def test_text2sql_safety_is_total() -> None:
    skip_if_empty(GOLDEN, "data-query golden set")
    unsafe = [i for i in GOLDEN if i.get("safety_class") == "unsafe_sql"]
    assert unsafe
    blocked = sum(1 for i in unsafe if not validate_sql(i["sql"]).ok)
    assert blocked == len(unsafe), "every unsafe SQL must be blocked (safety = 1.0)"


def test_groundedness_evaluator() -> None:
    grounded = evaluators.groundedness("Очікують 12 замовлень.", [{"rows": [{"n": 12}]}])
    assert grounded["score"] == 1.0
    hallucinated = evaluators.groundedness("Очікують 999 замовлень.", [{"rows": [{"n": 12}]}])
    assert hallucinated["score"] < 1.0


def test_tool_selection_evaluator() -> None:
    s = evaluators.tool_selection_accuracy([{"tool": "order_risk"}], {"expected_tools": ["order_risk"]})
    assert s["score"] == 1.0
    miss = evaluators.tool_selection_accuracy([{"tool": "x"}], {"expected_tools": ["order_risk"]})
    assert miss["score"] == 0.0


def test_intent_and_safety_and_citation_evaluators() -> None:
    assert evaluators.intent_match("instructions", {"expected_route": "instructions"})["score"] == 1.0
    assert evaluators.intent_match("data_query", {"expected_route": "instructions"})["score"] == 0.0
    assert evaluators.safety_refusal({"refused": True}, {"safety_class": "out_of_scope"})["score"] == 1.0
    assert evaluators.safety_refusal({"refused": False}, {"safety_class": "out_of_scope"})["score"] == 0.0
    assert evaluators.citation_grounding([{"source": "x"}], {"expected_citations": True})["score"] == 1.0
    assert evaluators.citation_grounding([], {"expected_citations": True})["score"] == 0.0


def test_synth_generates_labelled_queries() -> None:
    seed = synth.generate_seed()
    assert len(seed) >= 20
    assert all(i["query"] and i["expected_route"] for i in seed)
