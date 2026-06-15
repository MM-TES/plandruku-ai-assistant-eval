"""Phase 2 gate: intent router — heuristic precision + LLM fallback."""
from __future__ import annotations

from src.assistant.router import classify, heuristic_route
from src.assistant.schema import PageContext

from tests.assistant.conftest import FakeCompletion

# Obvious queries the heuristic fast-path must classify WITHOUT an LLM call.
OBVIOUS = [
    ("як передати у виробництво?", "instructions"),
    ("що мені тут робити?", "instructions"),
    ("скільки замовлень очікують?", "data_query"),
    ("покажи дефіцити", "data_query"),
    ("що змінилось за добу?", "history"),
    ("історія по замовленню 123", "history"),
    ("перебудуй розклад", "schedule_action"),
    ("вимкни машину ROTOMEC", "schedule_action"),
]
# Ambiguous → heuristic must abstain (None) and defer to the LLM.
AMBIGUOUS = ["Поясни чому це замовлення червоне", "Що з портфелем?"]


def test_heuristic_precision_and_coverage() -> None:
    correct = 0
    for q, expected in OBVIOUS:
        route = heuristic_route(q)
        if route is not None:
            assert route == expected, f"{q!r} → {route} (expected {expected})"  # precision
            correct += 1
    # coverage of the obvious set ≥ 0.85
    assert correct / len(OBVIOUS) >= 0.85


def test_ambiguous_fall_through() -> None:
    for q in AMBIGUOUS:
        assert heuristic_route(q) is None


def test_classify_uses_llm_when_no_heuristic(patch_llm_client) -> None:
    client = patch_llm_client([FakeCompletion('{"route": "analysis", "reason": "r"}')])
    res = classify("Поясни чому це замовлення червоне", PageContext(route="/workflow", stage="materialy"))
    assert res.route == "analysis"
    assert res.source == "llm"
    assert len(client.calls) == 1


def test_open_order_card_appears_in_router_context(patch_llm_client) -> None:
    client = patch_llm_client([FakeCompletion('{"route": "analysis", "reason": "r"}')])
    classify(
        "Поясни чому це замовлення червоне",
        PageContext(route="/workflow/portfolio", selected_order="15500"),
    )
    user_msg = client.calls[0]["messages"][-1]["content"]
    assert "#15500" in user_msg  # cheap router is told which order is open


def test_out_of_scope_refusal(patch_llm_client) -> None:
    patch_llm_client([FakeCompletion('{"route": "out_of_scope", "refusal": "Вибач, не можу."}')])
    res = classify("Купи мені акції Tesla")
    assert res.route == "out_of_scope"
    assert res.refusal


def test_screenshot_forces_vision_without_llm() -> None:
    # no client patched → must not touch the LLM
    res = classify("що тут на графіку?", has_screenshot=True)
    assert res.route == "screen_vision"
    assert res.source == "forced"
