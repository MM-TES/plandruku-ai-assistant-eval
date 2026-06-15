"""Phase 1 gate: pydantic contracts."""
from __future__ import annotations

from src.assistant.schema import AssistantRequest, PageContext


def test_request_roundtrip_and_defaults() -> None:
    req = AssistantRequest(message="скільки замовлень?", session_id="s1")
    assert req.scope == "standard"
    assert req.page_context.route == "/"
    assert req.screenshot_b64 is None


def test_scope_normalisation() -> None:
    assert AssistantRequest(message="x", scope="all").normalised_scope() == "all"
    assert AssistantRequest(message="x", scope="bogus").normalised_scope() == "standard"


def test_mode_normalisation() -> None:
    assert AssistantRequest(message="x").normalised_mode() == "hybrid"          # default
    assert AssistantRequest(message="x", mode="data").normalised_mode() == "data"
    assert AssistantRequest(message="x", mode="kb").normalised_mode() == "kb"
    assert AssistantRequest(message="x", mode="bogus").normalised_mode() == "hybrid"


def test_page_context_key() -> None:
    assert PageContext(route="/workflow", stage="materialy").key() == "/workflow:materialy"
    assert PageContext(route="/orders").key() == "/orders"


def test_page_context_carries_visible_ids() -> None:
    ctx = PageContext(route="/workflow", stage="zabezpechennia", visible_entity_ids=["123", "456"])
    assert ctx.visible_entity_ids == ["123", "456"]


def test_focus_order_prefers_selected_over_visible() -> None:
    ctx = PageContext(route="/workflow/portfolio", selected_order="15500",
                      visible_entity_ids=["100", "200"])
    assert ctx.focus_order_id() == 15500


def test_focus_order_falls_back_to_first_visible() -> None:
    ctx = PageContext(route="/workflow/portfolio", visible_entity_ids=["777", "888"])
    assert ctx.focus_order_id() == 777


def test_focus_order_none_when_nothing_open() -> None:
    assert PageContext(route="/workflow/portfolio").focus_order_id() is None


def test_focus_order_none_when_non_numeric() -> None:
    assert PageContext(route="/x", selected_order="abc").focus_order_id() is None


def test_focus_order_from_order_id_like_filter() -> None:
    # Виробництво filtered to a single order has no per-row data-order-id; the
    # ?order_id_like=12345 filter must still pin the focus.
    ctx = PageContext(route="/workflow", stage="vyrobnytstvo",
                      filters={"order_id_like": "12345"})
    assert ctx.focus_order_id() == 12345


def test_focus_order_ignores_partial_order_id_like() -> None:
    # a 3-digit substring filter is not a complete order id → not a focus
    assert PageContext(route="/workflow", filters={"order_id_like": "151"}).focus_order_id() is None
