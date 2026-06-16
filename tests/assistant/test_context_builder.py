"""Grounding: open-order summary + build() injection (DB-free via monkeypatched tools)."""
from __future__ import annotations

import pytest

from src.assistant.grounding import context_builder
from src.assistant.schema import PageContext

from tests.assistant.conftest import requires_import

_RISK_ROW = {
    "order_id": 15500,
    "customer_name": "Клієнт А",
    "structure_name": "PET/PE",
    "plan_shipment_op": "2026-06-10",
    "plan_kg": 1234.5,
    "total_allocated_kg": 980.0,
    "coverage_pct": 80.0,
    "total_deficit_kg": 250.0,
}


def _fake_tools(monkeypatch, *, risk=None, actions=None, raise_on=None):
    def _run(name, params=None, **kw):
        if raise_on and name == raise_on:
            raise RuntimeError("boom")
        if name in ("get_order", "order_risk"):
            return {"rows": list(risk or []), "error": None}
        if name == "pending_proposed_actions":
            return {"rows": list(actions or []), "error": None}
        return {"rows": [], "error": None}

    monkeypatch.setattr("src.assistant.data.tools.run_tool", _run)


def test_summarize_order_formats_params_and_deficit(monkeypatch) -> None:
    _fake_tools(monkeypatch, risk=[_RISK_ROW], actions=[])
    out = context_builder.summarize_order(15500)
    assert "#15500" in out
    assert "Клієнт А" in out and "PET/PE" in out
    assert "2026-06-10" in out
    assert "1234" in out or "1235" in out  # plan kg, rounded
    assert "підібрано рулонів 980 кг" in out  # allocated kg surfaced
    assert "80%" in out
    assert "250 кг" in out  # deficit surfaced


def test_summarize_order_lists_actions_human_text(monkeypatch) -> None:
    actions = [
        {"09_rationale_text": "Зняти дефіцит — склад покриває потребу.", "18_impact_kg": 120.0},
        {"09_rationale_text": "Створити закупку на нестачу.", "18_impact_kg": 80.0},
    ]
    _fake_tools(monkeypatch, risk=[_RISK_ROW], actions=actions)
    out = context_builder.summarize_order(15500)
    assert "Зняти дефіцит" in out
    assert "Створити закупку" in out
    assert "~120 кг" in out


def test_summarize_order_caps_actions_at_six(monkeypatch) -> None:
    actions = [{"09_rationale_text": f"Дія {i}", "18_impact_kg": i} for i in range(8)]
    _fake_tools(monkeypatch, risk=[_RISK_ROW], actions=actions)
    out = context_builder.summarize_order(15500)
    assert out.count("- ") == 6  # only the top 6 are surfaced


def test_summarize_order_truncates_long_rationale(monkeypatch) -> None:
    actions = [{"09_rationale_text": "x" * 400, "18_impact_kg": 10.0}]
    _fake_tools(monkeypatch, risk=[_RISK_ROW], actions=actions)
    out = context_builder.summarize_order(15500)
    assert "…" in out
    assert "x" * 400 not in out


def test_summarize_order_error_safe(monkeypatch) -> None:
    _fake_tools(monkeypatch, raise_on="get_order")
    assert context_builder.summarize_order(15500) == ""


def test_build_injects_order_block_and_does_not_truncate_it(monkeypatch) -> None:
    monkeypatch.setattr(context_builder, "live_summary", lambda: "")
    long_block = "Відкрита картка замовлення #15500\n" + "деталь; " * 400
    monkeypatch.setattr(context_builder, "summarize_order", lambda oid: long_block)
    ctx = PageContext(route="/workflow/portfolio", selected_order="15500")
    out = context_builder.build(ctx, max_chars=1400)
    assert "#15500" in out
    assert long_block in out  # whole block kept despite > max_chars base budget


def test_build_includes_plan_context(monkeypatch) -> None:
    monkeypatch.setattr(context_builder, "live_summary", lambda: "")
    ctx = PageContext(
        route="/schedule",
        plan_context="Активний розклад: «Мій план» — 59 замовлень у розкладі.",
    )
    out = context_builder.build(ctx, include_live=False)
    assert "«Мій план» — 59 замовлень" in out  # server-truth plan state reaches the prompt


def test_build_includes_visible_screen_text(monkeypatch) -> None:
    monkeypatch.setattr(context_builder, "live_summary", lambda: "")
    ctx = PageContext(route="/workflow/actions",
                      visible_text="До перегляду: 247 дій · 37 не зняти — потрібна закупка")
    out = context_builder.build(ctx, include_live=False)
    assert "37 не зняти" in out               # on-screen headline reaches the prompt
    assert "видимий зараз на екрані" in out


def test_available_links_resolves_order_id_and_keeps_global(monkeypatch) -> None:
    ctx = PageContext(route="/workflow/portfolio", selected_order="15500")
    out = context_builder.available_links(ctx)
    assert "/orders/15500/allocate" in out      # order-scoped link, id substituted
    assert "{order_id}" not in out              # placeholder fully resolved
    assert "/workflow/actions" in out           # global link always offered


def test_available_links_skips_order_links_when_none_open() -> None:
    out = context_builder.available_links(PageContext(route="/kpi"))
    assert "{order_id}" not in out
    assert "/orders/" not in out                # order-scoped links dropped
    assert "/workflow/actions" in out           # global links still present


def test_extract_sku_variants() -> None:
    e = context_builder.extract_sku
    assert e("який залишок по 2.01.51045?") == "2.01.51045"
    assert e("матеріал 6.0809.133 у дефіциті") == "6.0809.133"
    assert e("скільки дефіцитів?") is None
    assert e("до 27.07.2026") is None          # date, not a SKU
    assert e("") is None


def test_live_summary_has_counter_glossary(monkeypatch) -> None:
    requires_import("src.web.templating")
    monkeypatch.setattr(
        "src.web.templating._nav_counts",
        lambda: {"sales": 1, "materials": 2, "production": 3, "portfolio": 184,
                 "deficits": 73, "actions": 245, "commitments": 5},
    )
    out = context_builder.live_summary()
    assert "184" in out and "73" in out
    assert "портфель-ризик = замовлення з ризиком" in out  # counter semantics pinned
    assert "Не плутай ці лічильники" in out


def test_extract_order_id_variants() -> None:
    e = context_builder.extract_order_id
    assert e("а що з замовленням #12345?") == 12345
    assert e("замовлення 15636") == 15636
    assert e("покажи #15499") == 15499
    assert e("Що означає 37 не зняти — потрібна закупка") is None  # no false positive
    assert e("до 25 травня 2026 року") is None                    # date, not an order
    assert e("") is None


def test_summarize_order_flags_full_plus_deficit_anomaly(monkeypatch) -> None:
    row = dict(_RISK_ROW)
    row.update({"total_deficit_kg": 2832.0, "coverage_status": "full", "is_handed_off": True})
    _fake_tools(monkeypatch, risk=[row], actions=[])
    out = context_builder.summarize_order(12345)
    assert "АНОМАЛІЯ" in out
    assert "2832 кг" in out


def test_summarize_order_no_anomaly_when_consistent(monkeypatch) -> None:
    row = dict(_RISK_ROW)  # coverage 80%, deficit 250 — consistent, no full-coverage flag
    _fake_tools(monkeypatch, risk=[row], actions=[])
    assert "АНОМАЛІЯ" not in context_builder.summarize_order(15500)


def test_data_freshness_marker_is_error_safe() -> None:
    out = context_builder.data_freshness_marker()
    assert out.startswith("📊 За даними системи")  # never blank, never raises


def test_summarize_order_requested_header(monkeypatch) -> None:
    _fake_tools(monkeypatch, risk=[_RISK_ROW], actions=[])
    opened = context_builder.summarize_order(15500, opened=True)
    requested = context_builder.summarize_order(15500, opened=False)
    assert "Відкрита картка замовлення #15500" in opened
    assert "про яке запитує оператор" in requested


def test_build_no_order_block_when_nothing_open(monkeypatch) -> None:
    monkeypatch.setattr(context_builder, "live_summary", lambda: "")

    def _boom(_oid):  # must not be called when no card is open
        raise AssertionError("summarize_order called without a focus order")

    monkeypatch.setattr(context_builder, "summarize_order", _boom)
    out = context_builder.build(PageContext(route="/workflow/portfolio"), include_live=False)
    assert "Відкрита картка" not in out
