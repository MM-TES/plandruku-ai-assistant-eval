"""Builds the structured grounded-context block injected on every request.

Combines the page description (config) with a best-effort live-data summary
(reusing the error-safe ``_nav_counts`` already used by the nav badges). When the
operator has an order card open, the open order's live state (key params, deficit,
proposed next-step actions) is injected too, so the answer model grounds on the
order actually on screen. Never raises — grounding must not break the assistant.
"""
from __future__ import annotations

import re

from src.assistant.grounding.page_registry import describe
from src.assistant.schema import PageContext

# Order id explicitly named in a question: «#12345», «№12345», «замовлення 12345»,
# «order 12345». Requires a marker so plain numbers (kg, dates, counts) are not
# mistaken for order ids.
_ORDER_ID_RE = re.compile(
    r"(?:#|№|замовленн\w*\s*№?\s*|order\s*#?\s*)\s*(\d{3,7})", re.IGNORECASE
)


# Nomenclature SKU like 2.01.51045 / 6.0809.133 named in a question.
_SKU_RE = re.compile(r"\b\d\.\d{2,4}\.\d{2,6}\b")


def extract_sku(message: str) -> str | None:
    """The material SKU code the operator names in *message*, or ``None`` —
    used to resolve the material from verified data (get_material) regardless of
    what's on screen."""
    if not message:
        return None
    m = _SKU_RE.search(message)
    return m.group(0) if m else None


def extract_order_id(message: str) -> int | None:
    """The order id the operator explicitly names in *message*, or ``None``.

    Lets a named order override the open card so the assistant is not trapped on
    the drawer when the operator asks about a different order.
    """
    if not message:
        return None
    m = _ORDER_ID_RE.search(message)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def live_summary() -> str:
    """Page-agnostic live counters from the same source as the nav badges."""
    try:
        from src.web.templating import _nav_counts

        c = _nav_counts()
        return (
            "Поточні лічильники системи: продажі (нові) {sales}, матеріали {materials}, "
            "виробництво {production}, портфель-ризик {portfolio}, дефіцити {deficits}, "
            "дій у фіді «Що зробити» {actions}, активних зобов'язань постачання {commitments}. "
            "Що означає кожен лічильник: «Контроль»/портфель-ризик = замовлення з ризиком "
            "зриву строку (червоні+помаранчеві); «Дефіцити» = кількість матеріалів (SKU) у "
            "нестачі; «Продажі»/«Матеріали»/«Виробництво» = замовлення на відповідному етапі; "
            "«Що зробити» = пропоновані дії. Не плутай ці лічильники між собою."
        ).format(**{k: c.get(k, 0) for k in (
            "sales", "materials", "production", "portfolio",
            "deficits", "actions", "commitments")})
    except Exception:  # noqa: BLE001 — grounding must not break the assistant
        return ""


def data_freshness_marker() -> str:
    """Source + data-time marker (§4) appended to fact-bearing answers, so the
    operator sees where a number came from and how fresh it is. Error-safe → ""."""
    try:
        from src.change_tracking.readers import get_latest_etl_run_meta_cached
        from src.utils.db_connection import get_engine

        meta = get_latest_etl_run_meta_cached(get_engine())
        when = meta.get("completed_at") if isinstance(meta, dict) else getattr(meta, "completed_at", None)
        if when is not None and hasattr(when, "strftime"):
            return f"📊 За даними системи · станом на {when:%d.%m.%Y %H:%M}"
        return "📊 За даними системи"
    except Exception:  # noqa: BLE001 — marker must never break the answer
        return "📊 За даними системи"


def _round_kg(value) -> str | None:
    """Render a kg value as a whole number, or ``None`` if not numeric."""
    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return None


def summarize_order(order_id: int, *, opened: bool = True) -> str:
    """Live summary of an order: key params + deficit + the concrete proposed
    next-step actions (reuses the read-only data tools).

    ``opened=True`` labels it as the open card; ``opened=False`` labels it as the
    order the operator explicitly asked about (so a named order is not confused
    with the drawer). Surfaces the human ``09_rationale_text`` of each proposed
    action rather than the internal action code. Error-safe: returns ``""`` on any
    failure so a slow/absent DB never breaks the assistant.
    """
    try:
        from src.assistant.data import tools

        lines: list[str] = []
        # get_order resolves ANY order (incl. handed-off) and carries allocated kg.
        risk_rows = tools.run_tool("get_order", {"order_id": order_id}).get("rows") or []
        if risk_rows:
            r = risk_rows[0]
            head0 = (
                f"Відкрита картка замовлення #{order_id}"
                if opened
                else f"Замовлення #{order_id} (про яке запитує оператор)"
            )
            head = [head0]
            if r.get("customer_name"):
                head.append(f"клієнт {r['customer_name']}")
            if r.get("structure_name"):
                head.append(f"матеріал {r['structure_name']}")
            lines.append(", ".join(head) + ".")

            facts: list[str] = []
            if r.get("plan_shipment_op"):
                facts.append(f"строк {r['plan_shipment_op']}")
            plan = _round_kg(r.get("plan_kg"))
            if plan:
                facts.append(f"план {plan} кг")
            alloc = _round_kg(r.get("total_allocated_kg"))
            try:
                has_alloc = r.get("total_allocated_kg") is not None and float(r["total_allocated_kg"]) > 0
            except (TypeError, ValueError):
                has_alloc = False
            if alloc and has_alloc:
                facts.append(f"підібрано рулонів {alloc} кг")
            cov = r.get("coverage_pct")
            if cov is not None:
                try:
                    facts.append(f"покриття матеріалами {float(cov):.0f}%")
                except (TypeError, ValueError):
                    pass
            deficit = r.get("total_deficit_kg")
            deficit_kg = _round_kg(deficit)
            try:
                has_deficit = deficit is not None and float(deficit) > 0
            except (TypeError, ValueError):
                has_deficit = False
            if deficit_kg and has_deficit:
                facts.append(f"дефіцит {deficit_kg} кг")
            if facts:
                lines.append("; ".join(facts) + ".")

            # Anomaly flag (§8): a "full"/handed-off order that still has a deficit
            # is a data contradiction the operator must see, with the exact fields.
            if has_deficit:
                full_cov = (str(r.get("coverage_status") or "") == "full"
                            or str(r.get("readiness") or "") == "full_stock")
                if full_cov:
                    lines.append(
                        f"⚠ АНОМАЛІЯ: статус повного покриття, але дефіцит {deficit_kg} кг "
                        f"(поля coverage_status/total_deficit_kg суперечать) — познач це й порадь звірити."
                    )
                elif r.get("is_handed_off"):
                    lines.append(
                        f"⚠ АНОМАЛІЯ: замовлення передане у виробництво, але має дефіцит "
                        f"{deficit_kg} кг — познач це й порадь звірити."
                    )

        act_rows = tools.run_tool(
            "pending_proposed_actions", {"order_id": order_id}
        ).get("rows") or []
        if act_rows:
            lines.append("Пропоновані дії для цього замовлення:")
            for a in act_rows[:6]:
                txt = (a.get("09_rationale_text") or "").strip()
                if len(txt) > 160:
                    txt = txt[:157] + "…"
                impact = _round_kg(a.get("18_impact_kg"))
                suffix = f" (~{impact} кг)" if impact else ""
                lines.append(f"- {txt}{suffix}" if txt else f"- дія{suffix}")

        return "\n".join(lines)
    except Exception:  # noqa: BLE001 — grounding must not break the assistant
        return ""


def available_links(page_context: PageContext) -> str:
    """Allowlist of internal destinations the answer may turn into clickable links.

    Order-scoped links are resolved with the open order id and skipped when no
    card is open. The answer prompt instructs the model to use ONLY these URLs
    verbatim, so links are correct routes (never hallucinated). Error-safe → "".
    """
    try:
        from src.assistant import config

        oid = page_context.focus_order_id()
        rows: list[str] = []
        for it in config.links():
            url = str(it.get("url", ""))
            label = str(it.get("label", ""))
            if not url or not label:
                continue
            if "{order_id}" in url:
                if oid is None:
                    continue
                url = url.replace("{order_id}", str(oid))
            elif it.get("order_scoped") and oid is None:
                continue
            rows.append(f"- {label}: {url}")
        if not rows:
            return ""
        return (
            "Доступні посилання (для клікабельних markdown-посилань — "
            "використовуй ТІЛЬКИ ці URL дослівно):\n" + "\n".join(rows)
        )
    except Exception:  # noqa: BLE001 — grounding must not break the assistant
        return ""


def build(page_context: PageContext, *, include_live: bool = True, max_chars: int = 1400) -> str:
    """Return a compact grounded-context block for the prompt."""
    d = describe(page_context)
    parts: list[str] = [f"Поточна сторінка: {page_context.key()}"]
    if d.get("desc"):
        parts.append(str(d["desc"]))
    # The order the operator has open — injected for EVERY route (independent of
    # include_live) so instructions/clarify/analysis all see it.
    order_block = ""
    oid = page_context.focus_order_id()
    if oid is not None:
        order_block = summarize_order(oid)
        if order_block:
            parts.append(order_block)
    # Server truth about the active schedule — so plan questions get the real
    # active-plan count, NOT the pending-orders count.
    if page_context.plan_context:
        parts.append(page_context.plan_context)
    # Prominent text the operator actually sees (headings, summary bars) — lets the
    # assistant answer about on-screen numbers (e.g. «37 не зняти») via text.
    if page_context.visible_text:
        parts.append(f"Текст, видимий зараз на екрані: {page_context.visible_text[:700]}")
    if page_context.visible_entity_ids:
        ids = ", ".join(str(x) for x in page_context.visible_entity_ids[:20])
        parts.append(f"Видимі на екрані ID: {ids}")
    if page_context.filters:
        parts.append(f"Активні фільтри: {page_context.filters}")
    links_block = available_links(page_context)
    if links_block:
        parts.append(links_block)
    if include_live:
        ls = live_summary()
        if ls:
            parts.append(ls)
    # Give the open-order + links blocks their own headroom so they are never
    # truncated away — they are the whole point of grounding when a card is open.
    limit = max_chars + len(order_block) + len(links_block) + (256 if order_block else 0)
    return "\n".join(p for p in parts if p)[:limit]


__all__ = [
    "build", "live_summary", "summarize_order", "available_links",
    "extract_order_id", "extract_sku", "data_freshness_marker",
]
