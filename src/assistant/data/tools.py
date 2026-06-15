"""Read-only parameterized data tools over existing aps views.

These cover the common operator questions safely (named-constant SQL, bound
parameters only — no f-string user input, all SELECT). Open-ended questions fall
back to guarded text2SQL. Every tool runs on the read-only engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.assistant import config
from src.utils.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass
class ToolSpec:
    name: str
    description: str  # Ukrainian — surfaced to the model + UI
    sql: str          # named-constant; parameters only
    params: list[str] = field(default_factory=list)


# --- tool catalogue ---------------------------------------------------------
TOOLS: dict[str, ToolSpec] = {
    "pending_orders_count": ToolSpec(
        "pending_orders_count",
        "Кількість активних замовлень, що очікують планування.",
        "SELECT count(*) AS n_pending FROM aps.v_pending_orders",
    ),
    "material_readiness_breakdown": ToolSpec(
        "material_readiness_breakdown",
        "Розподіл замовлень за станом готовності матеріалів.",
        "SELECT readiness, count(*) AS n FROM aps.v_material_readiness "
        "GROUP BY readiness ORDER BY n DESC",
    ),
    "order_risk": ToolSpec(
        "order_risk",
        "Ризики/дефіцити замовлень (опційно по одному замовленню).",
        "SELECT * FROM aps.v_order_risk "
        "WHERE (:order_id IS NULL OR order_id = :order_id) "
        "ORDER BY total_deficit_kg DESC NULLS LAST",
        params=["order_id"],
    ),
    "allocation_status_breakdown": ToolSpec(
        "allocation_status_breakdown",
        "Розподіл замовлень за статусом покриття матеріалами.",
        "SELECT coverage_status, count(*) AS n FROM aps.v_allocation_status "
        "GROUP BY coverage_status ORDER BY n DESC",
    ),
    "shipments_for_order": ToolSpec(
        "shipments_for_order",
        "Відвантаження (продукт + зразки) по конкретному замовленню.",
        "SELECT \"1_data_vidvantazhennia\", \"68_vaha_netto_kg\", is_sample, source "
        "FROM aps.v_shipments_all WHERE \"36_zamovlennia\" = :order_id "
        "ORDER BY \"1_data_vidvantazhennia\" DESC",
        params=["order_id"],
    ),
    "stale_reservations": ToolSpec(
        "stale_reservations",
        "Прострочені/orphan-бронювання рулонів.",
        "SELECT * FROM aps.v_stale_reservations",
    ),
    "pending_proposed_actions": ToolSpec(
        "pending_proposed_actions",
        "Активні пропоновані дії у фіді «Що зробити» (опційно по одному замовленню).",
        "SELECT \"04_action_kind\", \"06_order_id\", \"07_sku\", \"18_impact_kg\", "
        "\"09_rationale_text\" FROM aps.proposed_actions WHERE \"12_status\" = 'pending' "
        "AND (:order_id IS NULL OR \"06_order_id\" = :order_id) "
        "ORDER BY \"18_impact_kg\" DESC NULLS LAST",
        params=["order_id"],
    ),
    "get_order": ToolSpec(
        "get_order",
        "Повні дані одного замовлення за № (будь-якого, навіть переданого у виробництво): "
        "клієнт, матеріал, план кг, підібрано рулонів (allocated кг), дефіцит кг, "
        "покриття %, статус покриття, строк, стан.",
        "SELECT o.\"10_order_id\" AS order_id, o.\"01_customer_name\" AS customer_name, "
        "o.\"24_structure_name\" AS structure_name, "
        "COALESCE(NULLIF(o.\"28_plan_kg\", 0), o.\"29_plan_kg_calc\", 0) AS plan_kg, "
        "o.\"50_plan_shipment_op\" AS plan_shipment_op, o.\"60_state_name\" AS state_name, "
        "mr.readiness, mr.coverage_pct, "
        "vas.coverage_status, vas.total_allocated_kg, vas.total_deficit_kg, "
        "EXISTS (SELECT 1 FROM aps.production_handoff_records h "
        "WHERE h.\"01_order_id\" = o.\"10_order_id\") AS is_handed_off "
        "FROM aps.orders o "
        "LEFT JOIN aps.v_material_readiness mr ON mr.order_id = o.\"10_order_id\" "
        "LEFT JOIN aps.v_allocation_status vas ON vas.order_id = o.\"10_order_id\" "
        "WHERE o.\"10_order_id\" = :order_id",
        params=["order_id"],
    ),
    "get_deficits_top": ToolSpec(
        "get_deficits_top",
        "Топ дефіцитів за матеріалом (SKU): сумарна нестача в кг і к-ть замовлень.",
        "SELECT \"04_nomenklatura_kod\" AS sku, "
        "SUM(\"07_deficit_kg\") AS total_deficit_kg, "
        "COUNT(DISTINCT \"02_order_id\") AS order_count "
        "FROM aps.deficit_confirmations WHERE \"11_revoked_at\" IS NULL "
        "GROUP BY \"04_nomenklatura_kod\" ORDER BY total_deficit_kg DESC "
        "LIMIT COALESCE(:limit, 10)",
        params=["limit"],
    ),
    "search_materials": ToolSpec(
        "search_materials",
        "Пошук матеріалів за підрядком назви або виробника (напр. Plastchim, Taghleef, "
        "FXCW, BOPP). Назви виробників у даних — ЛАТИНИЦЕЮ (Plastchim, не ПЛАСТХІМ): "
        "передавай латинську форму. Повертає код SKU, назву, залишок на складі (кг). "
        "Джерело — матеріали, що є на складі.",
        "SELECT nomenklatura_kod AS sku, nomenklatura_nazva AS name, "
        "total_masa_kg AS stock_kg, n_rolls "
        "FROM aps.v_inventory_summary "
        "WHERE nomenklatura_nazva ILIKE '%' || :query || '%' "
        "ORDER BY nomenklatura_nazva LIMIT COALESCE(:limit, 25)",
        params=["query", "limit"],
    ),
    "get_material": ToolSpec(
        "get_material",
        "Матеріал за кодом SKU: назва, залишок на складі (кг), активний дефіцит (кг).",
        "SELECT :sku AS sku, "
        "(SELECT nomenklatura_nazva FROM aps.v_inventory_summary WHERE nomenklatura_kod = :sku) AS name, "
        "COALESCE((SELECT total_masa_kg FROM aps.v_inventory_summary WHERE nomenklatura_kod = :sku), 0) AS stock_kg, "
        "COALESCE((SELECT n_rolls FROM aps.v_inventory_summary WHERE nomenklatura_kod = :sku), 0) AS n_rolls, "
        "COALESCE((SELECT SUM(\"07_deficit_kg\") FROM aps.deficit_confirmations "
        "WHERE \"11_revoked_at\" IS NULL AND \"04_nomenklatura_kod\" = :sku), 0) AS deficit_kg",
        params=["sku"],
    ),
    "etl_diff_summary": ToolSpec(
        "etl_diff_summary",
        "Зведення змін у даних за останній цикл оновлення.",
        "SELECT * FROM aps.v_etl_diff_summary",
    ),
    "latest_etl_run": ToolSpec(
        "latest_etl_run",
        "Метадані останнього завершеного оновлення даних (свіжість).",
        "SELECT * FROM aps.v_latest_completed_etl_run",
    ),
    "supply_commitment_events_recent": ToolSpec(
        "supply_commitment_events_recent",
        "Останні події постачання (append-only журнал).",
        "SELECT \"02_commitment_id\", \"03_action\", \"04_occurred_at\", \"06_details\" "
        "FROM aps.supply_commitment_events ORDER BY \"04_occurred_at\" DESC",
    ),
}


# param name → (json type, description) for the function-calling schema.
_PARAM_META: dict[str, tuple[str, str]] = {
    "order_id": ("integer", "Код замовлення (ціле число), напр. 12345."),
    "sku": ("string", "Код матеріалу (SKU), напр. 2.01.51045 або 6.0809.133."),
    "limit": ("integer", "Скільки рядків повернути (за замовчуванням 10)."),
    "stage": ("string", "Етап: prodazhi (продажі), materialy (матеріали) або vyrobnytstvo (виробництво)."),
    "customer_like": ("string", "Фільтр за клієнтом (підрядок назви)."),
    "order_id_like": ("string", "Фільтр за номером замовлення (підрядок)."),
    "sort": ("string", "Сортування: delivery_date_asc | plan_kg_desc | customer | order_id."),
    "query": ("string", "Підрядок назви матеріалу/виробника (латиницею: Plastchim, FXCW, BOPP)."),
}
# tools where the param is essential for a meaningful result → mark required.
_REQUIRED_PARAMS: dict[str, list[str]] = {
    "get_material": ["sku"],
    "get_order": ["order_id"],
    "search_materials": ["query"],
    "shipments_for_order": ["order_id"],
}


def list_tools() -> list[ToolSpec]:
    return list(TOOLS.values())


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-style function schemas for the whole catalogue — lets the answer
    model CHOOSE which typed function to call (and with what args) instead of a
    heuristic, so any phrasing resolves to verified data."""
    out: list[dict[str, Any]] = []
    catalogue: list[tuple[str, str, list[str], list[str]]] = [
        (s.name, s.description, s.params, _REQUIRED_PARAMS.get(s.name, []))
        for s in TOOLS.values()
    ] + [
        (name, meta["description"], meta["params"], meta["required"])
        for name, meta in _PY_TOOLS.items()
    ]
    for name, desc, params, required in catalogue:
        props = {p: {"type": _PARAM_META.get(p, ("string", p))[0],
                     "description": _PARAM_META.get(p, ("string", p))[1]} for p in params}
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        })
    return out


def tool_catalogue_text() -> str:
    """Compact catalogue for prompting the model which tools exist."""
    return "\n".join(f"- {t.name}: {t.description}" for t in TOOLS.values())


# --- Python-backed tools -----------------------------------------------------
# Some answers need data that isn't a single view (stage queues, counters,
# deficits enriched with supply). These wrap the existing Python query functions
# (lazy-imported so importing this module stays light). Same {rows} contract.
_VALID_STAGES = {"prodazhi", "materialy", "vyrobnytstvo"}


def _py_get_counters(params: dict[str, Any]) -> list[dict[str, Any]]:
    from src.material_planning.workflow.stages import fetch_stage_counters_cached
    from src.utils.db_connection import get_engine

    c = fetch_stage_counters_cached(get_engine())
    return [{
        "n_sales": c.n_sales, "n_materials": c.n_materials,
        "n_production": c.n_production,
        "n_zabezpechennia": getattr(c, "n_zabezpechennia", 0),
        "n_inbox_pending": getattr(c, "n_inbox_pending", 0),
    }]


def _py_get_orders(params: dict[str, Any]) -> list[dict[str, Any]]:
    from dataclasses import asdict

    from src.material_planning.workflow.stages import fetch_stage_queue
    from src.utils.db_connection import get_engine

    stage = str(params.get("stage") or "materialy").strip().lower()
    if stage not in _VALID_STAGES:
        stage = "materialy"
    try:
        limit = max(1, min(int(params.get("limit") or 10), 50))
    except (TypeError, ValueError):
        limit = 10
    q = fetch_stage_queue(
        get_engine(), stage, page=1,
        sort=str(params.get("sort") or "delivery_date_asc"),
        coverage_filter=str(params.get("coverage_filter") or "all"),
        customer_like=params.get("customer_like") or None,
        order_id_like=params.get("order_id_like") or None,
    )
    keep = ("order_id", "customer_name", "structure_name", "plan_kg",
            "plan_shipment_date", "coverage_status", "total_allocated_kg",
            "total_deficit_kg", "is_handed_off")
    out: list[dict[str, Any]] = []
    for o in (getattr(q, "orders", None) or [])[:limit]:
        d = asdict(o) if hasattr(o, "__dataclass_fields__") else dict(getattr(o, "__dict__", {}))
        out.append({k: d.get(k) for k in keep})
    return out


def _py_get_deficits(params: dict[str, Any]) -> list[dict[str, Any]]:
    from src.material_planning.supply.handlers import list_deficits_with_supply
    from src.utils.db_connection import get_engine

    rows = list_deficits_with_supply(get_engine(), filter_sku=params.get("sku") or None)
    try:
        limit = max(1, min(int(params.get("limit") or 10), 50))
    except (TypeError, ValueError):
        limit = 10
    keep = ("nomenklatura_kod", "sku_name", "total_deficit_kg", "order_count",
            "coverage_pct", "active_supply_kg", "draft_supply_kg", "worst_status", "next_eta")
    out: list[dict[str, Any]] = []
    for r in (rows or [])[:limit]:
        out.append({k: r.get(k) for k in keep if k in r})
    return out


_PY_TOOLS: dict[str, dict[str, Any]] = {
    "get_counters": {
        "description": "Лічильники етапів: скільки замовлень у Продажах/Матеріалах/Виробництві, активних дій.",
        "params": [], "required": [], "fn": _py_get_counters,
    },
    "get_orders": {
        "description": "Список замовлень на етапі (prodazhi/materialy/vyrobnytstvo) з фільтрами (клієнт, №, сортування). Повертає № замовлення, клієнта, матеріал, план кг, строк, статус покриття, дефіцит.",
        "params": ["stage", "customer_like", "order_id_like", "sort", "limit"],
        "required": ["stage"], "fn": _py_get_orders,
    },
    "get_deficits": {
        "description": "Дефіцити за матеріалом (SKU) із станом постачання: нестача кг, к-ть замовлень, % покриття поставками, активні/чернеткові закупки, найгірший статус, найближча дата прибуття. Без sku — топ за нестачею.",
        "params": ["sku", "limit"], "required": [], "fn": _py_get_deficits,
    },
}


def run_tool(name: str, params: dict[str, Any] | None = None, *, engine: Any = None,
             max_rows: int | None = None) -> dict[str, Any]:
    """Execute a catalogued tool (SQL-over-view or Python-backed). Returns
    ``{tool, rows, columns, error}``."""
    if name in _PY_TOOLS:
        try:
            rows = _PY_TOOLS[name]["fn"](params or {}) or []
        except Exception as exc:  # noqa: BLE001 — tool failure must not crash the assistant
            _logger.info("py-tool %s failed: %s", name, exc)
            return {"tool": name, "rows": [], "columns": [], "error": str(exc)}
        cap = int(max_rows if max_rows is not None else config.threshold("text2sql_max_rows", 200))
        cols = list(rows[0].keys()) if rows else []
        return {"tool": name, "rows": rows[:cap], "columns": cols, "error": None}

    spec = TOOLS.get(name)
    if spec is None:
        return {"tool": name, "rows": [], "columns": [], "error": f"невідомий інструмент: {name}"}

    from sqlalchemy import text

    from src.assistant.data.engine import read_engine

    max_rows = int(max_rows if max_rows is not None else config.threshold("text2sql_max_rows", 200))
    bind = {p: (params or {}).get(p) for p in spec.params}
    eng = engine or read_engine()
    ms = int(config.threshold("statement_timeout_ms", 5000))
    try:
        with eng.connect() as conn:
            conn.exec_driver_sql(f"SET statement_timeout = {ms}")
            result = conn.execute(text(spec.sql), bind)
            cols = list(result.keys())
            rows = [dict(r._mapping) for r in result.fetchmany(max_rows)]
        return {"tool": name, "rows": rows, "columns": cols, "error": None}
    except Exception as exc:  # noqa: BLE001 — tool failure must not crash the assistant
        _logger.info("tool %s failed: %s", name, exc)
        return {"tool": name, "rows": [], "columns": [], "error": str(exc)}


__all__ = ["ToolSpec", "TOOLS", "list_tools", "tool_catalogue_text", "tool_schemas", "run_tool"]
