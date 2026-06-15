"""Guarded text2SQL: generate → AST-validate → execute (read-only) → self-correct.

Safety is layered (defence in depth):
  1. read-only DB role (sql/068) — cannot write even if everything else fails;
  2. AST validation (sqlglot) — exactly one statement, SELECT-only, no DDL/DML
     anywhere (incl. CTEs), schema restricted to ``aps``;
  3. auto-LIMIT injected via AST (not string concat);
  4. statement_timeout per connection + at the role level;
  5. bounded self-correction loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import sqlglot
from sqlglot import exp

from src.assistant import config
from src.assistant.llm import LLMUsage, call_llm
from src.assistant.tracing import parse_json_object, traceable
from src.utils.logger import setup_logger

_logger = setup_logger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA_CARD = _ROOT / "R_and_D" / "assistant_rnd" / "schema_card.md"
_ALLOWED_SCHEMAS = {"aps"}

# Query-shaped top-level nodes that are allowed.
_QUERY_TYPES = tuple(
    t for t in (
        getattr(exp, n, None)
        for n in ("Select", "Union", "Except", "Intersect", "Subquery")
    ) if t is not None
)
# Any of these appearing ANYWHERE in the tree → reject (DDL / DML / admin).
_FORBIDDEN_TYPES = tuple(
    t for t in (
        getattr(exp, n, None)
        for n in (
            "Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable",
            "TruncateTable", "Truncate", "Grant", "Command", "Copy", "Merge",
            "Set", "Use",
        )
    ) if t is not None
)


@dataclass
class Validation:
    ok: bool
    error: Optional[str] = None
    sql: Optional[str] = None  # sanitized (LIMIT-injected) SQL when ok


def validate_sql(sql: str, *, max_rows: int | None = None) -> Validation:
    """Validate + sanitize a candidate query. Pure (no DB, no network)."""
    max_rows = int(max_rows if max_rows is not None else config.threshold("text2sql_max_rows", 200))
    raw = (sql or "").strip().rstrip(";").strip()
    if not raw:
        return Validation(False, "порожній запит")

    try:
        statements = [s for s in sqlglot.parse(raw, read="postgres") if s is not None]
    except Exception as exc:  # noqa: BLE001
        return Validation(False, f"не вдалося розпарсити SQL: {exc}")

    if len(statements) != 1:
        return Validation(False, "дозволено рівно один statement (без ';' chains)")

    stmt = statements[0]
    if not isinstance(stmt, _QUERY_TYPES):
        return Validation(False, "дозволено лише SELECT-запити")

    for forbidden in _FORBIDDEN_TYPES:
        if stmt.find(forbidden) is not None:
            return Validation(False, f"заборонена операція: {forbidden.__name__}")

    for table in stmt.find_all(exp.Table):
        schema = (table.db or "").lower()
        if schema and schema not in _ALLOWED_SCHEMAS:
            return Validation(False, f"дозволена лише схема aps (знайдено '{schema}')")

    try:
        capped = _enforce_limit(stmt, max_rows)
        return Validation(True, None, capped.sql(dialect="postgres"))
    except Exception as exc:  # noqa: BLE001 — fall back to a hard subquery cap
        _logger.warning("limit-injection failed (%s) — wrapping in capped subquery", exc)
        return Validation(True, None, f"SELECT * FROM ({raw}) AS _capped LIMIT {max_rows}")


def _enforce_limit(stmt: exp.Expression, max_rows: int) -> exp.Expression:
    limit = stmt.args.get("limit")
    if limit is None:
        return stmt.limit(max_rows)
    try:
        current = int(limit.expression.name)
        if current > max_rows:
            return stmt.limit(max_rows)
    except Exception:  # noqa: BLE001
        return stmt.limit(max_rows)
    return stmt


def _load_schema_card() -> str:
    try:
        return _SCHEMA_CARD.read_text(encoding="utf-8")
    except OSError:
        return ""


@traceable(name="assistant.text2sql.generate")
def _generate(
    question: str,
    schema_card: str,
    *,
    error: str | None = None,
    prev_sql: str | None = None,
    usage: LLMUsage | None = None,
) -> dict[str, Any]:
    user = f"Питання оператора: {question}\n\n[SCHEMA-CARD]\n{schema_card[:8000]}"
    if error and prev_sql:
        user += (
            f"\n\n[ВИПРАВ ПОМИЛКУ] Попередній SQL був невалідний/невиконуваний:\n{prev_sql}\n"
            f"Помилка: {error}\nЗгенеруй виправлений ОДИН SELECT."
        )
    resp = call_llm(
        agent_name="text2sql",
        role_key="answer",
        messages=[
            {"role": "system", "content": config.prompt("text2sql")},
            {"role": "user", "content": user},
        ],
        usage=usage,
        temperature=0.0,
        max_tokens=int(config.threshold("answer_max_tokens", 1024)),
        response_format={"type": "json_object"},
    )
    return parse_json_object(resp.choices[0].message.content or "{}")


def _execute(sql: str, engine: Any, max_rows: int) -> tuple[list[dict], list[str]]:
    from sqlalchemy import text

    from src.assistant.data.engine import get_ro_engine

    eng = engine or get_ro_engine()
    ms = int(config.threshold("statement_timeout_ms", 5000))
    with eng.connect() as conn:
        conn.exec_driver_sql(f"SET statement_timeout = {ms}")
        result = conn.execute(text(sql))
        cols = list(result.keys())
        rows = [dict(r._mapping) for r in result.fetchmany(max_rows)]
    return rows, cols


@traceable(name="assistant.text2sql")
def run_text2sql(
    question: str,
    *,
    usage: LLMUsage | None = None,
    engine: Any = None,
    max_rows: int | None = None,
    retries: int | None = None,
) -> dict[str, Any]:
    """Full guarded pipeline. Returns a result dict (never raises on bad SQL)."""
    max_rows = int(max_rows if max_rows is not None else config.threshold("text2sql_max_rows", 200))
    retries = int(retries if retries is not None else config.threshold("text2sql_retries", 1))
    schema_card = _load_schema_card()

    gen = _generate(question, schema_card, usage=usage)
    sql = gen.get("sql", "")
    last_error: str | None = None

    for attempt in range(retries + 1):
        v = validate_sql(sql, max_rows=max_rows)
        if not v.ok:
            last_error = v.error
            if attempt < retries:
                gen = _generate(question, schema_card, error=last_error, prev_sql=sql, usage=usage)
                sql = gen.get("sql", "")
                continue
            return {"ok": False, "error": last_error, "sql": sql, "rows": [], "columns": []}
        try:
            rows, cols = _execute(v.sql or "", engine, max_rows)
            return {
                "ok": True,
                "sql": v.sql,
                "rationale": gen.get("rationale", ""),
                "rows": rows,
                "columns": cols,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < retries:
                gen = _generate(question, schema_card, error=last_error, prev_sql=v.sql, usage=usage)
                sql = gen.get("sql", "")
                continue
            return {"ok": False, "error": last_error, "sql": v.sql, "rows": [], "columns": []}

    return {"ok": False, "error": last_error or "unknown", "sql": sql, "rows": [], "columns": []}


__all__ = ["Validation", "validate_sql", "run_text2sql"]
