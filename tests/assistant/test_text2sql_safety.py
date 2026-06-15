"""Phase 3 critical gate: text2SQL AST safety — 0 DML, 100% LIMIT enforcement."""
from __future__ import annotations

import pytest

from src.assistant.data.text2sql import validate_sql

# Every one of these MUST be rejected before any execution.
MALICIOUS = [
    "DROP TABLE aps.orders",
    "DROP TABLE aps.orders;",
    'UPDATE aps.orders SET "28_plan_kg" = 0',
    "INSERT INTO aps.proposed_actions(x) VALUES (1)",
    "DELETE FROM aps.orders",
    "TRUNCATE aps.orders",
    "ALTER TABLE aps.orders ADD COLUMN hack int",
    "CREATE TABLE aps.evil (id int)",
    "GRANT SELECT ON aps.orders TO postgres",
    "SELECT 1; DELETE FROM aps.orders",
    "SELECT 1; DROP TABLE aps.orders;",
    "WITH x AS (DELETE FROM aps.orders RETURNING *) SELECT * FROM x",
    "WITH x AS (UPDATE aps.orders SET \"28_plan_kg\"=0 RETURNING *) SELECT * FROM x",
    "SELECT * FROM pg_catalog.pg_user",
    "SELECT * FROM information_schema.tables",
    "COPY aps.orders TO '/tmp/x.csv'",
]

VALID = [
    "SELECT count(*) FROM aps.v_pending_orders",
    "SELECT * FROM aps.v_order_risk WHERE total_deficit_kg > 0 ORDER BY total_deficit_kg DESC",
    "SELECT readiness, count(*) FROM aps.v_material_readiness GROUP BY readiness",
]


@pytest.mark.parametrize("sql", MALICIOUS)
def test_malicious_sql_rejected(sql: str) -> None:
    v = validate_sql(sql, max_rows=200)
    assert v.ok is False, f"SHOULD REJECT: {sql}"
    assert v.sql is None


@pytest.mark.parametrize("sql", VALID)
def test_valid_select_accepted(sql: str) -> None:
    v = validate_sql(sql, max_rows=200)
    assert v.ok is True, f"should accept: {sql} ({v.error})"


@pytest.mark.parametrize("sql", VALID)
def test_limit_is_always_injected(sql: str) -> None:
    v = validate_sql(sql, max_rows=200)
    assert v.sql is not None
    assert "LIMIT" in v.sql.upper()


def test_existing_over_limit_is_clamped() -> None:
    v = validate_sql("SELECT * FROM aps.orders LIMIT 99999", max_rows=200)
    assert v.ok and v.sql is not None
    assert "200" in v.sql and "99999" not in v.sql


def test_empty_rejected() -> None:
    assert validate_sql("   ", max_rows=200).ok is False
