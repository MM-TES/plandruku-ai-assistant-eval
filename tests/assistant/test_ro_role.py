"""Phase 3 gate: read-only role enforcement (integration — skips without DB/role)."""
from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.assistant.data.engine import get_ro_engine, ro_available


@pytest.fixture(autouse=True)
def _require_ro_role() -> None:
    if not ro_available():
        pytest.skip("aps_assistant_ro role / DB not available")


def test_select_works() -> None:
    with get_ro_engine().connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1


def test_write_is_blocked() -> None:
    with pytest.raises(SQLAlchemyError):
        with get_ro_engine().connect() as conn:
            conn.execute(text("CREATE TEMP TABLE _assistant_probe (x int)"))
            conn.execute(text("INSERT INTO _assistant_probe VALUES (1)"))


def test_statement_timeout_enforced() -> None:
    with pytest.raises(SQLAlchemyError):
        with get_ro_engine().connect() as conn:
            conn.exec_driver_sql("SET statement_timeout = 300")
            conn.execute(text("SELECT pg_sleep(3)"))
