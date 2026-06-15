"""Separate SQLAlchemy engine bound to the read-only ``aps_assistant_ro`` role.

Kept apart from the main app engine so the assistant's ad-hoc reads never touch
the app's connection pool and can carry their own (small) pool + hard guards.
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.assistant import config
from src.utils.config_loader import get_assistant_ro_db_url
from src.utils.logger import setup_logger

_logger = setup_logger(__name__)
_RO_ENGINE: Engine | None = None
_MAIN_READ_ENGINE: Engine | None = None
_RO_OK: bool | None = None


def get_ro_engine() -> Engine:
    """Return a cached read-only engine (created lazily)."""
    global _RO_ENGINE
    if _RO_ENGINE is None:
        from sqlalchemy import create_engine

        ms = int(config.threshold("statement_timeout_ms", 5000))
        _RO_ENGINE = create_engine(
            get_assistant_ro_db_url(),
            pool_size=3,
            max_overflow=2,
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={
                "options": f"-c statement_timeout={ms} -c default_transaction_read_only=on"
            },
        )
    return _RO_ENGINE


def ro_available(force: bool = False) -> bool:
    """True if the read-only role can connect (cached; degrade gracefully)."""
    global _RO_OK
    if _RO_OK is None or force:
        try:
            with get_ro_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
            _RO_OK = True
        except Exception as exc:  # noqa: BLE001
            _logger.info("assistant read-only engine unavailable: %s", exc)
            _RO_OK = False
    return _RO_OK


def _main_read_engine() -> Engine:
    global _MAIN_READ_ENGINE
    if _MAIN_READ_ENGINE is None:
        from src.utils.db_connection import get_engine

        _MAIN_READ_ENGINE = get_engine()
    return _MAIN_READ_ENGINE


def read_engine() -> Engine:
    """Engine for the audited SELECT-only tool library.

    Prefers the read-only role; falls back to the main engine when the role is
    not provisioned. Safe because the tool SQLs are fixed, audited SELECTs (the
    AST guard proves it in tests). Arbitrary text2SQL never uses this — it
    strictly requires the RO role.
    """
    return get_ro_engine() if ro_available() else _main_read_engine()


__all__ = ["get_ro_engine", "ro_available", "read_engine"]
