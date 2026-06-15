"""PostgreSQL connection management."""

import psycopg2
import psycopg2.extensions
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.utils.config_loader import get_db_url, load_config
from src.utils.logger import setup_logger

_logger = setup_logger(__name__)


def get_engine() -> Engine:
    """Create SQLAlchemy engine with connection pool.

    Pool sizing rationale (Fix 5 of deficit-drawer hang batch, 2026-05-27):
    The drawer-redesign workflow runs ≥4 concurrent SQL roundtrips per
    user click (fetch_drawer_context parallel reads) on top of the
    /workflow/counters 5s polling loop and HTMX evidence lazy-loads. The
    old 5+10 pool would exhaust under a single operator with 2 tabs open,
    causing requests to queue and appear to "hang". 15+20 = 35 conns max,
    well under default Postgres ``max_connections=100`` with room for
    parallel ETL / training jobs.

    ``pool_pre_ping=True`` guards against stale Windows TCP sockets that
    were silently disconnected by the network stack (psycopg2 only learns
    this on next query, raising obscure ``OperationalError``).
    ``pool_recycle=3600`` proactively cycles each conn every hour to keep
    them fresh.

    Returns:
        SQLAlchemy Engine connected to aps_printing.
    """
    url = get_db_url()
    return create_engine(
        url,
        pool_size=15,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
    )


def get_raw_connection() -> psycopg2.extensions.connection:
    """Create a raw psycopg2 connection for COPY and bulk operations.

    Returns:
        psycopg2 connection object.
    """
    cfg = load_config("db")
    return psycopg2.connect(
        host=cfg["host"],
        port=int(cfg["port"]),
        dbname=cfg["database"],
        user=cfg["user"],
        password=cfg["password"],
    )


def ensure_schema(engine: Engine) -> None:
    """Create the aps schema if it does not exist.

    Args:
        engine: SQLAlchemy Engine instance.
    """
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS aps"))
        conn.commit()
    _logger.info("Schema 'aps' ensured.")


def test_connection() -> bool:
    """Test database connectivity by executing SELECT 1.

    Returns:
        True if connection succeeded, False otherwise.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        _logger.info("Database connection OK.")
        return True
    except Exception as exc:
        _logger.error("Database connection failed: %s", exc)
        return False


def execute_sql_file(engine: Engine, filepath: str) -> None:
    """Execute a .sql file against the database.

    Args:
        engine: SQLAlchemy Engine instance.
        filepath: Absolute or relative path to the .sql file.
    """
    with open(filepath, encoding="utf-8") as f:
        sql = f.read()

    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    _logger.info("Executed SQL file: %s", filepath)
