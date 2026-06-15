"""Database migration script. Executes sql/*.sql files in order."""

import sys
from pathlib import Path

# Allow running directly: python src/utils/db_migrate.py
_PROJECT_ROOT_EARLY = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT_EARLY) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_EARLY))

from src.utils.db_connection import get_raw_connection
from src.utils.logger import setup_logger

_logger = setup_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SQL_DIR = _PROJECT_ROOT / "sql"

# Files to skip (manual steps or documentation only)
_SKIP_FILES = {"000_create_database.sql"}


def run_migrations() -> None:
    """Execute all sql/*.sql files in sorted order.

    Skips 000_create_database.sql (must be run manually in pgAdmin).
    Uses psycopg2 directly to support multi-statement SQL files.

    Raises:
        SystemExit: If any SQL file fails to execute.
    """
    sql_files = sorted(
        f for f in _SQL_DIR.glob("*.sql")
        if f.name not in _SKIP_FILES
    )

    if not sql_files:
        _logger.warning("No SQL files found in %s", _SQL_DIR)
        return

    conn = get_raw_connection()
    conn.autocommit = False

    try:
        for sql_file in sql_files:
            _logger.info("Executing: %s", sql_file.name)
            sql = sql_file.read_text(encoding="utf-8")

            with conn.cursor() as cur:
                cur.execute(sql)

            conn.commit()
            _logger.info("OK: %s", sql_file.name)

    except Exception as exc:
        conn.rollback()
        _logger.error("Migration failed on %s: %s", sql_file.name, exc)
        conn.close()
        sys.exit(1)

    finally:
        conn.close()

    _logger.info("All migrations completed successfully.")


if __name__ == "__main__":
    run_migrations()
