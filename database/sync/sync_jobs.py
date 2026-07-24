"""EOD sync job runner.

Called from bot/_main_runner.end_of_day_summary() after market close.
Opens its own connections so it doesn't depend on the AnalyticsRepository
singleton or any open SQLite connection.
"""
from __future__ import annotations

import sqlite3

import duckdb
from loguru import logger

from config import TRADE_DB_PATH
from database.repositories.analytics_repository import DB_PATH as ANALYTICS_DB_PATH
from database.sync.sqlite_to_duckdb import sync_all
from database.sync.validators import validate_sync


def run_nightly_sync(
    sqlite_path: str = TRADE_DB_PATH,
    duck_path: str | None = None,
) -> dict[str, int]:
    """Run all SQLite → DuckDB sync jobs.

    Returns {table: rows_synced}. Negative value means that table failed.
    Logs warnings for any validation failures but never raises — a sync
    failure must not block the EOD summary from completing.
    """
    duck_path = duck_path or str(ANALYTICS_DB_PATH)
    sqlite: sqlite3.Connection | None = None
    duck: duckdb.DuckDBPyConnection | None = None
    results: dict[str, int] = {}

    try:
        sqlite = sqlite3.connect(sqlite_path)
        duck = duckdb.connect(duck_path)

        results = sync_all(sqlite, duck)

        warnings = validate_sync(sqlite, duck)
        for w in warnings:
            logger.warning(f"Sync validation: {w}")

        logger.info(f"Nightly DuckDB sync complete: {results}")

    except Exception as exc:
        logger.error(f"run_nightly_sync failed: {exc}")

    finally:
        for conn in (sqlite, duck):
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    return results
