"""SQLite → DuckDB nightly synchronisation.

Copies closed/historical data from the operational SQLite database into the
DuckDB analytics store. Never modifies SQLite. Idempotent — safe to re-run.

Tables synced
─────────────
trades_archive    closed trades (SELL rows) from SQLite trades
signal_archive    full signal_log snapshot — run BEFORE SQLite pruning
portfolio_history raw portfolio_snapshots (distinct from enriched DuckDB portfolio_snapshots)
"""
from __future__ import annotations

import sqlite3

import duckdb
import pandas as pd
from loguru import logger


# ── DDL ───────────────────────────────────────────────────────────────────────

_TRADES_ARCHIVE_DDL = """
CREATE TABLE IF NOT EXISTS trades_archive (
    id              INTEGER PRIMARY KEY,
    timestamp       VARCHAR,
    symbol          VARCHAR,
    action          VARCHAR,
    shares          DOUBLE,
    price           DOUBLE,
    notional        DOUBLE,
    regime          VARCHAR,
    portfolio_value DOUBLE,
    pnl_pct         DOUBLE,
    xgb_prob        DOUBLE,
    lstm_prob       DOUBLE,
    sentiment_score DOUBLE,
    macro_score     DOUBLE,
    ensemble_score  DOUBLE,
    realized_pnl    DOUBLE,
    order_id        VARCHAR,
    holding_days    INTEGER,
    feature_drivers VARCHAR
)
"""

_SIGNAL_ARCHIVE_DDL = """
CREATE TABLE IF NOT EXISTS signal_archive (
    id              INTEGER PRIMARY KEY,
    timestamp       VARCHAR,
    symbol          VARCHAR,
    xgb_prob        DOUBLE,
    lstm_prob       DOUBLE,
    sentiment_score DOUBLE,
    macro_score     DOUBLE,
    ensemble_score  DOUBLE,
    ensemble_action VARCHAR,
    regime          VARCHAR
)
"""

_PORTFOLIO_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS portfolio_history (
    timestamp       VARCHAR PRIMARY KEY,
    portfolio_value DOUBLE,
    available_cash  DOUBLE,
    open_positions  INTEGER
)
"""


def _ensure_tables(duck: duckdb.DuckDBPyConnection) -> None:
    for ddl in (_TRADES_ARCHIVE_DDL, _SIGNAL_ARCHIVE_DDL, _PORTFOLIO_HISTORY_DDL):
        for stmt in ddl.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                duck.execute(stmt)


# ── Per-table sync functions ───────────────────────────────────────────────────

def sync_trades_archive(
    sqlite: sqlite3.Connection,
    duck: duckdb.DuckDBPyConnection,
) -> int:
    """Upsert closed trades (SELL rows) into DuckDB trades_archive."""
    df = pd.read_sql_query(
        "SELECT id, timestamp, symbol, action, shares, price, notional, regime, "
        "portfolio_value, pnl_pct, xgb_prob, lstm_prob, sentiment_score, macro_score, "
        "ensemble_score, realized_pnl, order_id, holding_days, feature_drivers "
        "FROM trades WHERE action LIKE 'SELL%'",
        sqlite,
    )
    if df.empty:
        return 0
    # DuckDB Python scans the local namespace for DataFrames referenced in SQL
    duck.execute("INSERT OR REPLACE INTO trades_archive SELECT * FROM df")
    return len(df)


def sync_signal_archive(
    sqlite: sqlite3.Connection,
    duck: duckdb.DuckDBPyConnection,
) -> int:
    """Copy all signal_log rows to DuckDB signal_archive.

    Call BEFORE the SQLite 30-day pruning so history is preserved.
    """
    df = pd.read_sql_query(
        "SELECT id, timestamp, symbol, xgb_prob, lstm_prob, sentiment_score, "
        "macro_score, ensemble_score, ensemble_action, regime FROM signal_log",
        sqlite,
    )
    if df.empty:
        return 0
    duck.execute("INSERT OR REPLACE INTO signal_archive SELECT * FROM df")
    return len(df)


def sync_portfolio_history(
    sqlite: sqlite3.Connection,
    duck: duckdb.DuckDBPyConnection,
) -> int:
    """Copy portfolio_snapshots to DuckDB portfolio_history."""
    df = pd.read_sql_query(
        "SELECT timestamp, portfolio_value, available_cash, open_positions "
        "FROM portfolio_snapshots",
        sqlite,
    )
    if df.empty:
        return 0
    duck.execute("INSERT OR REPLACE INTO portfolio_history SELECT * FROM df")
    return len(df)


# ── Top-level entry point ──────────────────────────────────────────────────────

def sync_all(
    sqlite: sqlite3.Connection,
    duck: duckdb.DuckDBPyConnection,
) -> dict[str, int]:
    """Run all sync jobs and return {table: rows_synced}. Idempotent."""
    _ensure_tables(duck)
    results: dict[str, int] = {}

    jobs = [
        ("signal_archive",    sync_signal_archive),    # before SQLite pruning
        ("trades_archive",    sync_trades_archive),
        ("portfolio_history", sync_portfolio_history),
    ]
    for table, fn in jobs:
        try:
            results[table] = fn(sqlite, duck)
        except Exception as exc:
            logger.error(f"sync_all [{table}] failed: {exc}")
            results[table] = -1

    return results
