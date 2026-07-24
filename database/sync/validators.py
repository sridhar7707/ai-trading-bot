"""Post-sync data validators.

Each validator returns a list of warning strings.
An empty list means the check passed.
"""
from __future__ import annotations

import sqlite3

import duckdb


def validate_trade_sync(
    sqlite: sqlite3.Connection,
    duck: duckdb.DuckDBPyConnection,
) -> list[str]:
    """SELL row count must match between SQLite trades and DuckDB trades_archive."""
    warnings: list[str] = []
    try:
        sq = sqlite.execute(
            "SELECT COUNT(*) FROM trades WHERE action LIKE 'SELL%'"
        ).fetchone()[0]
        dq = duck.execute("SELECT COUNT(*) FROM trades_archive").fetchone()[0]
        if sq != dq:
            warnings.append(
                f"trades_archive mismatch: SQLite has {sq} SELL rows, "
                f"DuckDB has {dq}"
            )
    except Exception as exc:
        warnings.append(f"validate_trade_sync error: {exc}")
    return warnings


def validate_portfolio_sync(
    sqlite: sqlite3.Connection,
    duck: duckdb.DuckDBPyConnection,
) -> list[str]:
    """portfolio_history must have rows if SQLite portfolio_snapshots does."""
    warnings: list[str] = []
    try:
        sq = sqlite.execute("SELECT COUNT(*) FROM portfolio_snapshots").fetchone()[0]
        dq = duck.execute("SELECT COUNT(*) FROM portfolio_history").fetchone()[0]
        if sq > 0 and dq == 0:
            warnings.append(
                "portfolio_history is empty after sync "
                f"(SQLite has {sq} snapshots)"
            )
    except Exception as exc:
        warnings.append(f"validate_portfolio_sync error: {exc}")
    return warnings


def validate_sync(
    sqlite: sqlite3.Connection,
    duck: duckdb.DuckDBPyConnection,
) -> list[str]:
    """Run all validators. Returns combined warning list."""
    return validate_trade_sync(sqlite, duck) + validate_portfolio_sync(sqlite, duck)
