"""DuckDB analytics repository.

Uses a class-level shared connection so the DuckDB file is opened once per
process rather than once per query. DuckDB operates in auto-commit mode by
default — each execute() is committed immediately without explicit conn.commit().

Thread safety: DuckDB's Python API serialises concurrent execute() calls on
a single connection internally. All callers here are sequential (EOD job),
so no additional locking is needed.
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path

import duckdb
import pandas as pd

from bot.core.error_logger import log_exception

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "duckdb" / "schema.sql"
DB_PATH      = Path(__file__).resolve().parent.parent / "duckdb" / "analytics.duckdb"

_log = logging.getLogger("tradegenie.analytics")


class AnalyticsRepository:
    # Class-level shared connection — opened once, reused across all instances.
    _shared_conn: "duckdb.DuckDBPyConnection | None" = None
    _shared_conn_path: str = ""

    def __init__(self) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Connection management ──────────────────────────────────────────────────

    def _get_conn(self) -> "duckdb.DuckDBPyConnection":
        """Return the shared DuckDB connection, opening it if necessary.

        Re-opens when DB_PATH changes (e.g. tests that monkeypatch DB_PATH).
        """
        path = str(DB_PATH)
        if (
            AnalyticsRepository._shared_conn is None
            or AnalyticsRepository._shared_conn_path != path
        ):
            if AnalyticsRepository._shared_conn is not None:
                try:
                    AnalyticsRepository._shared_conn.close()
                except Exception:
                    pass
            AnalyticsRepository._shared_conn = duckdb.connect(path)
            AnalyticsRepository._shared_conn_path = path
        return AnalyticsRepository._shared_conn

    @classmethod
    def _reset_connection(cls) -> None:
        """Close and discard the shared connection. Tests only."""
        if cls._shared_conn is not None:
            try:
                cls._shared_conn.close()
            except Exception:
                pass
            cls._shared_conn = None
            cls._shared_conn_path = ""

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        conn = self._get_conn()
        schema = _SCHEMA_PATH.read_text()
        for stmt in schema.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)

    # ── Price history ───────────────────────────────────────────────────────────

    def save_price_history(self, df: pd.DataFrame, symbol: str) -> bool:
        """Bulk-upsert OHLCV rows for symbol. Uses DuckDB's native DataFrame scan."""
        try:
            conn = self._get_conn()
            # Add symbol column so the SELECT matches the table schema
            df = df.copy()
            df.insert(0, "symbol", symbol)
            # DuckDB Python scans the local namespace for DataFrames referenced in SQL
            conn.execute(
                "INSERT OR REPLACE INTO price_history "
                "(symbol, date, open, high, low, close, volume) "
                "SELECT symbol, date, "
                "CAST(open AS DOUBLE), CAST(high AS DOUBLE), "
                "CAST(low  AS DOUBLE), CAST(close AS DOUBLE), "
                "CAST(volume AS BIGINT) FROM df"
            )
            return True
        except Exception as exc:
            log_exception(_log, "save_price_history", exc, {"symbol": symbol})
            return False

    def load_price_history(self, symbol: str, days: int = 365) -> pd.DataFrame:
        cutoff = datetime.date.today() - datetime.timedelta(days=days)
        try:
            return self._get_conn().execute(
                "SELECT date, open, high, low, close, volume FROM price_history "
                "WHERE symbol = ? AND date >= ? ORDER BY date ASC",
                [symbol, cutoff],
            ).df()
        except Exception as exc:
            log_exception(_log, "load_price_history", exc, {"symbol": symbol, "days": days})
            return pd.DataFrame()

    # ── Portfolio snapshots ─────────────────────────────────────────────────────

    def save_portfolio_snapshot(self, snapshot: dict) -> bool:
        try:
            self._get_conn().execute(
                "INSERT OR REPLACE INTO portfolio_snapshots "
                "(snapshot_date, portfolio_value, cash_balance, health_score, "
                "max_drawdown, sharpe_ratio) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    snapshot["snapshot_date"],
                    snapshot["portfolio_value"],
                    snapshot["cash_balance"],
                    snapshot["health_score"],
                    snapshot.get("max_drawdown", 0.0),
                    snapshot.get("sharpe_ratio", 0.0),
                ],
            )
            return True
        except Exception as exc:
            log_exception(_log, "save_portfolio_snapshot", exc,
                          {"date": str(snapshot.get("snapshot_date"))})
            return False

    def load_snapshots(self, days: int = 365) -> pd.DataFrame:
        cutoff = datetime.date.today() - datetime.timedelta(days=days)
        try:
            return self._get_conn().execute(
                "SELECT * FROM portfolio_snapshots "
                "WHERE snapshot_date >= ? ORDER BY snapshot_date ASC",
                [cutoff],
            ).df()
        except Exception as exc:
            log_exception(_log, "load_snapshots", exc, {"days": days})
            return pd.DataFrame()

    # ── Recommendations ─────────────────────────────────────────────────────────

    def save_recommendation(
        self,
        symbol: str,
        recommendation: str,
        confidence: float,
        price: float | None = None,
        change_reason: str | None = None,
    ) -> bool:
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT recommendation FROM recommendation_history "
                "WHERE symbol = ? ORDER BY prediction_date DESC LIMIT 1",
                [symbol],
            ).fetchone()
            prev: str | None = row[0] if row else None

            if prev and prev != recommendation and not change_reason:
                change_reason = f"Changed from {prev} to {recommendation}"

            conn.execute(
                "INSERT OR REPLACE INTO recommendation_history "
                "(symbol, prediction_date, recommendation, confidence, "
                "prev_recommendation, change_reason, price_at_recommendation, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                [
                    symbol,
                    datetime.date.today(),
                    recommendation,
                    confidence,
                    prev,
                    change_reason,
                    price,
                ],
            )
            if prev and prev != recommendation:
                _log.info(
                    "Recommendation change: %s %s → %s confidence=%.0f%%",
                    symbol, prev, recommendation, confidence * 100,
                )
            return True
        except Exception as exc:
            log_exception(_log, "save_recommendation", exc,
                          {"symbol": symbol, "recommendation": recommendation})
            return False

    def load_recommendation_history(
        self,
        symbol: str | None = None,
        days: int = 30,
        changes_only: bool = False,
    ) -> pd.DataFrame:
        """Load recommendation history. symbol=None returns all symbols."""
        try:
            where_clauses = ["prediction_date >= CURRENT_DATE - ?"]
            params: list = [days]

            if symbol:
                where_clauses.append("symbol = ?")
                params.append(symbol)

            if changes_only:
                where_clauses.append(
                    "prev_recommendation IS NOT NULL "
                    "AND prev_recommendation != recommendation"
                )

            where = " AND ".join(where_clauses)
            return self._get_conn().execute(
                f"SELECT symbol, prediction_date, recommendation, "
                f"prev_recommendation, confidence, change_reason, "
                f"price_at_recommendation, actual_return, resolved "
                f"FROM recommendation_history WHERE {where} "
                f"ORDER BY prediction_date DESC, symbol ASC",
                params,
            ).df()
        except Exception as exc:
            log_exception(_log, "load_recommendation_history", exc,
                          {"symbol": symbol, "days": days})
            return pd.DataFrame()
