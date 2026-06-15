import datetime
import logging
from pathlib import Path

import duckdb
import pandas as pd

from bot.core.error_logger import log_exception

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "duckdb" / "schema.sql"
DB_PATH = Path(__file__).resolve().parent.parent / "duckdb" / "analytics.duckdb"

_log = logging.getLogger("tradegenie.analytics")


class AnalyticsRepository:
    def __init__(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_conn(self):
        return duckdb.connect(str(DB_PATH))

    def _init_schema(self):
        schema = _SCHEMA_PATH.read_text()
        with self._get_conn() as conn:
            for stmt in schema.split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)

    def save_price_history(self, df: pd.DataFrame, symbol: str) -> bool:
        try:
            with self._get_conn() as conn:
                for _, row in df.iterrows():
                    conn.execute(
                        "INSERT OR REPLACE INTO price_history "
                        "(symbol, date, open, high, low, close, volume) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [symbol, row["date"], float(row.get("open", 0)),
                         float(row.get("high", 0)), float(row.get("low", 0)),
                         float(row["close"]), int(row.get("volume", 0))],
                    )
            return True
        except Exception as exc:
            log_exception(_log, "save_price_history", exc, {"symbol": symbol})
            return False

    def load_price_history(self, symbol: str, days: int = 365) -> pd.DataFrame:
        cutoff = datetime.date.today() - datetime.timedelta(days=days)
        try:
            with self._get_conn() as conn:
                return conn.execute(
                    "SELECT date, open, high, low, close, volume FROM price_history "
                    "WHERE symbol = ? AND date >= ? ORDER BY date ASC",
                    [symbol, cutoff],
                ).df()
        except Exception as exc:
            log_exception(_log, "load_price_history", exc, {"symbol": symbol, "days": days})
            return pd.DataFrame()

    def save_portfolio_snapshot(self, snapshot: dict) -> bool:
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO portfolio_snapshots "
                    "(snapshot_date, portfolio_value, cash_balance, health_score, "
                    "max_drawdown, sharpe_ratio) VALUES (?, ?, ?, ?, ?, ?)",
                    [snapshot["snapshot_date"], snapshot["portfolio_value"],
                     snapshot["cash_balance"], snapshot["health_score"],
                     snapshot.get("max_drawdown", 0.0), snapshot.get("sharpe_ratio", 0.0)],
                )
            return True
        except Exception as exc:
            log_exception(_log, "save_portfolio_snapshot", exc, {
                "date": str(snapshot.get("snapshot_date")),
            })
            return False

    def load_snapshots(self, days: int = 365) -> pd.DataFrame:
        cutoff = datetime.date.today() - datetime.timedelta(days=days)
        try:
            with self._get_conn() as conn:
                return conn.execute(
                    "SELECT * FROM portfolio_snapshots "
                    "WHERE snapshot_date >= ? ORDER BY snapshot_date ASC",
                    [cutoff],
                ).df()
        except Exception as exc:
            log_exception(_log, "load_snapshots", exc, {"days": days})
            return pd.DataFrame()

    def save_recommendation(self, symbol: str, recommendation: str,
                            confidence: float, price: float = None,
                            change_reason: str = None) -> bool:
        try:
            # Detect previous recommendation for this symbol
            prev: str | None = None
            try:
                with self._get_conn() as conn:
                    row = conn.execute(
                        "SELECT recommendation FROM recommendation_history "
                        "WHERE symbol = ? ORDER BY prediction_date DESC LIMIT 1",
                        [symbol],
                    ).fetchone()
                    if row:
                        prev = row[0]
            except Exception as exc:
                log_exception(_log, "save_recommendation.get_prev", exc, {"symbol": symbol})

            if prev and prev != recommendation and not change_reason:
                change_reason = f"Changed from {prev} to {recommendation}"

            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO recommendation_history "
                    "(symbol, prediction_date, recommendation, confidence, "
                    "prev_recommendation, change_reason, price_at_recommendation, "
                    "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                    [symbol, datetime.date.today(), recommendation, confidence,
                     prev, change_reason, price],
                )

            if prev and prev != recommendation:
                _log.info(
                    "Recommendation change: %s %s → %s confidence=%.0f%%",
                    symbol, prev, recommendation, confidence * 100,
                )
            return True

        except Exception as exc:
            log_exception(_log, "save_recommendation", exc, {
                "symbol": symbol, "recommendation": recommendation,
            })
            return False

    def load_recommendation_history(self, symbol: str = None, days: int = 30,
                                    changes_only: bool = False) -> pd.DataFrame:
        """Load recommendation history; symbol=None returns all symbols."""
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
            with self._get_conn() as conn:
                return conn.execute(
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
