import datetime
import logging
import traceback
from pathlib import Path

import duckdb
import pandas as pd

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
            _log.error(f"save_price_history error: {exc}\n{traceback.format_exc()}")
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
            _log.error(f"load_price_history error: {exc}\n{traceback.format_exc()}")
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
            _log.error(f"save_portfolio_snapshot error: {exc}\n{traceback.format_exc()}")
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
            _log.error(f"load_snapshots error: {exc}\n{traceback.format_exc()}")
            return pd.DataFrame()

    def save_recommendation(self, symbol: str, prediction_date: datetime.date,
                            recommendation: str, confidence: float) -> bool:
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO recommendation_history "
                    "(symbol, prediction_date, recommendation, confidence) "
                    "VALUES (?, ?, ?, ?)",
                    [symbol, prediction_date, recommendation, confidence],
                )
            return True
        except Exception as exc:
            _log.error(f"save_recommendation error: {exc}\n{traceback.format_exc()}")
            return False
