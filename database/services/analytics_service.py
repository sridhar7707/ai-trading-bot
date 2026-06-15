import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

from bot.core.error_logger import log_exception
from database.repositories.analytics_repository import AnalyticsRepository

_log = logging.getLogger("tradegenie.analytics_service")


class AnalyticsService:
    def __init__(self, repo: Optional[AnalyticsRepository] = None):
        self._repo = repo if repo is not None else AnalyticsRepository()

    def get_sharpe_ratio(self, portfolio_returns: pd.Series,
                         risk_free_rate: float = 0.05) -> float:
        if len(portfolio_returns) < 2:
            return 0.0
        try:
            excess = portfolio_returns - risk_free_rate / 252
            std = float(excess.std())
            if std == 0:
                return 0.0
            return float(excess.mean() / std * np.sqrt(252))
        except Exception as exc:
            log_exception(_log, "get_sharpe_ratio", exc, {
                "returns_len": len(portfolio_returns) if portfolio_returns is not None else 0,
            })
            return 0.0

    def get_max_drawdown(self, portfolio_values: pd.Series) -> float:
        if len(portfolio_values) < 2:
            return 0.0
        try:
            peak = portfolio_values.cummax()
            dd = (portfolio_values - peak) / peak
            return float(dd.min() * 100)
        except Exception as exc:
            log_exception(_log, "get_max_drawdown", exc, {
                "values_len": len(portfolio_values) if portfolio_values is not None else 0,
            })
            return 0.0

    def save_recommendation(self, symbol: str, recommendation: str,
                            confidence: float) -> bool:
        return self._repo.save_recommendation(
            symbol, datetime.date.today(), recommendation, confidence
        )

    def save_daily_snapshot(self, portfolio_data: dict) -> bool:
        try:
            from bot.core.recommendation_engine import get_portfolio_health

            health = get_portfolio_health(portfolio_data)
            health_score = float(health.get("total", 0)) if isinstance(health, dict) else 0.0

            if health_score == 0:
                _log.warning(
                    "save_daily_snapshot: health score returned 0 — may indicate data issue"
                    " | portfolio_keys=%s", list(portfolio_data.keys()),
                )

            snapshot = {
                "snapshot_date":   datetime.date.today(),
                "portfolio_value": float(portfolio_data.get("portfolio_value", 0) or 0),
                "cash_balance":    float(portfolio_data.get("cash", 0) or 0),
                "health_score":    health_score,
                "max_drawdown":    0.0,
                "sharpe_ratio":    0.0,
            }

            result = self._repo.save_portfolio_snapshot(snapshot)

            if result:
                _log.info(
                    "Daily snapshot saved: value=%.2f health=%s",
                    snapshot["portfolio_value"], snapshot["health_score"],
                )
            else:
                _log.warning(
                    "save_daily_snapshot: repository returned False — snapshot not saved"
                )

            return result

        except Exception as exc:
            log_exception(_log, "save_daily_snapshot", exc, {
                "portfolio_keys": list(portfolio_data.keys()) if portfolio_data else [],
                "date": str(datetime.date.today()),
            })
            return False

    def check_health(self) -> dict:
        """Verify analytics service is working. Returns dict with status per component."""
        results: dict = {}

        # Check DuckDB connection via snapshot load
        try:
            snapshots = self._repo.load_snapshots(days=1)
            results["duckdb_connection"] = "ok"
            results["snapshots_today"]   = len(snapshots)
        except Exception as exc:
            log_exception(_log, "check_health.duckdb", exc)
            results["duckdb_connection"] = "error"
            results["snapshots_today"]   = 0

        # Check Sharpe calculation
        try:
            test_returns = pd.Series(np.random.normal(0.001, 0.02, 10))
            sharpe = self.get_sharpe_ratio(test_returns)
            results["sharpe_calculation"] = "ok" if isinstance(sharpe, float) else "error"
        except Exception as exc:
            log_exception(_log, "check_health.sharpe", exc)
            results["sharpe_calculation"] = "error"

        # Check recommendation_history table is accessible
        try:
            conn = self._repo._get_conn()
            conn.execute("SELECT COUNT(*) FROM recommendation_history").fetchone()
            results["recommendation_history"] = "ok"
        except Exception as exc:
            log_exception(_log, "check_health.rec_history", exc)
            results["recommendation_history"] = "error"

        overall = "ok" if "error" not in results.values() else "degraded"
        results["overall"] = overall

        if overall != "ok":
            _log.warning("Analytics service health check: %s — %s", overall, results)
        else:
            _log.info("Analytics service health check: ok")

        return results


analytics_service = AnalyticsService()
