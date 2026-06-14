import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

from database.repositories.analytics_repository import AnalyticsRepository

_log = logging.getLogger("tradegenie.analytics_service")


class AnalyticsService:
    def __init__(self, repo: Optional[AnalyticsRepository] = None):
        self._repo = repo if repo is not None else AnalyticsRepository()

    def get_sharpe_ratio(self, portfolio_returns: pd.Series,
                         risk_free_rate: float = 0.05) -> float:
        if len(portfolio_returns) < 2:
            return 0.0
        excess = portfolio_returns - risk_free_rate / 252
        std = float(excess.std())
        if std == 0:
            return 0.0
        return float(excess.mean() / std * np.sqrt(252))

    def get_max_drawdown(self, portfolio_values: pd.Series) -> float:
        if len(portfolio_values) < 2:
            return 0.0
        peak = portfolio_values.cummax()
        dd = (portfolio_values - peak) / peak
        return float(dd.min() * 100)

    def save_recommendation(self, symbol: str, recommendation: str,
                            confidence: float) -> bool:
        return self._repo.save_recommendation(
            symbol, datetime.date.today(), recommendation, confidence
        )

    def save_daily_snapshot(self, portfolio_data: dict) -> bool:
        health_score = 0.0
        try:
            from bot.core.recommendation_engine import get_portfolio_health
            health = get_portfolio_health(portfolio_data)
            health_score = float(health.get("total", 0)) if isinstance(health, dict) else 0.0
        except Exception:
            pass

        snapshot = {
            "snapshot_date": datetime.date.today(),
            "portfolio_value": float(portfolio_data.get("portfolio_value", 0)),
            "cash_balance": float(portfolio_data.get("cash", 0)),
            "health_score": health_score,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }
        return self._repo.save_portfolio_snapshot(snapshot)


analytics_service = AnalyticsService()
