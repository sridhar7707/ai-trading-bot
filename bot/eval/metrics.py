"""Pure metric functions for investment engine evaluation.

All functions operate on pandas Series/DataFrames — no DB access here.
"""
from __future__ import annotations
import math

import pandas as pd


def win_rate(pnl: pd.Series) -> float:
    """Fraction of trades with positive return."""
    if pnl.empty:
        return 0.0
    return float((pnl > 0).sum() / len(pnl))


def avg_return(pnl: pd.Series) -> float:
    """Mean return per completed trade."""
    return float(pnl.mean()) if not pnl.empty else 0.0


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough decline as a negative fraction.

    Returns 0.0 when equity has fewer than 2 points.
    """
    if equity.empty or len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak.where(peak > 0, other=1.0)
    return float(dd.min())


def sharpe_ratio(pnl: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio on per-trade returns (risk-free = 0)."""
    if pnl.empty:
        return 0.0
    std = pnl.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(pnl.mean() / std * math.sqrt(periods_per_year))


def sortino_ratio(pnl: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sortino ratio — downside deviation only."""
    if pnl.empty:
        return 0.0
    downside = pnl[pnl < 0]
    if downside.empty or downside.std(ddof=1) == 0:
        return float("inf") if pnl.mean() > 0 else 0.0
    return float(pnl.mean() / downside.std(ddof=1) * math.sqrt(periods_per_year))


def profit_factor(pnl: pd.Series) -> float:
    """Gross profit / gross loss. Returns inf when there are no losses."""
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss   = float(abs(pnl[pnl < 0].sum()))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def avg_holding_period(holding_days: pd.Series) -> float:
    """Mean days between buy and sell."""
    return float(holding_days.mean()) if not holding_days.empty else 0.0


def alpha_vs_spy(
    sell_dates: pd.Series,
    pnl: pd.Series,
    spy_daily_returns: pd.Series,
) -> float:
    """Annualized alpha: mean trade daily-equiv return minus SPY mean daily return.

    sell_dates: timestamps of trade close events
    pnl: per-trade pnl_pct (same length as sell_dates)
    spy_daily_returns: Series[date → daily_return] from yfinance
    Returns an annualized percentage difference (e.g. 0.05 = +5%).
    """
    if pnl.empty or spy_daily_returns.empty:
        return 0.0
    dates = pd.to_datetime(sell_dates).dt.date
    spy_aligned = pd.Series(
        [float(spy_daily_returns.get(d, 0.0)) for d in dates],
        index=pnl.index,
    )
    return float((pnl.mean() - spy_aligned.mean()) * 252)


def precision_recall(
    scores: pd.Series,
    pnl: pd.Series,
    score_threshold: float,
    profit_threshold: float = 0.0,
) -> dict:
    """Per-component precision, recall, F1.

    Precision: of trades where score >= threshold, fraction with pnl > profit_threshold.
    Recall: of profitable trades, fraction where score >= threshold.
    """
    if pnl.empty:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_above_threshold": 0, "n_profitable": 0, "tp": 0, "fp": 0, "fn": 0}

    predicted = scores >= score_threshold
    actual    = pnl > profit_threshold

    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "precision":          round(prec, 3),
        "recall":             round(rec, 3),
        "f1":                 round(f1, 3),
        "n_above_threshold":  int(predicted.sum()),
        "n_profitable":       int(actual.sum()),
        "tp": tp, "fp": fp, "fn": fn,
    }


def calibration_buckets(
    scores: pd.Series,
    pnl: pd.Series,
    bucket_edges: list[float],
) -> list[dict]:
    """Win rate and average return for each score bucket.

    Returns a list of dicts: {range, count, win_rate, avg_return, avg_score}
    """
    out = []
    for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
        mask   = (scores >= lo) & (scores < hi)
        subset = pnl[mask]
        out.append({
            "range":      f"{lo:.2f}–{hi:.2f}",
            "count":      int(mask.sum()),
            "win_rate":   round(win_rate(subset), 3),
            "avg_return": round(avg_return(subset), 4),
            "avg_score":  round(float(scores[mask].mean()), 3) if mask.any() else 0.0,
        })
    return out


def summary(
    pnl: pd.Series,
    holding_days: pd.Series,
    equity: pd.Series | None = None,
) -> dict:
    """All standard metrics in one dict."""
    dd = max_drawdown(equity) if equity is not None and len(equity) >= 2 else 0.0
    return {
        "n_trades":         len(pnl),
        "win_rate":         round(win_rate(pnl), 4),
        "avg_return":       round(avg_return(pnl), 5),
        "max_drawdown":     round(dd, 4),
        "sharpe":           round(sharpe_ratio(pnl), 3),
        "sortino":          round(sortino_ratio(pnl), 3),
        "profit_factor":    round(profit_factor(pnl), 3),
        "avg_holding_days": round(avg_holding_period(holding_days), 1),
    }
