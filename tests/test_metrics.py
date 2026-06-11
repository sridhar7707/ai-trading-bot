"""Tests for backtest/metrics.py."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from backtest.metrics import compute_metrics, _max_drawdown


# --- _max_drawdown ---

def test_max_drawdown_no_drawdown():
    vals = np.array([100.0, 110.0, 120.0, 130.0])
    assert _max_drawdown(vals) == pytest.approx(0.0, abs=1e-4)


def test_max_drawdown_single_drop():
    # From 100 peak to 80 → 20%
    vals = np.array([100.0, 120.0, 80.0])
    dd = _max_drawdown(vals)
    assert abs(dd - (120 - 80) / (120 + 1e-8)) < 1e-4


def test_max_drawdown_full_loss():
    vals = np.array([100.0, 50.0, 1.0])
    dd = _max_drawdown(vals)
    assert dd > 0.98  # nearly 100% drawdown


def test_max_drawdown_recovers_but_falls_further():
    # Peak at 150, valley at 80 → 70/150 = 0.466
    vals = np.array([100.0, 150.0, 100.0, 80.0])
    dd = _max_drawdown(vals)
    assert abs(dd - (150 - 80) / (150 + 1e-8)) < 1e-4


def test_max_drawdown_single_value():
    vals = np.array([10_000.0])
    assert _max_drawdown(vals) == pytest.approx(0.0, abs=1e-6)


def test_max_drawdown_monotonic_increase():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _max_drawdown(vals) == pytest.approx(0.0, abs=1e-6)


# --- compute_metrics ---

def _trade(action: str, pnl_pct: float) -> dict:
    return {"action": action, "pnl_pct": pnl_pct}


def test_compute_metrics_total_return():
    values = [10_000.0, 11_000.0]
    result = compute_metrics(values, [], 10_000.0)
    assert result["total_return"] == pytest.approx(0.10, abs=1e-4)


def test_compute_metrics_final_value():
    values = [10_000.0, 12_000.0]
    result = compute_metrics(values, [], 10_000.0)
    assert result["final_value"] == pytest.approx(12_000.0)


def test_compute_metrics_no_trades():
    values = [10_000.0, 10_500.0, 11_000.0]
    result = compute_metrics(values, [], 10_000.0)
    assert result["num_trades"] == 0
    assert result["win_rate"] == 0.0
    assert result["profit_factor"] == pytest.approx(0.0, abs=1e-6)


def test_compute_metrics_win_rate():
    trades = [
        _trade("SELL", 0.05),
        _trade("SELL", 0.03),
        _trade("SELL", -0.02),
        _trade("SELL", -0.04),
    ]
    values = [10_000.0] * 50  # flat portfolio
    result = compute_metrics(values, trades, 10_000.0)
    assert result["win_rate"] == pytest.approx(0.5, abs=1e-6)


def test_compute_metrics_profit_factor():
    trades = [
        _trade("SELL", 0.10),   # gross profit = 0.10
        _trade("SELL", -0.05),  # gross loss = 0.05
    ]
    values = [10_000.0] * 50
    result = compute_metrics(values, trades, 10_000.0)
    # profit_factor = 0.10 / (0.05 + 1e-8) ≈ 2.0
    assert result["profit_factor"] == pytest.approx(2.0, abs=1e-3)


def test_compute_metrics_expectancy_positive():
    trades = [
        _trade("SELL", 0.10),
        _trade("SELL", 0.08),
        _trade("SELL", -0.03),
    ]
    values = [10_000.0] * 50
    result = compute_metrics(values, trades, 10_000.0)
    assert result["expectancy"] > 0.0


def test_compute_metrics_expectancy_negative():
    trades = [
        _trade("SELL", 0.01),
        _trade("SELL", -0.10),
        _trade("SELL", -0.10),
    ]
    values = [10_000.0] * 50
    result = compute_metrics(values, trades, 10_000.0)
    assert result["expectancy"] < 0.0


def test_compute_metrics_ignores_buy_actions():
    trades = [
        _trade("BUY", 0.0),
        _trade("SELL", 0.05),
    ]
    values = [10_000.0] * 50
    result = compute_metrics(values, trades, 10_000.0)
    # Only SELL counts
    assert result["num_trades"] == 2
    assert result["win_rate"] == pytest.approx(1.0, abs=1e-6)


def test_compute_metrics_all_losses():
    trades = [_trade("SELL", -0.05)] * 5
    values = [10_000.0 - 50 * i for i in range(20)]
    result = compute_metrics(values, trades, 10_000.0)
    assert result["win_rate"] == 0.0
    assert result["avg_win"] == 0.0
    assert result["num_losses"] == 5


def test_compute_metrics_max_drawdown_present():
    # Portfolio goes up then collapses
    values = [10_000.0, 12_000.0, 8_000.0]
    result = compute_metrics(values, [], 10_000.0)
    # Peak 12000 → 8000: dd = 4000/12000 ≈ 0.333
    assert result["max_drawdown"] > 0.0


def test_compute_metrics_returns_all_expected_keys():
    values = [10_000.0, 10_500.0]
    result = compute_metrics(values, [], 10_000.0)
    expected_keys = {
        "total_return", "ann_return", "sharpe", "calmar", "max_drawdown",
        "profit_factor", "win_rate", "expectancy", "avg_win", "avg_loss",
        "num_trades", "num_wins", "num_losses", "final_value",
    }
    assert expected_keys <= set(result.keys())
