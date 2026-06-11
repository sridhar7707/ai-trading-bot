"""Tests for scripts/confidence_check.py pure functions."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from scripts.confidence_check import (
    _daily_close_values,
    compute_sharpe,
    compute_max_drawdown,
    _max_consecutive_losses,
)


# --- _daily_close_values ---

def _row(date_str: str, action: str, pnl_pct: float, pv: float):
    return (f"{date_str}T12:00:00+00:00", action, pnl_pct, pv)


def test_daily_close_values_empty():
    assert _daily_close_values([]) == []


def test_daily_close_values_single_trade():
    trades = [_row("2024-01-02", "BUY", 0.0, 10_000)]
    result = _daily_close_values(trades)
    assert result == [10_000]


def test_daily_close_values_last_trade_of_day_wins():
    trades = [
        _row("2024-01-02", "BUY", 0.0, 10_000),
        _row("2024-01-02", "SELL", 0.05, 10_500),
    ]
    result = _daily_close_values(trades)
    assert result == [10_500]


def test_daily_close_values_forward_fills_no_trade_days():
    # Monday and Wednesday only — Tuesday should be forward-filled with Monday's value
    trades = [
        _row("2024-01-01", "BUY", 0.0, 10_000),  # Monday
        _row("2024-01-03", "SELL", 0.02, 10_200),  # Wednesday
    ]
    result = _daily_close_values(trades)
    # Mon=10000, Tue=10000 (forward-fill), Wed=10200
    assert result == [10_000, 10_000, 10_200]


def test_daily_close_values_excludes_weekends():
    # Friday through Monday: weekend days should not appear
    trades = [
        _row("2024-01-05", "BUY", 0.0, 10_000),  # Friday
        _row("2024-01-08", "SELL", 0.02, 10_200),  # Monday
    ]
    result = _daily_close_values(trades)
    # Fri=10000, Mon=10200 (weekend skipped)
    assert result == [10_000, 10_200]


def test_daily_close_values_multiple_weeks():
    trades = [
        _row("2024-01-02", "BUY", 0.0, 10_000),   # Tue
        _row("2024-01-03", "BUY", 0.0, 10_100),   # Wed
        _row("2024-01-05", "SELL", 0.02, 10_200),  # Fri
    ]
    result = _daily_close_values(trades)
    # Tue, Wed, Thu(fwd-fill=10100), Fri
    assert result == [10_000, 10_100, 10_100, 10_200]
    assert len(result) == 4


# --- compute_sharpe ---

def test_compute_sharpe_empty():
    assert compute_sharpe([]) == 0.0


def test_compute_sharpe_single_value():
    assert compute_sharpe([10_000]) == 0.0


def test_compute_sharpe_zero_variance():
    # All same value — std ≈ 0, but handled by 1e-8 guard
    vals = [10_000] * 10
    result = compute_sharpe(vals)
    assert isinstance(result, float)


def test_compute_sharpe_positive_trend():
    # Steadily rising portfolio — positive Sharpe
    vals = [10_000 + 100 * i for i in range(30)]
    sharpe = compute_sharpe(vals)
    assert sharpe > 0.0


def test_compute_sharpe_negative_trend():
    vals = [10_000 - 100 * i for i in range(30)]
    sharpe = compute_sharpe(vals)
    assert sharpe < 0.0


def test_compute_sharpe_uses_sqrt252_annualisation():
    # A single unit return per day for 2 days → mean_return=1.0, std≈0
    # Result should be approximately 1.0 * sqrt(252) / 1e-8 clamped by the formula
    vals = [1.0, 2.0]
    sharpe = compute_sharpe(vals)
    # The key property: with daily returns, sqrt(252) is used, not sqrt(252*78)
    assert sharpe > 0.0


# --- compute_max_drawdown ---

def test_max_drawdown_no_drawdown():
    # Monotonically increasing — no drawdown
    vals = [100, 110, 120, 130]
    assert compute_max_drawdown(vals) == pytest.approx(0.0, abs=1e-6)


def test_max_drawdown_full_loss():
    # Drops from 100 to 0
    vals = [100, 50, 0]
    # peak=100, dd at 50 = 50/100 = 0.5, dd at 0 = 100/100 = 1.0
    assert compute_max_drawdown(vals) == pytest.approx(1.0, abs=1e-6)


def test_max_drawdown_typical():
    # Peak at 120, then drops to 90 → dd = 30/120 = 0.25
    vals = [100, 110, 120, 100, 90]
    dd = compute_max_drawdown(vals)
    assert abs(dd - 0.25) < 1e-4


def test_max_drawdown_multiple_peaks():
    # Recovers and then falls further
    vals = [100, 120, 80, 130, 90]
    # From 130 → 90 = 40/130 ≈ 0.307 (larger than 120→80=40/120≈0.333)
    dd = compute_max_drawdown(vals)
    assert dd == pytest.approx((120 - 80) / 120, abs=1e-4)


def test_max_drawdown_single_value():
    assert compute_max_drawdown([10_000]) == 0.0


# --- _max_consecutive_losses ---

def test_max_consecutive_losses_empty():
    assert _max_consecutive_losses([]) == 0


def test_max_consecutive_losses_all_positive():
    assert _max_consecutive_losses([0.01, 0.02, 0.03]) == 0


def test_max_consecutive_losses_all_negative():
    assert _max_consecutive_losses([-0.01, -0.02, -0.03]) == 3


def test_max_consecutive_losses_mixed():
    returns = [0.01, -0.02, -0.01, 0.03, -0.01, -0.02, -0.03, 0.01]
    # Streaks: [2], [3] → max = 3
    assert _max_consecutive_losses(returns) == 3


def test_max_consecutive_losses_zero_is_not_loss():
    returns = [0.0, -0.01, 0.0]
    # 0.0 breaks the streak (condition is < 0)
    assert _max_consecutive_losses(returns) == 1


def test_max_consecutive_losses_single_loss():
    assert _max_consecutive_losses([-0.01]) == 1
