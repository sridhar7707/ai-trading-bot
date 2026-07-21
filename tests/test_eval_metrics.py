"""Tests for bot/eval/metrics.py — pure metric functions."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pandas as pd
import pytest

from bot.eval.metrics import (
    win_rate,
    avg_return,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    profit_factor,
    avg_holding_period,
    alpha_vs_spy,
    precision_recall,
    calibration_buckets,
    summary,
)


# ── win_rate ─────────────────────────────────────────────────────────────────

def test_win_rate_all_wins():
    assert win_rate(pd.Series([0.01, 0.05, 0.02])) == pytest.approx(1.0)


def test_win_rate_all_losses():
    assert win_rate(pd.Series([-0.01, -0.03])) == pytest.approx(0.0)


def test_win_rate_mixed():
    assert win_rate(pd.Series([0.05, -0.02, 0.01, -0.01])) == pytest.approx(0.5)


def test_win_rate_empty():
    assert win_rate(pd.Series(dtype=float)) == 0.0


# ── avg_return ────────────────────────────────────────────────────────────────

def test_avg_return_positive():
    assert avg_return(pd.Series([0.10, 0.20])) == pytest.approx(0.15)


def test_avg_return_empty():
    assert avg_return(pd.Series(dtype=float)) == 0.0


# ── max_drawdown ──────────────────────────────────────────────────────────────

def test_max_drawdown_monotone_increase():
    assert max_drawdown(pd.Series([100.0, 110.0, 120.0])) == pytest.approx(0.0)


def test_max_drawdown_single_drop():
    # peak 120, trough 80 → dd = (120−80)/120
    eq = pd.Series([100.0, 120.0, 80.0])
    assert max_drawdown(eq) == pytest.approx(-(120 - 80) / 120, abs=1e-4)


def test_max_drawdown_empty():
    assert max_drawdown(pd.Series(dtype=float)) == 0.0


def test_max_drawdown_single_value():
    assert max_drawdown(pd.Series([1000.0])) == 0.0


# ── sharpe_ratio ──────────────────────────────────────────────────────────────

def test_sharpe_positive_mean_zero_std():
    assert sharpe_ratio(pd.Series([0.05, 0.05, 0.05])) == 0.0


def test_sharpe_empty():
    assert sharpe_ratio(pd.Series(dtype=float)) == 0.0


def test_sharpe_direction():
    # Higher mean → higher Sharpe
    high = sharpe_ratio(pd.Series([0.10, 0.08, 0.12]))
    low  = sharpe_ratio(pd.Series([0.01, -0.01, 0.01]))
    assert high > low


# ── sortino_ratio ─────────────────────────────────────────────────────────────

def test_sortino_no_losses():
    sr = sortino_ratio(pd.Series([0.05, 0.03, 0.07]))
    assert sr == float("inf")


def test_sortino_all_losses():
    sr = sortino_ratio(pd.Series([-0.02, -0.04]))
    assert sr < 0


def test_sortino_empty():
    assert sortino_ratio(pd.Series(dtype=float)) == 0.0


# ── profit_factor ─────────────────────────────────────────────────────────────

def test_profit_factor_no_losses():
    pf = profit_factor(pd.Series([0.05, 0.10]))
    assert pf == float("inf")


def test_profit_factor_no_wins():
    pf = profit_factor(pd.Series([-0.05, -0.10]))
    assert pf == 0.0


def test_profit_factor_balanced():
    # gross profit = 0.15, gross loss = 0.05 → PF = 3.0
    pf = profit_factor(pd.Series([0.10, 0.05, -0.05]))
    assert pf == pytest.approx(3.0, abs=1e-6)


# ── avg_holding_period ────────────────────────────────────────────────────────

def test_avg_holding_period_basic():
    assert avg_holding_period(pd.Series([5.0, 10.0, 15.0])) == pytest.approx(10.0)


def test_avg_holding_period_empty():
    assert avg_holding_period(pd.Series(dtype=float)) == 0.0


# ── alpha_vs_spy ──────────────────────────────────────────────────────────────

def test_alpha_empty_spy():
    dates = pd.Series(["2026-01-02", "2026-01-03"])
    pnl   = pd.Series([0.02, 0.01])
    assert alpha_vs_spy(dates, pnl, pd.Series(dtype=float)) == 0.0


def test_alpha_beats_spy():
    import datetime
    dates = pd.Series([datetime.date(2026, 1, 2), datetime.date(2026, 1, 3)])
    pnl   = pd.Series([0.05, 0.04])
    spy   = pd.Series({datetime.date(2026, 1, 2): 0.01, datetime.date(2026, 1, 3): 0.01})
    al = alpha_vs_spy(dates, pnl, spy)
    assert al > 0


# ── precision_recall ──────────────────────────────────────────────────────────

def test_precision_recall_all_correct():
    scores = pd.Series([0.60, 0.65, 0.70])
    pnl    = pd.Series([0.05, 0.03, 0.02])
    pr = precision_recall(scores, pnl, score_threshold=0.55)
    assert pr["precision"] == pytest.approx(1.0)
    assert pr["recall"]    == pytest.approx(1.0)


def test_precision_recall_all_wrong():
    scores = pd.Series([0.60, 0.65])
    pnl    = pd.Series([-0.05, -0.03])
    pr = precision_recall(scores, pnl, score_threshold=0.55)
    assert pr["precision"] == pytest.approx(0.0)
    assert pr["recall"]    == pytest.approx(0.0)


def test_precision_recall_empty():
    pr = precision_recall(pd.Series(dtype=float), pd.Series(dtype=float), 0.55)
    assert pr["f1"] == 0.0


def test_precision_recall_fn():
    # score below threshold but trade was profitable → FN
    scores = pd.Series([0.40, 0.60])
    pnl    = pd.Series([0.05, 0.05])
    pr = precision_recall(scores, pnl, score_threshold=0.55)
    assert pr["fn"] == 1
    assert pr["recall"] < 1.0


# ── calibration_buckets ───────────────────────────────────────────────────────

def test_calibration_buckets_counts():
    scores = pd.Series([0.50, 0.56, 0.62, 0.68])
    pnl    = pd.Series([0.01, 0.02, -0.01, 0.03])
    edges  = [0.50, 0.55, 0.65, 1.01]
    buckets = calibration_buckets(scores, pnl, edges)
    total = sum(b["count"] for b in buckets)
    assert total == 4  # all values covered


def test_calibration_bucket_win_rate():
    scores = pd.Series([0.60, 0.62])
    pnl    = pd.Series([0.01, -0.01])
    buckets = calibration_buckets(scores, pnl, [0.55, 0.65, 1.01])
    assert buckets[0]["win_rate"] == pytest.approx(0.5)


# ── summary ───────────────────────────────────────────────────────────────────

def test_summary_keys():
    m = summary(
        pd.Series([0.05, -0.02, 0.03]),
        pd.Series([5.0, 8.0, 10.0]),
    )
    for key in ("n_trades", "win_rate", "avg_return", "max_drawdown",
                "sharpe", "sortino", "profit_factor", "avg_holding_days"):
        assert key in m


def test_summary_n_trades():
    m = summary(pd.Series([0.01, -0.01, 0.02]), pd.Series([3.0, 4.0, 5.0]))
    assert m["n_trades"] == 3
