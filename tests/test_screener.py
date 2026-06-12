"""Tests for the pre-market universe screener (scripts/screen_universe.py).

Covers the pure/unit-testable functions without network calls:
  - _avg_overnight_gap
  - _analyst_signal  (requests mocked)
  - _sector_etf_momentum
  - _factor_weights  (sum-to-1 and regime-correctness)
  - _rank_pct        (monotonicity)
  - _earnings_blackout_set (yfinance mocked)
  - _detect_regime
  - _trend_r2
"""
import math
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from scripts.screen_universe import (
    _analyst_signal,
    _avg_overnight_gap,
    _detect_regime,
    _earnings_blackout_set,
    _factor_weights,
    _rank_pct,
    _sector_etf_momentum,
    _trend_r2,
)


# ── _avg_overnight_gap ────────────────────────────────────────────────────────

def _make_close_open(gaps: list[float], base_close: float = 100.0) -> tuple[pd.Series, pd.Series]:
    """Build aligned close/open Series where each day's open = prior close * (1 + gap)."""
    idx = pd.date_range("2026-01-01", periods=len(gaps) + 1, freq="B")
    closes = pd.Series([base_close] * (len(gaps) + 1), index=idx)
    opens  = pd.Series(
        [base_close] + [base_close * (1 + g) for g in gaps],
        index=idx,
    )
    return closes, opens


def test_avg_overnight_gap_zero_for_no_gaps():
    closes, opens = _make_close_open([0.0] * 25)
    assert _avg_overnight_gap(closes, opens) == pytest.approx(0.0, abs=1e-6)


def test_avg_overnight_gap_correct_for_uniform_gaps():
    # Every overnight gap is exactly 2%
    closes, opens = _make_close_open([0.02] * 25)
    assert _avg_overnight_gap(closes, opens) == pytest.approx(0.02, abs=0.001)


def test_avg_overnight_gap_uses_absolute_value():
    # Mix of +2% and -2% gaps → avg abs should still be ~2%
    gaps = ([0.02, -0.02] * 13)[:25]
    closes, opens = _make_close_open(gaps)
    assert _avg_overnight_gap(closes, opens) == pytest.approx(0.02, abs=0.005)


def test_avg_overnight_gap_returns_zero_on_short_series():
    # Fewer than 20 bars → not enough history, returns 0 (conservative: don't filter)
    closes, opens = _make_close_open([0.05] * 10)
    assert _avg_overnight_gap(closes, opens) == 0.0


def test_avg_overnight_gap_only_uses_last_20_bars():
    # First 30 bars have 10% gaps, last 20 are flat — result should be ~0
    gaps = [0.10] * 30 + [0.0] * 20
    closes, opens = _make_close_open(gaps)
    gap = _avg_overnight_gap(closes, opens)
    assert gap < 0.01   # last 20 bars dominate


# ── _analyst_signal ───────────────────────────────────────────────────────────

def test_analyst_signal_returns_zero_without_token():
    assert _analyst_signal("AAPL", token="") == 0.0


def test_analyst_signal_returns_zero_on_http_error():
    with patch("scripts.screen_universe.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=429)
        assert _analyst_signal("AAPL", token="test") == 0.0


def test_analyst_signal_returns_zero_on_empty_response():
    with patch("scripts.screen_universe.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [])
        assert _analyst_signal("AAPL", token="test") == 0.0


def test_analyst_signal_positive_for_recent_upgrade():
    now_ts = datetime.now(timezone.utc).timestamp()
    with patch("scripts.screen_universe.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{
                "action": "up", "toGrade": "Buy",
                "fromGrade": "Hold", "gradeTime": now_ts,
                "symbol": "AAPL", "company": "GS",
            }],
        )
        sig = _analyst_signal("AAPL", token="test")
    assert sig > 0.0


def test_analyst_signal_negative_for_recent_downgrade():
    now_ts = datetime.now(timezone.utc).timestamp()
    with patch("scripts.screen_universe.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{
                "action": "down", "toGrade": "Sell",
                "fromGrade": "Buy", "gradeTime": now_ts,
                "symbol": "AAPL", "company": "MS",
            }],
        )
        sig = _analyst_signal("AAPL", token="test")
    assert sig < 0.0


def test_analyst_signal_ignores_stale_actions():
    # Action from 10 days ago — outside the 5-day lookback window
    old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).timestamp()
    with patch("scripts.screen_universe.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{
                "action": "up", "toGrade": "Buy",
                "fromGrade": "Hold", "gradeTime": old_ts,
                "symbol": "AAPL", "company": "GS",
            }],
        )
        sig = _analyst_signal("AAPL", token="test")
    assert sig == 0.0


def test_analyst_signal_clipped_to_plus_minus_1():
    now_ts = datetime.now(timezone.utc).timestamp()
    # 5 upgrades → raw score = 5/5 = 1.0 → clipped to 1.0
    items = [
        {"action": "up", "toGrade": "Buy", "fromGrade": "Hold",
         "gradeTime": now_ts, "symbol": "X", "company": f"Bank{i}"}
        for i in range(5)
    ]
    with patch("scripts.screen_universe.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: items)
        sig = _analyst_signal("X", token="test")
    assert sig == pytest.approx(1.0)


def test_analyst_signal_handles_exception_gracefully():
    with patch("scripts.screen_universe.requests.get", side_effect=ConnectionError("timeout")):
        assert _analyst_signal("AAPL", token="test") == 0.0


# ── _sector_etf_momentum ──────────────────────────────────────────────────────

def _make_close_df(etf: str, prices: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({etf: prices}, index=idx)


def test_sector_etf_momentum_returns_1_when_above_sma():
    # 20 days trending up → last close well above SMA
    prices = list(range(1, 22))   # 1..21, last=21, sma=11
    df = _make_close_df("XLK", prices)
    assert _sector_etf_momentum("Technology", df) == 1.0


def test_sector_etf_momentum_returns_0_when_below_sma():
    # 20 days trending down → last close below SMA
    prices = list(range(21, 0, -1))  # 21..1, last=1, sma=11
    df = _make_close_df("XLK", prices)
    assert _sector_etf_momentum("Technology", df) == 0.0


def test_sector_etf_momentum_neutral_when_etf_not_in_df():
    df = pd.DataFrame({"SPY": [100.0] * 25})
    # Healthcare ETF (XLV) is not in this df
    assert _sector_etf_momentum("Healthcare", df) == 0.5


def test_sector_etf_momentum_neutral_on_insufficient_history():
    prices = [100.0] * 10   # only 10 bars, need 20
    df = _make_close_df("XLV", prices)
    assert _sector_etf_momentum("Healthcare", df) == 0.5


def test_sector_etf_momentum_neutral_for_unknown_sector():
    df = pd.DataFrame({"SPY": [100.0] * 25})
    assert _sector_etf_momentum("UnknownSector", df) == 0.5


# ── _factor_weights ───────────────────────────────────────────────────────────

def test_factor_weights_bull_sum_to_one():
    w = _factor_weights("BULL")
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)


def test_factor_weights_bear_sum_to_one():
    w = _factor_weights("BEAR")
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)


def test_factor_weights_bull_emphasises_momentum():
    w = _factor_weights("BULL")
    assert w["risk_adj_mom"] >= w.get("defensive", 0)


def test_factor_weights_bear_emphasises_rs_and_defensive():
    w = _factor_weights("BEAR")
    assert w.get("defensive", 0) >= w["risk_adj_mom"]


def test_factor_weights_etf_momentum_present_in_both_regimes():
    for regime in ("BULL", "BEAR"):
        w = _factor_weights(regime)
        assert "etf_momentum" in w
        assert w["etf_momentum"] > 0


# ── _rank_pct ─────────────────────────────────────────────────────────────────

def test_rank_pct_monotonically_increasing():
    s = pd.Series([10.0, 30.0, 20.0, 50.0, 40.0])
    ranked = _rank_pct(s)
    # The ranking of values should be monotonically related to the original values
    assert ranked.corr(s) > 0.99


def test_rank_pct_output_in_zero_one():
    s = pd.Series([5.0, 3.0, 8.0, 1.0])
    ranked = _rank_pct(s)
    assert ranked.min() > 0.0
    assert ranked.max() <= 1.0


# ── _detect_regime ────────────────────────────────────────────────────────────

def test_detect_regime_bull_when_above_50sma():
    # Last price above 50-day SMA → BULL
    prices = pd.Series([100.0] * 49 + [110.0])  # last bar jumps above average
    assert _detect_regime(prices) == "BULL"


def test_detect_regime_bear_when_below_50sma():
    # Last price well below 50-day SMA → BEAR
    prices = pd.Series([100.0] * 49 + [80.0])
    assert _detect_regime(prices) == "BEAR"


def test_detect_regime_defaults_bull_on_short_series():
    # Fewer than 50 bars → default BULL (conservative: don't restrict universe)
    prices = pd.Series([100.0] * 30)
    assert _detect_regime(prices) == "BULL"


# ── _trend_r2 ─────────────────────────────────────────────────────────────────

def test_trend_r2_perfect_trend_is_one():
    # Perfectly linear uptrend → R² = 1.0
    prices = pd.Series(list(range(1, 21)), dtype=float)
    assert _trend_r2(prices) == pytest.approx(1.0, abs=0.001)


def test_trend_r2_random_noise_is_low():
    rng = np.random.default_rng(42)
    prices = pd.Series(rng.standard_normal(20))
    # Random noise should have low R² (not a trend)
    assert _trend_r2(prices) < 0.5


def test_trend_r2_returns_zero_on_short_series():
    prices = pd.Series([100.0, 101.0, 102.0])  # < 5 bars
    assert _trend_r2(prices) == 0.0


# ── _earnings_blackout_set ────────────────────────────────────────────────────

def test_earnings_blackout_blocks_symbol_with_earnings_today():
    today = date.today()
    mock_cal = pd.Series({"Earnings Date": pd.Timestamp(today)})
    with patch("scripts.screen_universe.yf.Ticker") as MockTicker:
        MockTicker.return_value.calendar = mock_cal
        blocked = _earnings_blackout_set(["AAPL"])
    assert "AAPL" in blocked


def test_earnings_blackout_allows_symbol_with_distant_earnings():
    future = date.today() + timedelta(days=30)
    mock_cal = pd.Series({"Earnings Date": pd.Timestamp(future)})
    with patch("scripts.screen_universe.yf.Ticker") as MockTicker:
        MockTicker.return_value.calendar = mock_cal
        blocked = _earnings_blackout_set(["AAPL"])
    assert "AAPL" not in blocked


def test_earnings_blackout_skips_symbol_on_api_error():
    with patch("scripts.screen_universe.yf.Ticker", side_effect=Exception("network error")):
        blocked = _earnings_blackout_set(["AAPL"])
    # Should silently skip — better to include than falsely exclude
    assert "AAPL" not in blocked


def test_earnings_blackout_empty_input_returns_empty_set():
    blocked = _earnings_blackout_set([])
    assert blocked == set()
