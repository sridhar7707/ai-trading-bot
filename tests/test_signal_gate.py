"""Tests for bot/strategy/signal_gate.py.

Thresholds from signal_gate.py:
    XGB      >= 0.62
    LSTM     > 0.55  (indeterminate zone [0.45, 0.55] also blocks)
    Volume   >= 1.0
    Macro    >= 0.50
    SPY      > 0.0   (strictly positive)
    R:R      >= 2.0  (always 2.0 by construction; can't fail with normal inputs)
    Setup    must be 'breakout' or 'pullback'
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("_BOT_LOG_HANDLER_ADDED", "1")

from bot.strategy.signal_gate import check_signal_gate


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_bars(
    n: int = 35,
    close: float = 100.0,
    high_52w: float | None = None,
    sma_20: float | None = None,
) -> pd.DataFrame:
    """Minimal DataFrame with 'close' and 'high' columns accepted by check_signal_gate."""
    peak = high_52w if high_52w is not None else close
    # First row carries the 52-week peak; rest match close
    highs = [peak] + [close] * (n - 1)
    df = pd.DataFrame({"close": [close] * n, "high": highs})
    if sma_20 is not None:
        df["sma_20"] = float(sma_20)
    return df


def _all_pass_kwargs(bars: pd.DataFrame) -> dict:
    """Keyword arguments that cause every gate to pass when bars are valid."""
    return dict(
        symbol="TEST",
        xgb_prob=0.65,
        lstm_prob=0.60,
        macro_score=0.55,
        bars_daily=bars,
        volume_ratio=1.5,
        spy_today_pct=0.01,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def breakout_bars():
    """Breakout setup: close equals 52-week high (pct_from_52wh = 0)."""
    return _make_bars(close=100.0, high_52w=100.0)


@pytest.fixture
def pullback_bars():
    """Pullback setup: price 1% above SMA20; 52-week high is 20% above (not breakout)."""
    return _make_bars(close=101.0, high_52w=120.0, sma_20=100.0)


# ── Gate 1: XGB threshold (0.62) ─────────────────────────────────────────────

def test_xgb_passes_above_threshold(breakout_bars):
    _, meta = check_signal_gate(
        "TEST", xgb_prob=0.63, lstm_prob=0.60, macro_score=0.55,
        bars_daily=breakout_bars, volume_ratio=1.5, spy_today_pct=0.01,
    )
    assert "XGB" not in meta["reason"]


def test_xgb_blocked_below_threshold(breakout_bars):
    _, meta = check_signal_gate(
        "TEST", xgb_prob=0.61, lstm_prob=0.60, macro_score=0.55,
        bars_daily=breakout_bars, volume_ratio=1.5, spy_today_pct=0.01,
    )
    assert "XGB" in meta["reason"]


# ── Gate 2: LSTM threshold (>0.55; [0.45, 0.55] is indeterminate) ────────────

def test_lstm_passes_above_indeterminate_zone(breakout_bars):
    _, meta = check_signal_gate(
        "TEST", xgb_prob=0.65, lstm_prob=0.56, macro_score=0.55,
        bars_daily=breakout_bars, volume_ratio=1.5, spy_today_pct=0.01,
    )
    assert "LSTM" not in meta["reason"]


def test_lstm_blocked_in_indeterminate_zone(breakout_bars):
    _, meta = check_signal_gate(
        "TEST", xgb_prob=0.65, lstm_prob=0.50, macro_score=0.55,
        bars_daily=breakout_bars, volume_ratio=1.5, spy_today_pct=0.01,
    )
    assert "LSTM" in meta["reason"]


def test_lstm_blocked_below_threshold(breakout_bars):
    _, meta = check_signal_gate(
        "TEST", xgb_prob=0.65, lstm_prob=0.44, macro_score=0.55,
        bars_daily=breakout_bars, volume_ratio=1.5, spy_today_pct=0.01,
    )
    assert "LSTM" in meta["reason"]


# ── Gate 3: Volume ratio (>= 1.0) ────────────────────────────────────────────

def test_volume_passes_at_threshold(breakout_bars):
    _, meta = check_signal_gate(
        "TEST", xgb_prob=0.65, lstm_prob=0.60, macro_score=0.55,
        bars_daily=breakout_bars, volume_ratio=1.0, spy_today_pct=0.01,
    )
    assert "volume" not in meta["reason"]


def test_volume_blocked_below_threshold(breakout_bars):
    _, meta = check_signal_gate(
        "TEST", xgb_prob=0.65, lstm_prob=0.60, macro_score=0.55,
        bars_daily=breakout_bars, volume_ratio=0.99, spy_today_pct=0.01,
    )
    assert "volume" in meta["reason"]


# ── Gate 4: Technical setup (breakout or pullback) ────────────────────────────

def test_setup_type_is_breakout(breakout_bars):
    _, meta = check_signal_gate(**_all_pass_kwargs(breakout_bars))
    assert meta["setup_type"] == "breakout"


def test_setup_type_is_pullback(pullback_bars):
    _, meta = check_signal_gate(**_all_pass_kwargs(pullback_bars))
    assert meta["setup_type"] == "pullback"


def test_no_technical_setup_blocks():
    # price 15% below 52-week high (not breakout) and below SMA20 (not pullback)
    bars = _make_bars(close=85.0, high_52w=100.0, sma_20=100.0)
    _, meta = check_signal_gate(
        "TEST", xgb_prob=0.65, lstm_prob=0.60, macro_score=0.55,
        bars_daily=bars, volume_ratio=1.5, spy_today_pct=0.01,
    )
    assert meta["setup_type"] == "none"
    assert "no technical setup" in meta["reason"]


# ── Gate 5: SPY strictly positive ────────────────────────────────────────────

def test_spy_positive_passes(breakout_bars):
    _, meta = check_signal_gate(
        "TEST", xgb_prob=0.65, lstm_prob=0.60, macro_score=0.55,
        bars_daily=breakout_bars, volume_ratio=1.5, spy_today_pct=0.001,
    )
    assert "SPY" not in meta["reason"]


def test_spy_zero_blocked(breakout_bars):
    _, meta = check_signal_gate(
        "TEST", xgb_prob=0.65, lstm_prob=0.60, macro_score=0.55,
        bars_daily=breakout_bars, volume_ratio=1.5, spy_today_pct=0.0,
    )
    assert "SPY" in meta["reason"]


def test_spy_negative_blocked(breakout_bars):
    _, meta = check_signal_gate(
        "TEST", xgb_prob=0.65, lstm_prob=0.60, macro_score=0.55,
        bars_daily=breakout_bars, volume_ratio=1.5, spy_today_pct=-0.005,
    )
    assert "SPY" in meta["reason"]


# ── Gate 7: Macro threshold (>= 0.50) ────────────────────────────────────────

def test_macro_passes_at_threshold(breakout_bars):
    _, meta = check_signal_gate(
        "TEST", xgb_prob=0.65, lstm_prob=0.60, macro_score=0.50,
        bars_daily=breakout_bars, volume_ratio=1.5, spy_today_pct=0.01,
    )
    assert "macro" not in meta["reason"]


def test_macro_blocked_below_threshold(breakout_bars):
    _, meta = check_signal_gate(
        "TEST", xgb_prob=0.65, lstm_prob=0.60, macro_score=0.49,
        bars_daily=breakout_bars, volume_ratio=1.5, spy_today_pct=0.01,
    )
    assert "macro" in meta["reason"]


# ── check_signal_gate: overall pass / fail ────────────────────────────────────

def test_all_gates_pass(breakout_bars):
    passed, meta = check_signal_gate(**_all_pass_kwargs(breakout_bars))
    assert passed is True
    assert meta["reason"] == "all gates passed"


def test_one_gate_blocks_overall(breakout_bars):
    kwargs = _all_pass_kwargs(breakout_bars)
    kwargs["xgb_prob"] = 0.50   # XGB fails
    passed, meta = check_signal_gate(**kwargs)
    assert passed is False
    assert "XGB" in meta["reason"]


# ── Insufficient bar history ──────────────────────────────────────────────────

def test_insufficient_bars_returns_false():
    bars = pd.DataFrame({"close": [100.0] * 10, "high": [100.0] * 10})
    passed, meta = check_signal_gate("TEST", 0.65, 0.60, 0.55, bars, 1.5, 0.01)
    assert passed is False
    assert "insufficient" in meta["reason"]


def test_none_bars_returns_false():
    passed, meta = check_signal_gate("TEST", 0.65, 0.60, 0.55, None, 1.5, 0.01)
    assert passed is False
    assert "insufficient" in meta["reason"]


def test_empty_bars_returns_false():
    bars = pd.DataFrame({"close": [], "high": []})
    passed, meta = check_signal_gate("TEST", 0.65, 0.60, 0.55, bars, 1.5, 0.01)
    assert passed is False


# ── Meta dict structure ───────────────────────────────────────────────────────

def test_meta_contains_expected_keys(breakout_bars):
    _, meta = check_signal_gate(**_all_pass_kwargs(breakout_bars))
    expected = {
        "entry_price", "stop_price", "target_price", "rr_ratio",
        "setup_type", "high_52w", "volume_ratio", "spy_today_pct", "reason",
    }
    assert expected <= set(meta.keys())


def test_meta_entry_price_matches_close(breakout_bars):
    _, meta = check_signal_gate(**_all_pass_kwargs(breakout_bars))
    assert meta["entry_price"] == pytest.approx(100.0)


def test_meta_stop_is_4pct_below_entry(breakout_bars):
    _, meta = check_signal_gate(**_all_pass_kwargs(breakout_bars))
    assert meta["stop_price"] == pytest.approx(96.0)


def test_meta_rr_ratio_at_least_2(breakout_bars):
    _, meta = check_signal_gate(**_all_pass_kwargs(breakout_bars))
    assert meta["rr_ratio"] >= 2.0


def test_meta_setup_type_is_valid_string(breakout_bars):
    _, meta = check_signal_gate(**_all_pass_kwargs(breakout_bars))
    assert meta["setup_type"] in {"breakout", "pullback", "none"}
