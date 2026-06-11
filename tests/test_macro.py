"""Tests for bot/strategy/macro.py pure functions."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
from unittest.mock import patch
from bot.strategy.macro import _sigmoid, _compute_from_raw


# --- _sigmoid ---

def test_sigmoid_center_is_0_5():
    # At x == center, output should be exactly 0.5
    assert _sigmoid(20.0, center=20.0, scale=5.0) == pytest.approx(0.5, abs=1e-6)


def test_sigmoid_approaches_1_above_center():
    # Far above center → approaches 1.0
    result = _sigmoid(100.0, center=20.0, scale=5.0)
    assert result > 0.99


def test_sigmoid_approaches_0_below_center():
    # Far below center → approaches 0.0
    result = _sigmoid(-100.0, center=20.0, scale=5.0)
    assert result < 0.01


def test_sigmoid_monotonic():
    vals = [_sigmoid(x, center=0.0, scale=1.0) for x in range(-10, 11)]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


# --- _compute_from_raw ---

def test_compute_from_raw_returns_expected_keys():
    raw = {"yield_curve": 0.5, "vix": 15.0, "fed_rate": 2.0}
    result = _compute_from_raw(raw)
    assert {"score", "cap", "halt"} <= set(result.keys())


def test_compute_from_raw_score_bounded():
    raw = {"yield_curve": 0.5, "vix": 15.0, "fed_rate": 2.0}
    result = _compute_from_raw(raw)
    assert 0.0 <= result["score"] <= 1.0


def test_compute_from_raw_bullish_conditions():
    # Low VIX, positive yield curve, low fed rate → high score
    raw = {"yield_curve": 1.0, "vix": 10.0, "fed_rate": 1.0}
    result = _compute_from_raw(raw)
    assert result["score"] > 0.6
    assert result["halt"] is False
    assert result["cap"] == 1.0


def test_compute_from_raw_bearish_conditions():
    # High VIX, inverted yield curve, high fed rate → low score
    raw = {"yield_curve": -0.5, "vix": 35.0, "fed_rate": 6.0}
    result = _compute_from_raw(raw)
    assert result["score"] < 0.4


def test_compute_from_raw_cap_reduced_when_vix_high():
    raw = {"yield_curve": 0.5, "vix": 31.0, "fed_rate": 2.0}
    result = _compute_from_raw(raw)
    assert result["cap"] == 0.5


def test_compute_from_raw_cap_reduced_when_yield_curve_inverted():
    raw = {"yield_curve": -0.1, "vix": 18.0, "fed_rate": 2.0}
    result = _compute_from_raw(raw)
    assert result["cap"] == 0.5


def test_compute_from_raw_cap_full_when_normal():
    raw = {"yield_curve": 0.3, "vix": 20.0, "fed_rate": 3.0}
    result = _compute_from_raw(raw)
    assert result["cap"] == 1.0


def test_compute_from_raw_halt_at_macro_halt_vix():
    from config import MACRO_HALT_VIX
    raw = {"yield_curve": 0.5, "vix": float(MACRO_HALT_VIX), "fed_rate": 2.0}
    result = _compute_from_raw(raw)
    assert result["halt"] is True


def test_compute_from_raw_no_halt_below_threshold():
    from config import MACRO_HALT_VIX
    raw = {"yield_curve": 0.5, "vix": float(MACRO_HALT_VIX) - 1.0, "fed_rate": 2.0}
    result = _compute_from_raw(raw)
    assert result["halt"] is False


def test_compute_from_raw_score_not_nan():
    raw = {"yield_curve": 0.0, "vix": 20.0, "fed_rate": 3.5}
    result = _compute_from_raw(raw)
    assert not math.isnan(result["score"])
