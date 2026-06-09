import math
import pytest
from bot.strategy.ensemble import (
    ensemble_signal,
    action_to_int,
    STRONG_BUY_THRESHOLD,
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    STRONG_SELL_THRESHOLD,
    STRONG_BUY_FRACTION,
    BUY_FRACTION,
)


# --- ensemble_signal ---

def test_strong_buy():
    # xgb=0.9, lstm=0.9, sentiment=0.8, regime=TRENDING_UP → score ≈ 0.92
    action, fraction = ensemble_signal(0.9, 0.9, 0.8, "TRENDING_UP")
    assert action == "STRONG_BUY"
    assert fraction == STRONG_BUY_FRACTION


def test_buy():
    # xgb=0.7, lstm=0.7, sentiment=0.0, regime=RANGING → score ≈ 0.62
    action, fraction = ensemble_signal(0.7, 0.7, 0.0, "RANGING")
    assert action == "BUY"
    assert fraction == BUY_FRACTION


def test_hold():
    # xgb=0.5, lstm=0.5, sentiment=0.0, regime=RANGING → score = 0.50
    action, fraction = ensemble_signal(0.5, 0.5, 0.0, "RANGING")
    assert action == "HOLD"
    assert fraction == 0.0


def test_sell():
    # xgb=0.4, lstm=0.4, sentiment=-0.2, regime=TRENDING_DOWN → score ≈ 0.32
    action, fraction = ensemble_signal(0.4, 0.4, -0.2, "TRENDING_DOWN")
    assert action == "SELL"
    assert fraction == 0.0


def test_strong_sell():
    # xgb=0.2, lstm=0.2, sentiment=-0.8, regime=TRENDING_DOWN → score ≈ 0.14
    action, fraction = ensemble_signal(0.2, 0.2, -0.8, "TRENDING_DOWN")
    assert action == "STRONG_SELL"
    assert fraction == 0.0


def test_nan_input_defaults_to_hold():
    action, fraction = ensemble_signal(float("nan"), 0.5, 0.0, "RANGING")
    assert action == "HOLD"
    assert fraction == 0.0


def test_unknown_regime_uses_neutral_score():
    # Unknown regime → REGIME_SCORES.get(regime, 0.5) = 0.5
    action_known, _ = ensemble_signal(0.5, 0.5, 0.0, "RANGING")
    action_unknown, _ = ensemble_signal(0.5, 0.5, 0.0, "UNKNOWN_REGIME")
    assert action_known == action_unknown


def test_score_boundary_above_strong_buy():
    # Score just above STRONG_BUY_THRESHOLD
    action, _ = ensemble_signal(1.0, 1.0, 1.0, "TRENDING_UP")
    assert action == "STRONG_BUY"


def test_score_boundary_below_strong_sell():
    action, _ = ensemble_signal(0.0, 0.0, -1.0, "TRENDING_DOWN")
    assert action == "STRONG_SELL"


# --- action_to_int ---

@pytest.mark.parametrize("action,expected", [
    ("STRONG_BUY", 1),
    ("BUY", 1),
    ("STRONG_SELL", 2),
    ("SELL", 2),
    ("HOLD", 0),
])
def test_action_to_int(action, expected):
    assert action_to_int(action) == expected
