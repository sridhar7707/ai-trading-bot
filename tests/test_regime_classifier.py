import pandas as pd
import pytest
from bot.strategy.regime_classifier import RegimeClassifier, ATR_VOLATILITY_THRESHOLD


@pytest.fixture
def clf():
    rc = RegimeClassifier()
    rc.model = None  # force rule-based fallback; no model file needed
    return rc


def _row(**kwargs) -> pd.Series:
    defaults = {"close": 100.0, "atr": 0.0, "rsi": 50.0, "macd_diff": 0.0}
    defaults.update(kwargs)
    return pd.Series(defaults)


def test_high_volatility(clf):
    row = _row(atr=4.0, close=100.0)  # atr_ratio = 0.04 > 0.03
    assert clf.predict(row) == 3


def test_trending_up(clf):
    row = _row(rsi=60.0, macd_diff=0.5, atr=1.0)  # atr_ratio = 0.01 < threshold
    assert clf.predict(row) == 0


def test_trending_down(clf):
    row = _row(rsi=40.0, macd_diff=-0.5, atr=1.0)
    assert clf.predict(row) == 1


def test_ranging(clf):
    row = _row(rsi=50.0, macd_diff=0.0, atr=1.0)
    assert clf.predict(row) == 2


def test_close_zero_does_not_divide_by_zero(clf):
    # close=0 must not raise ZeroDivisionError
    row = _row(close=0.0, atr=1.0)
    result = clf.predict(row)
    assert result in (0, 1, 2, 3)


def test_atr_exactly_at_threshold_is_ranging(clf):
    # atr_ratio == ATR_VOLATILITY_THRESHOLD is not > threshold → not HIGH_VOLATILITY
    row = _row(atr=ATR_VOLATILITY_THRESHOLD * 100, close=100.0)  # ratio exactly 0.03
    assert clf.predict(row) != 3  # boundary is exclusive (>), not (>=)


def test_regime_name(clf):
    assert clf.regime_name(0) == "TRENDING_UP"
    assert clf.regime_name(1) == "TRENDING_DOWN"
    assert clf.regime_name(2) == "RANGING"
    assert clf.regime_name(3) == "HIGH_VOLATILITY"
    assert clf.regime_name(99) == "UNKNOWN"
