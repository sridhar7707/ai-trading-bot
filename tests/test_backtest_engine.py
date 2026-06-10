from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest
from bot.strategy.features import FEATURE_COLS

SLIPPAGE = 7 / 10_000  # must match engine.py SLIPPAGE_BPS


def _make_df(prices: list[float]) -> pd.DataFrame:
    n = len(prices)
    data = {col: [0.5] * n for col in FEATURE_COLS}
    data["close"] = prices
    data["atr"] = [0.0] * n
    return pd.DataFrame(data)


def _make_df_with_nan(prices: list[float]) -> pd.DataFrame:
    """First row has NaN in all feature columns; subsequent rows are clean."""
    n = len(prices)
    data = {col: [np.nan] + [0.5] * (n - 1) for col in FEATURE_COLS}
    data["close"] = prices
    data["atr"] = [0.0] * n
    return pd.DataFrame(data)


@pytest.fixture(autouse=True)
def mock_heavy_deps():
    """Patch ML deps that need model files so tests run cold."""
    with patch("backtest.engine.compute_features", side_effect=lambda df: df), \
         patch("backtest.engine.RegimeClassifier") as mock_rc_class:
        mock_rc = MagicMock()
        mock_rc.predict.return_value = 0
        mock_rc.regime_name.return_value = "RANGING"
        mock_rc_class.return_value = mock_rc
        yield


def _run(df, xgb_probas: list[float]):
    """
    Run backtest with a controlled XGB proba sequence (LSTM held at 0.5).
    With LSTM=0.5 and macro=0.5 (defaults), the thresholds are:
      xgb > 0.64 → BUY    (score = 0.35*xgb + 0.35*0.5 + 0.15*0.5 + 0.15*0.5)
      xgb < 0.36 → SELL
      else       → HOLD
    So xgb=0.9 → BUY, xgb=0.1 → SELL, xgb=0.5 → HOLD.
    Returns (portfolio_values, trades) passed to compute_metrics.
    """
    proba_iter = iter(xgb_probas)
    with patch("backtest.engine.compute_metrics") as mock_cm, \
         patch("backtest.engine.XGBPredictor") as mock_xgb_class, \
         patch("backtest.engine.LSTMPredictor") as mock_lstm_class:

        mock_xgb = MagicMock()
        mock_xgb.predict_proba.side_effect = lambda row: next(proba_iter)
        mock_xgb_class.return_value = mock_xgb

        mock_lstm = MagicMock()
        mock_lstm.predict_proba.return_value = 0.5
        mock_lstm_class.return_value = mock_lstm

        mock_cm.return_value = {"sharpe": 0.0, "total_return": 0.0}

        from backtest.engine import run_backtest
        run_backtest(df, initial_balance=1000.0)
        portfolio_values, trades, _ = mock_cm.call_args[0]
        return portfolio_values, trades


# --- Cost basis ---

def test_single_buy_sell_pnl():
    # BUY at 100, SELL at 102 (2% rise — below 6% TP threshold so signal fires)
    portfolio_values, trades = _run(_make_df([100.0, 102.0]), [0.9, 0.1])

    sell_trade = next(t for t in trades if t["action"] == "SELL")
    fill_buy  = 100.0 * (1 + SLIPPAGE)
    fill_sell = 102.0 * (1 - SLIPPAGE)
    expected_pnl = (fill_sell - fill_buy) / fill_buy
    assert abs(sell_trade["pnl_pct"] - expected_pnl) < 1e-6


def test_multiple_buys_use_weighted_average_cost_basis():
    # BUY at 100 (spend 200), BUY at 102 (spend 160), SELL at 104
    # entry_price = total_cost / total_shares = weighted average of fill prices
    portfolio_values, trades = _run(_make_df([100.0, 102.0, 104.0]), [0.9, 0.9, 0.1])

    sell = next(t for t in trades if t["action"] == "SELL")

    fill_buy1 = 100.0 * (1 + SLIPPAGE)
    fill_buy2 = 102.0 * (1 + SLIPPAGE)
    fill_sell = 104.0 * (1 - SLIPPAGE)

    # initial_balance=1000, MAX_POSITION_PCT=0.20
    spend1 = 1000.0 * 0.20          # 200
    spend2 = (1000.0 - spend1) * 0.20  # 160
    shares1 = spend1 / fill_buy1
    shares2 = spend2 / fill_buy2
    expected_entry = (spend1 + spend2) / (shares1 + shares2)
    expected_pnl   = (fill_sell - expected_entry) / expected_entry

    assert abs(sell["pnl_pct"] - expected_pnl) < 1e-4


# --- NaN row skipping ---

def test_nan_rows_are_skipped_no_trades():
    # First row has NaN features → model is never called for that row.
    # Row 1 gets HOLD signal → no trade at all.
    portfolio_values, trades = _run(_make_df_with_nan([100.0, 110.0]), [0.5])

    buy_steps = [t["step"] for t in trades if t["action"] == "BUY"]
    assert 0 not in buy_steps


def test_nan_rows_still_tracked_in_portfolio_values():
    # Both rows (NaN and clean) must appear in portfolio_values.
    portfolio_values, trades = _run(_make_df_with_nan([100.0, 110.0]), [0.5])

    assert len(portfolio_values) == 2
