from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest
from bot.strategy.features import FEATURE_COLS


def _make_df(actions: list[int], prices: list[float]) -> pd.DataFrame:
    """Build a minimal DataFrame with FEATURE_COLS all set to 0.5 (no NaN)."""
    n = len(actions)
    data = {col: [0.5] * n for col in FEATURE_COLS}
    data["close"] = prices
    data["_action"] = actions  # consumed by mock RLAgent
    return pd.DataFrame(data)


def _make_df_with_nan(prices: list[float]) -> pd.DataFrame:
    """DataFrame where the first row has NaN in all feature columns."""
    n = len(prices)
    data = {col: [np.nan] + [0.5] * (n - 1) for col in FEATURE_COLS}
    data["close"] = prices
    data["_action"] = [1] + [0] * (n - 1)
    return pd.DataFrame(data)


@pytest.fixture(autouse=True)
def mock_deps():
    """Patch the heavy ML dependencies so tests run without model files."""
    with patch("backtest.engine.compute_features", side_effect=lambda df: df), \
         patch("backtest.engine.compute_metrics", return_value={"sharpe": 1.0, "total_return": 0.1}) as mock_metrics, \
         patch("backtest.engine.RegimeClassifier") as mock_rc_class, \
         patch("backtest.engine.RLAgent") as mock_rl_class:

        mock_rc = MagicMock()
        mock_rc.predict.return_value = 0
        mock_rc_class.return_value = mock_rc

        # RLAgent reads action from the _action column of each row
        mock_rl = MagicMock()
        mock_rl.predict.side_effect = lambda obs: int(obs[-1])  # last element = action marker
        mock_rl_class.return_value = mock_rl

        yield


def _run(df):
    """Attach the _action value as the last obs element so RLAgent mock can read it."""
    from backtest.engine import run_backtest
    # Append _action to FEATURE_COLS obs by monkey-patching isn't clean;
    # instead, store action in regime slot (3rd extra obs element) via mock
    # Simpler: use a custom mock that returns actions in sequence
    actions = df["_action"].tolist()
    action_iter = iter(actions)

    with patch("backtest.engine.RLAgent") as mock_rl_class:
        mock_rl = MagicMock()
        mock_rl.predict.side_effect = lambda obs: next(action_iter)
        mock_rl_class.return_value = mock_rl
        return run_backtest(df.drop(columns=["_action"]), initial_balance=1000.0)


# --- Cost basis ---

def test_single_buy_sell_pnl():
    # BUY at 100, SELL at 120 → pnl_pct = (120-100)/100 = 0.20
    from backtest.engine import run_backtest
    with patch("backtest.engine.compute_metrics") as mock_cm, \
         patch("backtest.engine.RLAgent") as mock_rl_class:
        mock_rl = MagicMock()
        mock_rl.predict.side_effect = [1, 2]  # BUY then SELL
        mock_rl_class.return_value = mock_rl
        mock_cm.return_value = {"sharpe": 0.0, "total_return": 0.0}
        run_backtest(_make_df([1, 2], [100.0, 120.0]).drop(columns=["_action"]), 1000.0)
        trades = mock_cm.call_args[0][1]
        sell_trade = next(t for t in trades if t["action"] == "SELL")
        assert abs(sell_trade["pnl_pct"] - 0.20) < 1e-6


def test_multiple_buys_use_weighted_average_cost_basis():
    # BUY at 100 (spend 200), BUY at 120 (spend ~160), SELL at 130
    # avg cost = (200 + 160) / (2.0 + 160/120) = 360 / 3.333 ≈ 108
    # pnl_pct = (130 - 108) / 108 ≈ 0.2037
    with patch("backtest.engine.compute_metrics") as mock_cm, \
         patch("backtest.engine.RLAgent") as mock_rl_class:
        mock_rl = MagicMock()
        mock_rl.predict.side_effect = [1, 1, 2]  # BUY, BUY, SELL
        mock_rl_class.return_value = mock_rl
        mock_cm.return_value = {"sharpe": 0.0, "total_return": 0.0}
        df = _make_df([1, 1, 2], [100.0, 120.0, 130.0]).drop(columns=["_action"])
        from backtest.engine import run_backtest
        run_backtest(df, initial_balance=1000.0)
        trades = mock_cm.call_args[0][1]
        sell = next(t for t in trades if t["action"] == "SELL")
        expected_avg = 360.0 / (2.0 + 160.0 / 120.0)
        expected_pnl = (130.0 - expected_avg) / expected_avg
        assert abs(sell["pnl_pct"] - expected_pnl) < 1e-4


# --- NaN row skipping ---

def test_nan_rows_are_skipped_no_trades():
    # First row has NaN features → should be skipped, no BUY executed
    with patch("backtest.engine.compute_metrics") as mock_cm, \
         patch("backtest.engine.RLAgent") as mock_rl_class:
        mock_rl = MagicMock()
        mock_rl.predict.return_value = 1  # always BUY (should never fire on NaN row)
        mock_rl_class.return_value = mock_rl
        mock_cm.return_value = {"sharpe": 0.0, "total_return": 0.0}
        df = _make_df_with_nan([100.0, 110.0]).drop(columns=["_action"])
        from backtest.engine import run_backtest
        run_backtest(df, initial_balance=1000.0)
        trades = mock_cm.call_args[0][1]
        # No trade should be from the NaN row (step 0 is skipped)
        buy_steps = [t["step"] for t in trades if t["action"] == "BUY"]
        assert 0 not in buy_steps


def test_nan_rows_still_tracked_in_portfolio_values():
    with patch("backtest.engine.compute_metrics") as mock_cm, \
         patch("backtest.engine.RLAgent") as mock_rl_class:
        mock_rl = MagicMock()
        mock_rl.predict.return_value = 0  # HOLD
        mock_rl_class.return_value = mock_rl
        mock_cm.return_value = {"sharpe": 0.0, "total_return": 0.0}
        df = _make_df_with_nan([100.0, 110.0]).drop(columns=["_action"])
        from backtest.engine import run_backtest
        run_backtest(df, initial_balance=1000.0)
        portfolio_values = mock_cm.call_args[0][0]
        assert len(portfolio_values) == 2  # one entry per row, including skipped NaN row
