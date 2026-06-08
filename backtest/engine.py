"""Backtesting engine — replays historical data through the full strategy stack."""
import pandas as pd
import numpy as np
from loguru import logger
from bot.strategy.features import compute_features, FEATURE_COLS
from bot.strategy.regime_classifier import RegimeClassifier
from bot.strategy.rl_agent import RLAgent, TradingEnv
from backtest.metrics import compute_metrics
from config import INITIAL_CAPITAL, MAX_POSITION_PCT, STOP_LOSS_PCT


def run_backtest(df: pd.DataFrame, initial_balance: float = INITIAL_CAPITAL) -> dict:
    """Run a full backtest on a feature-engineered OHLCV DataFrame."""
    df = compute_features(df.copy())
    regime_clf = RegimeClassifier()
    rl_agent = RLAgent()

    balance = initial_balance
    shares = 0.0
    entry_price = 0.0  # tracks the actual position entry price
    portfolio_values = []
    trades = []

    for i, (idx, row) in enumerate(df.iterrows()):
        price = float(row["close"])
        regime_code = regime_clf.predict(row)

        # Stop-loss check
        if shares > 0 and entry_price > 0:
            pnl_pct = (price - entry_price) / entry_price
            if pnl_pct <= -STOP_LOSS_PCT:
                balance += shares * price
                trades.append({"step": i, "action": "SELL_STOP", "price": price, "pnl_pct": pnl_pct})
                shares = 0.0
                entry_price = 0.0

        obs = np.concatenate([
            row[FEATURE_COLS].values.astype(np.float32),
            [balance / initial_balance, shares / 100.0, float(regime_code)],
        ])
        action = rl_agent.predict(obs)

        if action == 1 and balance > 1:
            spend = balance * MAX_POSITION_PCT
            shares += spend / price
            balance -= spend
            entry_price = price  # record position entry price
            trades.append({"step": i, "action": "BUY", "price": price, "pnl_pct": 0.0})
        elif action == 2 and shares > 0:
            proceeds = shares * price
            pnl_pct = (price - entry_price) / entry_price if entry_price > 0 else 0.0
            balance += proceeds
            trades.append({"step": i, "action": "SELL", "price": price, "pnl_pct": pnl_pct})
            shares = 0.0
            entry_price = 0.0

        portfolio_values.append(balance + shares * price)

    final_value = portfolio_values[-1] if portfolio_values else initial_balance
    metrics = compute_metrics(portfolio_values, trades, initial_balance)
    logger.info(f"Backtest complete — final=${final_value:.2f}, return={metrics['total_return']:.2%}, sharpe={metrics['sharpe']:.2f}")
    return metrics
