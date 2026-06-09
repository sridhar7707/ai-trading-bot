"""Backtesting engine — replays historical data through the full strategy stack."""
import pandas as pd
import numpy as np
from loguru import logger
from bot.strategy.features import compute_features, FEATURE_COLS
from bot.strategy.regime_classifier import RegimeClassifier
from bot.strategy.rl_agent import RLAgent, TradingEnv
from backtest.metrics import compute_metrics
from config import (
    INITIAL_CAPITAL, MAX_POSITION_PCT, STOP_LOSS_PCT,
    ATR_STOP_MULTIPLIER, ATR_TRAIL_MULTIPLIER,
    ATR_MIN_STOP_PCT, ATR_MAX_STOP_PCT,
)


def _atr_stop_price(entry_price: float, atr: float) -> float:
    """Mirrors live RiskManager.check_stop_loss() — ATR-based stop with pct clamp."""
    if atr <= 0 or entry_price <= 0:
        return entry_price * (1 - STOP_LOSS_PCT)
    stop_pct = max(ATR_MIN_STOP_PCT, min(ATR_MAX_STOP_PCT,
                                          (ATR_STOP_MULTIPLIER * atr) / entry_price))
    return entry_price * (1 - stop_pct)


def _trail_price(high_water_mark: float, atr: float) -> float:
    """Mirrors live RiskManager.check_trailing_stop()."""
    return high_water_mark - ATR_TRAIL_MULTIPLIER * atr


def run_backtest(df: pd.DataFrame, initial_balance: float = INITIAL_CAPITAL) -> dict:
    """Run a full backtest on a feature-engineered OHLCV DataFrame."""
    df = compute_features(df.copy())
    regime_clf = RegimeClassifier()
    rl_agent   = RLAgent()

    balance          = initial_balance
    shares           = 0.0
    total_cost       = 0.0
    entry_price      = 0.0
    high_water_mark  = 0.0
    portfolio_values = []
    trades           = []

    for i, (idx, row) in enumerate(df.iterrows()):
        price = float(row["close"])
        atr   = float(row.get("atr", 0) or 0)

        if np.isnan(row[FEATURE_COLS].values).any():
            portfolio_values.append(balance + shares * price)
            continue

        regime_code = regime_clf.predict(row)

        # ── Exit checks ───────────────────────────────────────────────────────
        if shares > 0 and entry_price > 0:
            high_water_mark = max(high_water_mark, price)

            # ATR stop-loss (with flat fallback when ATR is zero)
            stop_px = _atr_stop_price(entry_price, atr)
            if price <= stop_px:
                pnl_pct = (price - entry_price) / entry_price
                balance += shares * price
                trades.append({"step": i, "action": "SELL_STOP", "price": price, "pnl_pct": pnl_pct})
                shares = high_water_mark = entry_price = total_cost = 0.0
                portfolio_values.append(balance)
                continue

            # Trailing stop (only after meaningful gain)
            if high_water_mark > entry_price * 1.005 and atr > 0:
                trail_px = _trail_price(high_water_mark, atr)
                if price <= trail_px:
                    pnl_pct = (price - entry_price) / entry_price
                    balance += shares * price
                    trades.append({"step": i, "action": "SELL_TRAIL", "price": price, "pnl_pct": pnl_pct})
                    shares = high_water_mark = entry_price = total_cost = 0.0
                    portfolio_values.append(balance)
                    continue

            # Take-profit: 3×ATR or 6%, capped at 8%
            if entry_price > 0:
                tp_pct = max(0.06, min(0.08, (3 * atr) / entry_price)) if atr > 0 else 0.06
                current_pnl = (price - entry_price) / entry_price
                if current_pnl >= tp_pct:
                    balance += shares * price
                    trades.append({"step": i, "action": "SELL_TP", "price": price, "pnl_pct": current_pnl})
                    shares = high_water_mark = entry_price = total_cost = 0.0
                    portfolio_values.append(balance)
                    continue

        # ── Signal ───────────────────────────────────────────────────────────
        obs = np.concatenate([
            row[FEATURE_COLS].values.astype(np.float32),
            [balance / initial_balance, shares / 100.0, float(regime_code)],
        ])
        action = rl_agent.predict(obs)

        if action == 1 and balance > 1:
            spend = balance * MAX_POSITION_PCT
            shares += spend / price
            balance -= spend
            total_cost += spend
            entry_price = total_cost / shares
            high_water_mark = price
            trades.append({"step": i, "action": "BUY", "price": price, "pnl_pct": 0.0})

        elif action == 2 and shares > 0:
            pnl_pct = (price - entry_price) / entry_price if entry_price > 0 else 0.0
            balance += shares * price
            trades.append({"step": i, "action": "SELL", "price": price, "pnl_pct": pnl_pct})
            shares = high_water_mark = entry_price = total_cost = 0.0

        portfolio_values.append(balance + shares * price)

    final_value = portfolio_values[-1] if portfolio_values else initial_balance
    metrics = compute_metrics(portfolio_values, trades, initial_balance)
    logger.info(
        f"Backtest complete — final=${final_value:.2f}, "
        f"return={metrics['total_return']:.2%}, sharpe={metrics['sharpe']:.2f}"
    )
    return metrics
