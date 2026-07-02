import numpy as np
from typing import List


def compute_metrics(portfolio_values: List[float], trades: list, initial_balance: float) -> dict:
    values  = np.array(portfolio_values, dtype=float)
    returns = np.diff(values) / (values[:-1] + 1e-8)

    total_return = (values[-1] - initial_balance) / initial_balance
    trading_days = max(len(values) / 78, 1)  # 78 five-min bars per trading day
    ann_return   = (1 + total_return) ** (252 / trading_days) - 1
    sharpe       = (float(np.mean(returns)) / (float(np.std(returns)) + 1e-8)
                    * np.sqrt(252 * 78)) if len(returns) > 1 else 0.0  # 78 five-min bars/day
    downside     = returns[returns < 0]
    down_std     = float(np.std(downside)) if len(downside) > 1 else 0.0
    sortino      = (float(np.mean(returns)) / (down_std + 1e-8)
                    * np.sqrt(252 * 78)) if len(returns) > 1 else 0.0

    max_dd   = _max_drawdown(values)
    calmar   = ann_return / (max_dd + 1e-8) if max_dd > 0 else 0.0

    sell_trades  = [t for t in trades if "SELL" in t.get("action", "")]
    win_trades   = [t for t in sell_trades if t.get("pnl_pct", 0) > 0]
    loss_trades  = [t for t in sell_trades if t.get("pnl_pct", 0) < 0]
    win_rate     = len(win_trades) / len(sell_trades) if sell_trades else 0.0
    gross_profit = sum(t["pnl_pct"] for t in win_trades)
    gross_loss   = abs(sum(t["pnl_pct"] for t in loss_trades))
    profit_factor = gross_profit / (gross_loss + 1e-8)

    avg_win  = float(np.mean([t["pnl_pct"] for t in win_trades]))  if win_trades  else 0.0
    avg_loss = float(np.mean([t["pnl_pct"] for t in loss_trades])) if loss_trades else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    return {
        "total_return":   total_return,
        "ann_return":     ann_return,
        "sharpe":         sharpe,
        "sortino":        sortino,
        "calmar":         calmar,
        "max_drawdown":   max_dd,
        "profit_factor":  profit_factor,
        "win_rate":       win_rate,
        "expectancy":     expectancy,
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "num_trades":     len(trades),
        "num_wins":       len(win_trades),
        "num_losses":     len(loss_trades),
        "final_value":    float(values[-1]),
    }


def _max_drawdown(values: np.ndarray) -> float:
    peak  = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / (peak + 1e-8)
        if dd > max_dd:
            max_dd = dd
    return max_dd
