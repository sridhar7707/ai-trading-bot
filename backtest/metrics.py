import numpy as np
from typing import List


def compute_metrics(portfolio_values: List[float], trades: list, initial_balance: float) -> dict:
    values = np.array(portfolio_values)
    returns = np.diff(values) / values[:-1]

    total_return = (values[-1] - initial_balance) / initial_balance
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)) if len(returns) > 1 else 0.0
    max_drawdown = _max_drawdown(values)

    sell_trades = [t for t in trades if t["action"] in ("SELL", "SELL_STOP")]
    win_rate = sum(1 for t in sell_trades if t["pnl_pct"] > 0) / len(sell_trades) if sell_trades else 0.0

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "num_trades": len(trades),
        "final_value": float(values[-1]),
    }


def _max_drawdown(values: np.ndarray) -> float:
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd
