"""Compute actual weekly performance stats and send Telegram report."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from loguru import logger
from config import TRADE_DB_PATH
import bot.monitor.telegram_bot as tg


def compute_weekly_report():
    con = sqlite3.connect(TRADE_DB_PATH)
    week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
    rows = con.execute(
        "SELECT timestamp, action, pnl_pct, portfolio_value FROM trades WHERE timestamp >= ? ORDER BY timestamp",
        (week_ago,),
    ).fetchall()
    con.close()

    if not rows:
        logger.warning("No trades in the last 7 days — sending zeroed report.")
        tg.alert_weekly_report(0.0, 0.0, 0.0, 0.0, 0.0)
        return

    portfolio_values = [r[3] for r in rows]
    start_val, end_val = portfolio_values[0], portfolio_values[-1]
    week_return = (end_val - start_val) / start_val if start_val else 0.0

    values = np.array(portfolio_values)
    returns = np.diff(values) / values[:-1]
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)) if len(returns) > 1 else 0.0

    peak, max_dd = values[0], 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    sell_trades = [r for r in rows if r[1] in ("SELL", "SELL_STOP")]
    win_rate = sum(1 for r in sell_trades if r[2] > 0) / len(sell_trades) if sell_trades else 0.0

    # SPY return over the same window
    start_ts = datetime.fromisoformat(rows[0][0])
    end_ts = datetime.fromisoformat(rows[-1][0])
    spy_return = 0.0
    try:
        spy = yf.download("SPY", start=start_ts.date(), end=end_ts.date(), progress=False, auto_adjust=True)
        if len(spy) > 1:
            spy_return = float((spy["Close"].iloc[-1] - spy["Close"].iloc[0]) / spy["Close"].iloc[0])
    except Exception as e:
        logger.warning(f"Could not fetch SPY data: {e}")

    tg.alert_weekly_report(week_return, spy_return, win_rate, sharpe, max_dd)
    logger.info(
        f"Weekly report sent — return={week_return:.2%}, vs SPY={spy_return:.2%}, "
        f"win_rate={win_rate:.1%}, sharpe={sharpe:.2f}, max_dd={max_dd:.2%}"
    )


if __name__ == "__main__":
    compute_weekly_report()
