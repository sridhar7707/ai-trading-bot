"""Compute actual weekly performance stats and send Telegram report."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from loguru import logger
from config import TRADE_DB_PATH
import bot.monitor.telegram_bot as tg


def compute_weekly_report():
    con = sqlite3.connect(TRADE_DB_PATH)
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    rows = con.execute(
        "SELECT timestamp, action, pnl_pct, portfolio_value FROM trades WHERE timestamp >= ? ORDER BY timestamp",
        (week_ago,),
    ).fetchall()
    con.close()

    if not rows:
        logger.warning("No trades in the last 7 days — sending zeroed report.")
        tg.alert_weekly_report(0.0, 0.0, 0.0, 0.0, 0.0)
        return

    # Build daily snapshots (last trade per day, forward-fill no-trade days)
    # so Sharpe/drawdown aren't inflated by excluding quiet days.
    import datetime as dt
    _daily: dict[str, float] = {}
    for ts, action, pnl_pct, pv in rows:
        _daily[ts[:10]] = pv
    _keys = sorted(_daily.keys())
    daily_values: list[float] = []
    if _keys:
        _d = dt.date.fromisoformat(_keys[0])
        _end = dt.date.fromisoformat(_keys[-1])
        _prev = _daily[_keys[0]]
        while _d <= _end:
            _ds = _d.isoformat()
            if _ds in _daily:
                _prev = _daily[_ds]
            if _d.weekday() < 5:
                daily_values.append(_prev)
            _d += dt.timedelta(days=1)

    start_val = daily_values[0] if daily_values else rows[0][3]
    end_val   = daily_values[-1] if daily_values else rows[-1][3]
    week_return = (end_val - start_val) / start_val if start_val else 0.0

    dv = np.array(daily_values) if daily_values else np.array([r[3] for r in rows])
    ret = np.diff(dv) / (dv[:-1] + 1e-8)
    sharpe = float(np.mean(ret) / (np.std(ret) + 1e-8) * np.sqrt(252)) if len(ret) > 1 else 0.0

    peak, max_dd = dv[0], 0.0
    for v in dv:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    sell_trades = [r for r in rows if r[1].startswith("SELL")]
    win_rate = sum(1 for r in sell_trades if r[2] > 0) / len(sell_trades) if sell_trades else 0.0

    # SPY return over the same window
    start_ts = datetime.fromisoformat(rows[0][0])
    end_ts = datetime.fromisoformat(rows[-1][0])
    spy_return = 0.0
    try:
        spy = yf.download("SPY", start=start_ts.date(), end=end_ts.date(), progress=False, auto_adjust=True)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = [col[0] for col in spy.columns]
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
