"""Check if the bot is ready to graduate from paper trading to real money."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import sqlite3
import numpy as np
from loguru import logger
from config import TRADE_DB_PATH, BENCHMARK
import yfinance as yf
from datetime import datetime, timedelta
import bot.monitor.telegram_bot as tg

THRESHOLDS = {
    "min_days": 60,
    "min_win_rate": 0.52,
    "min_sharpe": 1.0,
    "max_drawdown": 0.15,
    "max_consecutive_losing_days": 4,
}


def load_trades():
    con = sqlite3.connect(TRADE_DB_PATH)
    try:
        rows = con.execute(
            "SELECT timestamp, action, pnl_pct, portfolio_value FROM trades ORDER BY timestamp"
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        con.close()
    return rows


def _daily_close_values(trades: list) -> list:
    """One portfolio snapshot per trading weekday (last trade of day), forward-filled for no-trade days.
    Quiet days contribute zero return rather than being excluded — prevents Sharpe inflation.
    """
    import datetime as dt
    daily: dict[str, float] = {}
    for ts, action, pnl_pct, pv in trades:
        daily[ts[:10]] = pv  # last trade of each day wins
    if not daily:
        return []
    keys = sorted(daily.keys())
    start_d = dt.date.fromisoformat(keys[0])
    end_d   = dt.date.fromisoformat(keys[-1])
    result, prev_val, d = [], daily[keys[0]], start_d
    while d <= end_d:
        ds = d.isoformat()
        if ds in daily:
            prev_val = daily[ds]
        if d.weekday() < 5:  # Mon–Fri only
            result.append(prev_val)
        d += dt.timedelta(days=1)
    return result


def compute_sharpe(values: list) -> float:
    v = np.array(values)
    if len(v) < 2:
        return 0.0
    returns = np.diff(v) / (v[:-1] + 1e-8)
    # Daily returns → annualise with sqrt(252)
    return float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252))


def compute_max_drawdown(values):
    values = np.array(values)
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def run_check():
    trades = load_trades()
    if not trades:
        logger.warning("No trades found. Run paper trading first.")
        return False

    timestamps = [t[0] for t in trades]
    start = datetime.fromisoformat(timestamps[0])
    end = datetime.fromisoformat(timestamps[-1])
    days_trading = (end - start).days

    sell_trades = [t for t in trades if t[1].startswith("SELL")]
    wins = sum(1 for t in sell_trades if t[2] > 0)
    win_rate = wins / len(sell_trades) if sell_trades else 0.0

    # Use daily snapshots for Sharpe/drawdown/streaks — trade-row values inflate Sharpe
    # by excluding quiet (no-trade) days from the denominator.
    daily_values = _daily_close_values(trades)
    sharpe = compute_sharpe(daily_values)
    max_dd = compute_max_drawdown(daily_values)

    daily_arr = np.array(daily_values)
    if len(daily_arr) >= 2:
        daily_returns = (np.diff(daily_arr) / (daily_arr[:-1] + 1e-8)).tolist()
    else:
        daily_returns = []
    max_consec_loss = _max_consecutive_losses(daily_returns)

    # vs S&P 500 over the same period
    bot_return = (daily_values[-1] - daily_values[0]) / daily_values[0] if daily_values and daily_values[0] else 0.0
    spy_return = 0.0
    try:
        spy = yf.download("SPY", start=start.date(), end=end.date(), progress=False, auto_adjust=True)
        if len(spy) > 1:
            spy_return = float((spy["Close"].iloc[-1] - spy["Close"].iloc[0]) / spy["Close"].iloc[0])
    except Exception:
        pass

    results = {
        "Days trading":          (days_trading, THRESHOLDS["min_days"], days_trading >= THRESHOLDS["min_days"]),
        "Win rate":              (f"{win_rate:.1%}", f"{THRESHOLDS['min_win_rate']:.1%}", win_rate >= THRESHOLDS["min_win_rate"]),
        "Sharpe ratio":          (f"{sharpe:.2f}", f"{THRESHOLDS['min_sharpe']:.2f}", sharpe >= THRESHOLDS["min_sharpe"]),
        "Max drawdown":          (f"{max_dd:.1%}", f"{THRESHOLDS['max_drawdown']:.1%}", max_dd <= THRESHOLDS["max_drawdown"]),
        "Max consec losing days":(max_consec_loss, THRESHOLDS["max_consecutive_losing_days"], max_consec_loss <= THRESHOLDS["max_consecutive_losing_days"]),
        "vs S&P 500":            (f"{bot_return:.1%} vs {spy_return:.1%}", "Outperforming", bot_return > spy_return),
    }

    all_pass = True
    logger.info("=== CONFIDENCE CHECK ===")
    for metric, (value, threshold, passed) in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {status}  {metric}: {value} (min: {threshold})")
        if not passed:
            all_pass = False

    if all_pass:
        logger.info("ALL CHECKS PASSED — Bot is ready for real money.")
        tg.alert_confidence_passed()
    else:
        logger.warning("NOT READY — Keep paper trading.")
    return all_pass


def _max_consecutive_losses(day_returns):
    max_streak = current = 0
    for r in day_returns:
        if r < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


if __name__ == "__main__":
    run_check()
