"""Performance metrics functions extracted from dashboard_data.py."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

_EMPTY_HINT = ("The bot trades at market open (9:30 AM ET, Mon–Fri). "
               "Data appears here after the first cycle of the day.")

_MIN_SHARPE_OBS  = 20
_MIN_SHARPE_DAYS = 20


def _con():
    import bot.monitor.dashboard_data as _dd
    db = _dd._DB
    if not Path(db).exists():
        return None
    return sqlite3.connect(db, check_same_thread=False)


def get_performance_metrics(days: int = 60) -> dict:
    con = _con()
    if con is None:
        return {}
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    _empty = {"sharpe": None, "sortino": None, "win_rate": 0.0, "max_drawdown": 0.0,
              "total_return": 0.0, "trade_count": 0, "closed_trades": 0,
              "avg_win": 0.0, "avg_loss": 0.0, "calmar": None, "alpha": None}

    try:
        pv_rows = con.execute(
            "SELECT timestamp, portfolio_value FROM trades WHERE timestamp >= ? "
            "UNION ALL "
            "SELECT timestamp, portfolio_value FROM portfolio_snapshots WHERE timestamp >= ? "
            "ORDER BY timestamp",
            (since, since),
        ).fetchall()
    except Exception:
        pv_rows = con.execute(
            "SELECT timestamp, portfolio_value FROM trades WHERE timestamp >= ? ORDER BY timestamp",
            (since,),
        ).fetchall()

    sells = con.execute(
        "SELECT pnl_pct FROM trades WHERE action LIKE 'SELL%' AND timestamp >= ?", (since,)
    ).fetchall()
    trade_count = con.execute(
        "SELECT COUNT(*) FROM trades WHERE timestamp >= ?", (since,)
    ).fetchone()[0]
    con.close()

    if not pv_rows:
        return _empty
    vals = np.array([r[1] for r in pv_rows if r[1] is not None], dtype=float)
    if len(vals) == 0:
        return _empty
    rets   = np.diff(vals) / (vals[:-1] + 1e-8)
    distinct_days = len({ts[:10] for ts, _ in pv_rows if ts})
    qualified = len(rets) >= _MIN_SHARPE_OBS and distinct_days >= _MIN_SHARPE_DAYS
    if qualified and np.std(rets) > 0:
        sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252 * 78))
    else:
        sharpe = None
    downside = rets[rets < 0]
    down_std = float(np.std(downside)) if len(downside) > 1 else 0.0
    if qualified and down_std > 0:
        sortino = float(np.mean(rets) / down_std * np.sqrt(252 * 78))
    else:
        sortino = None
    peak = vals[0]; max_dd = 0.0
    for v in vals:
        peak   = max(peak, v)
        max_dd = max(max_dd, (peak - v) / (peak + 1e-8))
    total_return = float((vals[-1] - vals[0]) / (vals[0] + 1e-8))
    ann_return   = (1 + total_return) ** (252 / max(distinct_days, 1)) - 1
    calmar = round(ann_return / (max_dd + 1e-8), 2) if max_dd > 0 else None
    pnl_values = [r[0] for r in sells if r[0] is not None]
    wins_pnl   = [p for p in pnl_values if p > 0]
    losses_pnl = [p for p in pnl_values if p <= 0]
    closed   = len(sells)
    win_rate = len(wins_pnl) / closed if closed else 0.0
    avg_win  = float(np.mean(wins_pnl))   if wins_pnl   else 0.0
    avg_loss = float(np.mean(losses_pnl)) if losses_pnl else 0.0
    since_day = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    # Lazy import avoids circular dependency — dashboard_data is loaded before this is called
    from bot.monitor.dashboard_data import spy_return_since as _spy_return_since
    spy_ret  = _spy_return_since(since_day)
    alpha    = round(total_return - spy_ret, 4) if spy_ret is not None else None
    return {
        "sharpe":        round(sharpe, 2) if sharpe is not None else None,
        "sortino":       round(sortino, 2) if sortino is not None else None,
        "win_rate":      round(win_rate, 4),
        "avg_win":       round(avg_win, 4),
        "avg_loss":      round(avg_loss, 4),
        "max_drawdown":  round(max_dd, 4),
        "calmar":        calmar,
        "alpha":         alpha,
        "total_return":  round(total_return, 4),
        "trade_count":   trade_count,
        "closed_trades": closed,
    }


def performance_md(m: dict) -> str:
    if not m:
        return f"No performance data yet. {_EMPTY_HINT}"
    closed = m.get("closed_trades", 0)
    win_str    = f"{m['win_rate']:.1%}" if closed else "n/a (no closed trades yet)"
    sharpe_str = f"{m['sharpe']:.2f}"   if m.get("sharpe")  is not None else "n/a (need more history)"
    sortino_str= f"{m['sortino']:.2f}"  if m.get("sortino") is not None else "n/a (need more history)"
    calmar_str = f"{m['calmar']:.2f}"   if m.get("calmar")  is not None else "n/a"
    alpha_str  = f"{m['alpha']:+.2%}"   if m.get("alpha")   is not None else "n/a (no benchmark data)"
    avg_win_str  = f"{m.get('avg_win', 0):+.2%}"  if closed else "n/a"
    avg_loss_str = f"{m.get('avg_loss', 0):+.2%}" if closed else "n/a"
    return (
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Sharpe Ratio | **{sharpe_str}** |\n"
        f"| Sortino Ratio | **{sortino_str}** |\n"
        f"| Win Rate | **{win_str}** |\n"
        f"| Avg Win / Avg Loss | **{avg_win_str} / {avg_loss_str}** |\n"
        f"| Max Drawdown | **{m['max_drawdown']:.1%}** |\n"
        f"| Calmar Ratio | **{calmar_str}** |\n"
        f"| Alpha vs S&P 500 | **{alpha_str}** |\n"
        f"| Total Return | **{m['total_return']:+.2%}** |\n"
        f"| Trades Analysed | {m['trade_count']} ({closed} closed) |"
    )
