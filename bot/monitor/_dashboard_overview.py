"""Overview and status functions extracted from dashboard_data.py."""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from config import DAILY_LOSS_LIMIT_PCT, WEEKLY_LOSS_LIMIT_PCT
from bot.monitor._dashboard_state import _last_sync

_TEXT   = "#111827"
_MUTED  = "#6b7280"
_FONT   = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
_EMPTY_HINT = ("The bot trades at market open (9:30 AM ET, Mon–Fri). "
               "Data appears here after the first cycle of the day.")


def _con():
    import bot.monitor.dashboard_data as _dd
    db = _dd._DB
    if not Path(db).exists():
        return None
    return sqlite3.connect(db, check_same_thread=False)


def _latest_portfolio_value(con) -> float:
    """Most recent portfolio value from either a trade row or a heartbeat snapshot."""
    candidates: list[tuple[str, float]] = []
    try:
        t = con.execute(
            "SELECT timestamp, portfolio_value FROM trades ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if t and t[1] is not None:
            candidates.append((t[0], float(t[1])))
    except Exception:
        pass
    try:
        s = con.execute(
            "SELECT timestamp, portfolio_value FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if s and s[1] is not None:
            candidates.append((s[0], float(s[1])))
    except Exception:
        pass
    return max(candidates, key=lambda r: r[0])[1] if candidates else 0.0


def _inception(con) -> tuple[float, str | None]:
    """Earliest (portfolio_value, timestamp) across trades + snapshots."""
    candidates: list[tuple[str, float]] = []
    for table in ("trades", "portfolio_snapshots"):
        try:
            r = con.execute(
                f"SELECT timestamp, portfolio_value FROM {table} "
                f"WHERE portfolio_value IS NOT NULL ORDER BY timestamp LIMIT 1"
            ).fetchone()
            if r and r[1] is not None:
                candidates.append((r[0], float(r[1])))
        except Exception:
            pass
    if not candidates:
        return 0.0, None
    ts, val = min(candidates, key=lambda r: r[0])
    return val, ts


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    return f"{int(seconds / 3600)}h ago"


def _money(x: float) -> str:
    """Signed dollar amount, e.g. +$480.00 / -$120.50."""
    return f"{'+' if x >= 0 else '-'}${abs(x):,.2f}"


def _provenance_line(d: dict) -> str:
    """One-line credibility caption: paper-trading, since when, how many trades."""
    n = d.get("total_trades", 0)
    inc = d.get("inception_date")
    since = ""
    if inc:
        try:
            since = " since " + pd.to_datetime(inc).strftime("%b %d, %Y")
        except Exception:
            pass
    return f"📄 **Paper trading{since}** · {n} trade{'s' if n != 1 else ''} executed (simulated capital — no real money)"


def get_overview() -> dict:
    con = _con()
    if con is None:
        return {"_error": "trades.db not found — no trading data yet."}
    today = date.today().isoformat()
    wk    = date.today().strftime("%G-W%V")
    try:
        rs = {r[0]: r[1] for r in con.execute("SELECT key, value FROM risk_state")}
    except Exception:
        rs = {}
    try:
        mc = {r[0]: float(r[1]) for r in con.execute("SELECT key, value FROM macro_cache")}
    except Exception:
        mc = {}

    portfolio = _latest_portfolio_value(con)

    daily_start  = float(rs.get("daily_start_value",  0) or 0)
    weekly_start = float(rs.get("weekly_start_value", 0) or 0)
    is_today     = rs.get("daily_start_date")  == today
    is_this_week = rs.get("weekly_start_week") == wk

    day_pnl  = (portfolio - daily_start)  / daily_start  if is_today  and daily_start  else 0.0
    week_pnl = (portfolio - weekly_start) / weekly_start if is_this_week and weekly_start else 0.0
    day_pnl_dollars  = (portfolio - daily_start)  if is_today  and daily_start  else 0.0
    week_pnl_dollars = (portfolio - weekly_start) if is_this_week and weekly_start else 0.0

    inception_value, inception_date = _inception(con)
    total_return = (portfolio - inception_value) / inception_value if inception_value else 0.0

    try:
        day_trade_dates = json.loads(rs.get("day_trade_dates", "[]"))
    except Exception:
        day_trade_dates = []

    try:
        trades_today = con.execute(
            "SELECT COUNT(*) FROM trades WHERE timestamp LIKE ?", (today + "%",)
        ).fetchone()[0]
    except Exception:
        trades_today = 0
    try:
        open_positions = con.execute("SELECT COUNT(*) FROM position_state").fetchone()[0]
    except Exception:
        open_positions = 0
    try:
        total_trades = con.execute(
            "SELECT COUNT(*) FROM trades WHERE action='BUY' OR action LIKE 'SELL%'"
        ).fetchone()[0]
    except Exception:
        total_trades = 0
    con.close()

    sync_age_s: float | None = None
    sync_ok: bool | None = _last_sync.get("ok")
    if _last_sync.get("ts"):
        sync_age_s = (datetime.now(timezone.utc) - _last_sync["ts"]).total_seconds()

    db_mtime_s: float | None = None
    try:
        db_mtime_s = (datetime.now(timezone.utc) - datetime.fromtimestamp(
            Path(_DB).stat().st_mtime, tz=timezone.utc
        )).total_seconds()
    except Exception:
        pass

    return {
        "portfolio":        portfolio,
        "day_pnl":          day_pnl,
        "week_pnl":         week_pnl,
        "day_pnl_dollars":  day_pnl_dollars,
        "week_pnl_dollars": week_pnl_dollars,
        "total_return":     total_return,
        "inception_date":   inception_date,
        "total_trades":     total_trades,
        "spy_return":       None,
        "trades_today":     trades_today,
        "open_positions":   open_positions,
        "day_trades_used":  day_trade_dates.count(today),
        "macro_score":      mc.get("score", 0.5),
        "macro_halt":       bool(mc.get("halt", 0)),
        "emergency_halt":   Path("data/HALT_TRADING").exists(),
        "daily_limit_hit":  is_today  and daily_start  and day_pnl  <= -DAILY_LOSS_LIMIT_PCT,
        "weekly_limit_hit": is_this_week and weekly_start and week_pnl <= -WEEKLY_LOSS_LIMIT_PCT,
        "sync_ok":          sync_ok,
        "sync_age_s":       sync_age_s,
        "sync_err":         _last_sync.get("err", ""),
        "db_age_s":         db_mtime_s,
    }


def overview_md(d: dict) -> str:
    if "_error" in d:
        return f"⚠️ {d['_error']}\n\n_{_EMPTY_HINT}_"

    status = "🔴 EMERGENCY HALT" if d["emergency_halt"] else (
             "🔴 VIX HALT"        if d["macro_halt"]      else (
             "🔴 DAILY LIMIT HIT" if d["daily_limit_hit"] else (
             "🟡 WEEKLY LIMIT"    if d["weekly_limit_hit"] else "🟢 ACTIVE")))

    sync_ok  = d.get("sync_ok")
    db_age   = d.get("db_age_s")
    sync_err = d.get("sync_err", "")
    age_str  = _fmt_age(db_age)
    if sync_ok is True:
        sync_line = f"🟢 Data last updated {age_str}"
    elif sync_ok is False:
        err_hint = f" ({sync_err})" if sync_err else ""
        sync_line = f"🔴 Data update failed{err_hint} · last updated {age_str}"
    else:
        sync_line = f"⚪ Data freshness unknown · last updated {age_str}"

    day_str  = f"{_money(d.get('day_pnl_dollars', 0))} ({d['day_pnl']:+.2%})"
    week_str = f"{_money(d.get('week_pnl_dollars', 0))} ({d['week_pnl']:+.2%})"

    bot_ret = d.get("total_return", 0.0)
    spy_ret = d.get("spy_return")
    if spy_ret is not None:
        verdict = "🟢 beating" if bot_ret > spy_ret else "🔴 trailing"
        vs_spy  = f"{bot_ret:+.2%} vs S&P {spy_ret:+.2%} — {verdict} the market"
    else:
        inc_date = d.get("inception_date")
        inc_str  = ""
        if inc_date:
            try:
                inc_str = " since " + pd.to_datetime(inc_date).strftime("%b %d, %Y")
            except Exception:
                pass
        vs_spy = f"{bot_ret:+.2%}{inc_str}"

    macro = d['macro_score']
    macro_label = "favorable" if macro >= 0.65 else ("cautious" if macro <= 0.35 else "neutral")
    macro_str = f"{macro:.2f} — {macro_label}"

    return (
        f"### Bot Status: {status}\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Portfolio Value | **${d['portfolio']:,.2f}** |\n"
        f"| Day P&L | **{day_str}** |\n"
        f"| Week P&L | **{week_str}** |\n"
        f"| Return vs S&P 500 | **{vs_spy}** |\n"
        f"| Trades Today | {d['trades_today']} |\n"
        f"| Open Positions | {d['open_positions']} |\n"
        f"| Day Trades | {d['day_trades_used']}/3 |\n"
        f"| Market Conditions | {macro_str} |\n\n"
        f"{_provenance_line(d)}\n\n"
        f"*{sync_line}*\n\n"
        f"> 💡 **Day/Week P&L** = total account change since today's/Monday's open. "
        f"**P&L in the Holdings table** = each position's gain since its entry date — "
        f"these won't add up to Day P&L because most capital sits in cash.\n"
    )
