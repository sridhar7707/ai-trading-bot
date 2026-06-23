"""Read-only data layer for the trading dashboard."""
from __future__ import annotations
import json
import sqlite3
import sys
import os
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TRADE_DB_PATH, DAILY_LOSS_LIMIT_PCT, WEEKLY_LOSS_LIMIT_PCT

_DB   = TRADE_DB_PATH

# ── Unified light theme ───────────────────────────────────────────────────────
_BG     = "#ffffff"
_CARD   = "#f7f8fa"
_TEXT   = "#111827"
_MUTED  = "#6b7280"
_GRID   = "#e5e7eb"
_ACCENT = "#0891b2"
_POS    = "#15803d"
_NEG    = "#dc2626"
_FONT   = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

_EMPTY_HINT = ("The bot trades at market open (9:30 AM ET, Mon–Fri). "
               "Data appears here after the first cycle of the day.")

# Shared mutable state — imported by sub-modules via _dashboard_state
from bot.monitor._dashboard_state import _last_sync, _pull_lock, _spy_cache

_ALPACA_DATA = "https://data.alpaca.markets"


def _alpaca_headers() -> dict | None:
    """Auth headers for the Alpaca data API, or None when credentials are absent."""
    try:
        from config import ALPACA_KEY, ALPACA_SECRET
    except Exception:
        ALPACA_KEY = ALPACA_SECRET = ""
    key = ALPACA_KEY or os.environ.get("ALPACA_KEY", "")
    sec = ALPACA_SECRET or os.environ.get("ALPACA_SECRET", "")
    if not key or not sec:
        return None
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}


def _con() -> sqlite3.Connection | None:
    # Every read goes through here, so pull the latest DB first. This is the one
    # place that guarantees ALL tabs (not just Overview) see fresh data — without
    # it, tabs that render before Overview's async pull read the empty bundled DB.
    refresh_db_from_hf()
    if not Path(_DB).exists():
        logger.warning(f"dashboard_data: {_DB} not found — no data to show")
        return None
    return sqlite3.connect(_DB, check_same_thread=False)


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


def _spy_return_alpaca(start_day: str) -> float | None:
    """SPY return from Alpaca daily bars (official). None on failure."""
    headers = _alpaca_headers()
    if headers is None:
        return None
    try:
        import requests
        r = requests.get(
            f"{_ALPACA_DATA}/v2/stocks/SPY/bars",
            params={"timeframe": "1Day", "start": f"{start_day}T00:00:00Z",
                    "adjustment": "all", "limit": 10000},
            headers=headers, timeout=8,
        )
        if r.status_code != 200:
            logger.warning(f"_spy_return_alpaca: HTTP {r.status_code}")
            return None
        bars = r.json().get("bars", []) or []
        if len(bars) >= 2:
            first, last = float(bars[0]["c"]), float(bars[-1]["c"])
            return (last - first) / first if first else None
    except Exception as exc:
        logger.warning(f"_spy_return_alpaca: {exc}")
    return None


def _spy_return_yfinance(start_day: str) -> float | None:
    """SPY return from yfinance (unofficial fallback). None on failure."""
    try:
        import yfinance as yf
        spy = yf.download("SPY", start=start_day, progress=False, auto_adjust=True)
        if spy is not None and len(spy) > 1:
            first = float(spy["Close"].iloc[0])
            last  = float(spy["Close"].iloc[-1])
            return (last - first) / first if first else None
    except Exception as exc:
        logger.warning(f"_spy_return_yfinance: {exc}")
    return None


def spy_return_since(start_iso: str | None) -> float | None:
    """S&P 500 (SPY) total return since `start_iso`, best-effort."""
    if not start_iso or not os.environ.get("SPACE_ID"):
        return None
    today = date.today().isoformat()
    key = (today, start_iso)
    if _spy_cache.get("key") == key:
        return _spy_cache.get("ret")
    start_day = start_iso[:10]
    ret = _spy_return_alpaca(start_day)
    if ret is None:
        ret = _spy_return_yfinance(start_day)
    _spy_cache["key"] = key
    _spy_cache["ret"] = ret
    return ret


def refresh_db_from_hf(force: bool = False) -> None:
    """Pull a fresh trades.db from HF dataset (Space-only)."""
    if not os.environ.get("SPACE_ID"):
        return
    try:
        from bot.monitor.sync_db import pull_db
        with _pull_lock:
            ok = pull_db(force=force)
        _last_sync["ok"] = ok
        _last_sync["ts"] = datetime.now(timezone.utc)
        _last_sync["err"] = "" if ok else "pull_db returned False — check HF_TOKEN and HF_DB_REPO_ID"
        if not ok:
            logger.error("refresh_db_from_hf: DB pull returned False — dashboard may show stale data")
    except Exception as exc:
        _last_sync["ok"] = False
        _last_sync["ts"] = datetime.now(timezone.utc)
        _last_sync["err"] = str(exc)
        logger.error(f"refresh_db_from_hf: exception — {exc}")


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    return f"{int(seconds / 3600)}h ago"


def diagnostics() -> dict:
    """Log a full snapshot of the dashboard's environment and DB state."""
    import sys as _sys
    from config import HF_DB_REPO_ID, HF_TOKEN

    in_space = bool(os.environ.get("SPACE_ID"))
    token_present = bool(HF_TOKEN or os.environ.get("HF_TOKEN"))
    db_path = Path(_DB)
    db_exists = db_path.exists()

    info: dict = {
        "in_hf_space":   in_space,
        "space_id":      os.environ.get("SPACE_ID", "—"),
        "hf_token_set":  token_present,
        "hf_db_repo":    HF_DB_REPO_ID or "—",
        "db_path":       str(db_path.resolve()),
        "db_exists":     db_exists,
        "python":        _sys.version.split()[0],
    }

    logger.info("─── DASHBOARD DIAGNOSTICS ───────────────────────────────")
    logger.info(f"  Running in HF Space : {in_space}  (SPACE_ID={info['space_id']})")
    logger.info(f"  HF_TOKEN present    : {token_present}")
    logger.info(f"  HF_DB_REPO_ID       : {info['hf_db_repo']}")
    logger.info(f"  DB path             : {info['db_path']}")
    logger.info(f"  DB exists           : {db_exists}")

    if not in_space:
        logger.warning("  ⚠ SPACE_ID not set — refresh_db_from_hf() will SKIP pulling from HF.")
    if in_space and not token_present:
        logger.error("  ⚠ In a Space but HF_TOKEN is MISSING — add it as a Space secret, "
                     "otherwise the private dataset cannot be pulled and the dashboard stays at $0.00.")

    if db_exists:
        try:
            size_kb = db_path.stat().st_size / 1024
            age_s   = (datetime.now(timezone.utc) -
                       datetime.fromtimestamp(db_path.stat().st_mtime, tz=timezone.utc)).total_seconds()
            info["db_size_kb"] = round(size_kb, 1)
            info["db_age_s"]   = round(age_s)
            logger.info(f"  DB size             : {size_kb:.1f} KB")
            logger.info(f"  DB file age         : {_fmt_age(age_s)}")
        except Exception as exc:
            logger.warning(f"  Could not stat DB file: {exc}")

        con = _con()
        if con is not None:
            for table in ("trades", "position_state", "risk_state", "macro_cache", "portfolio_snapshots"):
                try:
                    n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    info[f"rows_{table}"] = n
                    logger.info(f"  rows[{table:<14}]: {n}")
                except Exception as exc:
                    info[f"rows_{table}"] = f"ERROR: {exc}"
                    logger.warning(f"  rows[{table:<14}]: table missing/unreadable — {exc}")
            try:
                row = con.execute(
                    "SELECT timestamp, symbol, action, portfolio_value "
                    "FROM trades ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if row:
                    info["latest_trade"] = row[0]
                    logger.info(f"  latest trade        : {row[0]} {row[1]} {row[2]} "
                                f"portfolio=${row[3]:,.2f}")
                else:
                    logger.warning("  latest trade        : NONE — trades table is empty, "
                                   "dashboard will show $0.00 portfolio")
            except Exception as exc:
                logger.warning(f"  latest trade        : query failed — {exc}")
            con.close()
    else:
        logger.error("  ⚠ trades.db does NOT exist locally — nothing to display. "
                     "Either the bot has never run, or the HF pull failed.")

    logger.info("──────────────────────────────────────────────────────────")
    return info


# ── Audit Trail (Institutional) ───────────────────────────────────────────────

def get_audit_df(days: int = 60) -> pd.DataFrame:
    con = _con()
    if con is None:
        return pd.DataFrame()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    df = pd.read_sql_query(
        "SELECT timestamp, symbol, action, price, notional, pnl_pct, "
        "xgb_prob, lstm_prob, sentiment_score, macro_score, ensemble_score, regime "
        "FROM trades WHERE timestamp >= ? ORDER BY timestamp DESC",
        con, params=(since,),
    )
    con.close()
    return df


# ── Compliance (Institutional) ────────────────────────────────────────────────

def get_compliance_state() -> dict:
    con = _con()
    if con is None:
        return {}
    today = date.today().isoformat()
    wk    = date.today().strftime("%G-W%V")
    try:
        rs = {r[0]: r[1] for r in con.execute("SELECT key, value FROM risk_state")}
    except Exception:
        rs = {}
    portfolio = _latest_portfolio_value(con)
    is_today     = rs.get("daily_start_date")  == today
    is_this_week = rs.get("weekly_start_week") == wk
    daily_start  = float(rs.get("daily_start_value",  0) or 0)
    weekly_start = float(rs.get("weekly_start_value", 0) or 0)
    try:
        dtd = json.loads(rs.get("day_trade_dates", "[]"))
    except Exception:
        dtd = []
    day_trades_rolling = len(dtd)
    con.close()
    return {
        "portfolio":           portfolio,
        "day_pnl_pct":         (portfolio - daily_start)  / daily_start  if is_today  and daily_start  else 0.0,
        "week_pnl_pct":        (portfolio - weekly_start) / weekly_start if is_this_week and weekly_start else 0.0,
        "daily_limit_pct":     DAILY_LOSS_LIMIT_PCT,
        "weekly_limit_pct":    WEEKLY_LOSS_LIMIT_PCT,
        "day_trades_used":     day_trades_rolling,
        "day_trades_limit":    3,
        "daily_warning_sent":  rs.get("daily_warning_sent_date") == today,
        "weekly_halt_alerted": rs.get("weekly_halt_alerted_week") == wk,
    }


def compliance_md(c: dict) -> str:
    if not c:
        return f"No compliance data yet. {_EMPTY_HINT}"
    daily_used  = max(0.0, -c["day_pnl_pct"])  / c["daily_limit_pct"]  * 100
    weekly_used = max(0.0, -c["week_pnl_pct"]) / c["weekly_limit_pct"] * 100
    pdt_used    = c["day_trades_used"]          / c["day_trades_limit"]  * 100
    return (
        f"### Risk Limits\n\n"
        f"| Limit | Used | Threshold | Status |\n|-------|------|-----------|--------|\n"
        f"| Daily Loss | {c['day_pnl_pct']:+.2%} | -{c['daily_limit_pct']:.0%} | "
        f"{'🔴 HIT' if daily_used >= 100 else f'🟡 {daily_used:.0f}%' if daily_used >= 50 else f'🟢 {daily_used:.0f}%'} |\n"
        f"| Weekly Loss | {c['week_pnl_pct']:+.2%} | -{c['weekly_limit_pct']:.0%} | "
        f"{'🔴 HIT' if weekly_used >= 100 else f'🟡 {weekly_used:.0f}%' if weekly_used >= 50 else f'🟢 {weekly_used:.0f}%'} |\n"
        f"| PDT Day Trades | {c['day_trades_used']}/{c['day_trades_limit']} | 3 / rolling 5 business days | "
        f"{'🔴 FULL' if pdt_used >= 100 else f'🟡 {pdt_used:.0f}%' if pdt_used >= 66 else f'🟢 {pdt_used:.0f}%'} |\n"
        f"\n### Flags\n\n"
        f"- Daily warning sent: {'✅ Yes' if c['daily_warning_sent'] else '— No'}\n"
        f"- Weekly halt alerted: {'✅ Yes' if c['weekly_halt_alerted'] else '— No'}\n"
    )


# ── Emergency halt toggle ─────────────────────────────────────────────────────

_HALT_FILE = Path("data/HALT_TRADING")


def halt_status_html() -> str:
    active = _HALT_FILE.exists()
    color  = _NEG if active else _POS
    text   = "🔴 HALT ACTIVE — trading paused" if active else "🟢 TRADING ENABLED — no halt file"
    btn_label = "▶ Remove Halt &amp; Resume" if active else "⏹ Activate Emergency Halt"
    return (
        f"<div style='background:{color};padding:12px 16px;border-radius:8px;"
        f"color:#fff;font-size:14px;font-weight:bold;font-family:{_FONT}'>{text}</div>"
    ), btn_label


def toggle_halt() -> tuple[str, str]:
    _HALT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _HALT_FILE.exists():
        _HALT_FILE.unlink()
        msg = "✅ Halt removed — bot will resume on next cycle."
    else:
        _HALT_FILE.touch()
        msg = "🔴 HALT FILE CREATED — bot will skip trading until removed."
    status_html, btn_label = halt_status_html()
    return status_html, btn_label, msg


# ── Re-exports from sub-modules ───────────────────────────────────────────────

from bot.monitor._dashboard_overview import (  # noqa: E402
    get_overview, overview_md, _provenance_line, _money,
)
from bot.monitor._dashboard_positions import (  # noqa: E402
    get_positions_df, get_returns_summary_df, get_trades_df,
    _POSITION_COLS, _RETURNS_COLS, _live_prices, _prices_alpaca, _prices_yfinance,
)
from bot.monitor._dashboard_performance import (  # noqa: E402
    get_performance_metrics, performance_md,
)
from bot.monitor._dashboard_charts import (  # noqa: E402
    portfolio_chart, signals_chart, monthly_chart,
)
from bot.monitor._dashboard_html import (  # noqa: E402
    compliance_gauges_html, trades_html_table, _trade_rationale,
)
from bot.monitor._dashboard_signals import (  # noqa: E402
    get_latest_signals_df, get_screener_df,
)
