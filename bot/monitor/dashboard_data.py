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
# One palette + font for every custom-rendered surface (charts, HTML cards,
# tables) so all tabs share the same look and feel as the light Gradio chrome.
_BG     = "#ffffff"   # figure / card background
_CARD   = "#f7f8fa"   # subtle inset panel (chart plot area, gauge track)
_TEXT   = "#111827"   # primary text
_MUTED  = "#6b7280"   # secondary / labels
_GRID   = "#e5e7eb"   # gridlines, borders, row dividers
_ACCENT = "#0891b2"   # primary accent (cyan-700 — readable on white)
_POS    = "#15803d"   # gains (green)
_NEG    = "#dc2626"   # losses (red)
_FONT   = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

# Shown when a tab has no data yet — prevents the "is it broken?" confusion by
# telling the user when data will appear.
_EMPTY_HINT = ("The bot trades at market open (9:30 AM ET, Mon–Fri). "
               "Data appears here after the first cycle of the day.")

# Tracks last pull outcome for display in overview
_last_sync: dict = {"ok": None, "ts": None, "err": ""}
# Serializes HF pulls so concurrent tab loads don't race the download/copy.
_pull_lock = threading.Lock()
# Caches the daily SPY benchmark return (network) so Overview doesn't refetch per render.
_spy_cache: dict = {"key": None, "ret": None}

# Alpaca market-data API (free IEX feed) — the durable, official price source.
# yfinance is kept only as a last-resort fallback (unofficial Yahoo scrape).
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
    # Cheap after the first call: pull_db respects a 5-min cache, and off-Space
    # (tests/local) refresh_db_from_hf() is a no-op.
    refresh_db_from_hf()
    if not Path(_DB).exists():
        logger.warning(f"dashboard_data: {_DB} not found — no data to show")
        return None
    return sqlite3.connect(_DB, check_same_thread=False)


def _latest_portfolio_value(con) -> float:
    """Most recent portfolio value from either a trade row or a heartbeat snapshot.

    Snapshots are written every cycle (even with no trade), so this stays live
    instead of reading $0.00 until the first fill.
    """
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
    """Earliest (portfolio_value, timestamp) across trades + snapshots — the
    starting point for total-return and benchmark comparison."""
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
    """S&P 500 (SPY) total return since `start_iso`, best-effort.

    Prefers Alpaca's official data API; falls back to yfinance. Space-only and
    cached for the day so Overview doesn't refetch per render. Returns None on
    failure so the UI degrades gracefully.
    """
    if not start_iso or not os.environ.get("SPACE_ID"):
        return None
    global _spy_cache
    today = date.today().isoformat()
    key = (today, start_iso)
    if _spy_cache.get("key") == key:
        return _spy_cache.get("ret")
    start_day = start_iso[:10]
    ret = _spy_return_alpaca(start_day)
    if ret is None:
        ret = _spy_return_yfinance(start_day)
    _spy_cache = {"key": key, "ret": ret}
    return ret


def refresh_db_from_hf(force: bool = False) -> None:
    """Pull a fresh trades.db from HF dataset (Space-only).

    Called on every _con() (cheap — pull_db respects a 5-min cache) and with
    force=True by the explicit Refresh button. The lock serializes concurrent
    pulls so two tab loads can't download/copy the file at the same time.
    pull_db() already logs the meaningful events (download, cache-skip, failure),
    so this only records the outcome for the overview status line.
    """
    global _last_sync
    if not os.environ.get("SPACE_ID"):
        return
    try:
        from bot.monitor.sync_db import pull_db
        with _pull_lock:
            ok = pull_db(force=force)
        _last_sync = {
            "ok": ok,
            "ts": datetime.now(timezone.utc),
            "err": "" if ok else "pull_db returned False — check HF_TOKEN and HF_DB_REPO_ID",
        }
        if not ok:
            logger.error("refresh_db_from_hf: DB pull returned False — dashboard may show stale data")
    except Exception as exc:
        _last_sync = {"ok": False, "ts": datetime.now(timezone.utc), "err": str(exc)}
        logger.error(f"refresh_db_from_hf: exception — {exc}")


def diagnostics() -> dict:
    """Log a full snapshot of the dashboard's environment and DB state.

    Call this once at Space startup. The log output makes the common failure
    modes (missing HF_TOKEN, empty DB, wrong repo, never-synced) obvious at a
    glance instead of silently rendering $0.00.
    """
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


def _ax_style(ax):
    ax.set_facecolor(_CARD)
    ax.tick_params(colors=_MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(axis="y", color=_GRID, linestyle="--", alpha=0.8)


# ── Overview (Subscriber+) ────────────────────────────────────────────────────

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

    # Live portfolio value — from latest trade or heartbeat snapshot, whichever newer
    portfolio = _latest_portfolio_value(con)

    daily_start  = float(rs.get("daily_start_value",  0) or 0)
    weekly_start = float(rs.get("weekly_start_value", 0) or 0)
    is_today     = rs.get("daily_start_date")  == today
    is_this_week = rs.get("weekly_start_week") == wk

    day_pnl  = (portfolio - daily_start)  / daily_start  if is_today  and daily_start  else 0.0
    week_pnl = (portfolio - weekly_start) / weekly_start if is_this_week and weekly_start else 0.0
    # Dollar P&L alongside the percentages — retail users think in dollars.
    day_pnl_dollars  = (portfolio - daily_start)  if is_today  and daily_start  else 0.0
    week_pnl_dollars = (portfolio - weekly_start) if is_this_week and weekly_start else 0.0

    # Bot total return since inception (DB-only; benchmark comparison added by caller)
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
    # Provenance: total executed trades (paper-trading "since" date = inception_date above).
    try:
        total_trades = con.execute(
            "SELECT COUNT(*) FROM trades WHERE action='BUY' OR action LIKE 'SELL%'"
        ).fetchone()[0]
    except Exception:
        total_trades = 0
    con.close()

    # DB sync status for display
    sync_age_s: float | None = None
    sync_ok: bool | None = _last_sync.get("ok")
    if _last_sync.get("ts"):
        sync_age_s = (datetime.now(timezone.utc) - _last_sync["ts"]).total_seconds()

    # Also check file mtime as a fallback indicator
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
        "spy_return":       None,   # filled in by the caller (network, best-effort)
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


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    return f"{int(seconds / 3600)}h ago"


def overview_md(d: dict) -> str:
    if "_error" in d:
        return f"⚠️ {d['_error']}\n\n_{_EMPTY_HINT}_"

    status = "🔴 EMERGENCY HALT" if d["emergency_halt"] else (
             "🔴 VIX HALT"        if d["macro_halt"]      else (
             "🔴 DAILY LIMIT HIT" if d["daily_limit_hit"] else (
             "🟡 WEEKLY LIMIT"    if d["weekly_limit_hit"] else "🟢 ACTIVE")))

    # Data freshness line (subscriber-friendly: no "DB" jargon)
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

    # Dollar + percent together (retail users think in dollars)
    day_str  = f"{_money(d.get('day_pnl_dollars', 0))} ({d['day_pnl']:+.2%})"
    week_str = f"{_money(d.get('week_pnl_dollars', 0))} ({d['week_pnl']:+.2%})"

    # Bot return vs the S&P 500 — the single most meaningful comparison
    bot_ret = d.get("total_return", 0.0)
    spy_ret = d.get("spy_return")
    if spy_ret is not None:
        verdict = "🟢 beating" if bot_ret > spy_ret else "🔴 trailing"
        vs_spy  = f"{bot_ret:+.2%} vs S&P {spy_ret:+.2%} — {verdict} the market"
    else:
        # Include the inception date so "since inception" has meaning for a subscriber
        inc_date = d.get("inception_date")
        inc_str  = ""
        if inc_date:
            try:
                inc_str = " since " + pd.to_datetime(inc_date).strftime("%b %d, %Y")
            except Exception:
                pass
        vs_spy = f"{bot_ret:+.2%}{inc_str}"

    # Macro score with plain-language tier so subscribers know what 0.52 means
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


# ── Positions (Subscriber+) ───────────────────────────────────────────────────

_POSITION_COLS = ["Symbol", "Shares", "Avg Cost $", "Current $",
                  "Unrealized $", "Unrealized %", "Value $", "% of Portfolio", "Days Held"]


def _prices_alpaca(symbols: list[str]) -> dict:
    """Latest prices from Alpaca snapshots (official IEX feed). {} on failure."""
    headers = _alpaca_headers()
    if headers is None:
        return {}
    try:
        import requests
        r = requests.get(
            f"{_ALPACA_DATA}/v2/stocks/snapshots",
            params={"symbols": ",".join(symbols)}, headers=headers, timeout=8,
        )
        if r.status_code != 200:
            logger.warning(f"_prices_alpaca: HTTP {r.status_code}")
            return {}
        snaps = r.json() or {}
        out: dict = {}
        for sym, snap in snaps.items():
            if not isinstance(snap, dict):
                continue
            # Prefer the latest trade; fall back to the latest daily-bar close.
            p = ((snap.get("latestTrade") or {}).get("p")
                 or (snap.get("dailyBar") or {}).get("c")
                 or (snap.get("minuteBar") or {}).get("c"))
            if p:
                out[sym] = float(p)
        return out
    except Exception as exc:
        logger.warning(f"_prices_alpaca: {exc}")
        return {}


def _prices_yfinance(symbols: list[str]) -> dict:
    """Latest prices from yfinance (unofficial fallback). {} on failure."""
    try:
        import yfinance as yf
        data = yf.download(" ".join(symbols), period="1d", progress=False, auto_adjust=True)
        out: dict = {}
        if "Close" in data:
            close = data["Close"]
            if hasattr(close, "columns"):              # multiple symbols → DataFrame
                for s in symbols:
                    try:
                        out[s] = float(close[s].dropna().iloc[-1])
                    except Exception:
                        pass
            else:                                       # single symbol → Series
                try:
                    out[symbols[0]] = float(close.dropna().iloc[-1])
                except Exception:
                    pass
        return out
    except Exception as exc:
        logger.warning(f"_prices_yfinance: {exc}")
        return {}


def _live_prices(symbols: list[str]) -> dict:
    """Latest prices for `symbols` — Alpaca official feed first, yfinance fallback.

    Space-only and best-effort: returns {} off-Space (tests/local) or on failure
    so callers degrade to showing '—' instead of breaking or blocking.
    """
    if not symbols or not os.environ.get("SPACE_ID"):
        return {}
    out = _prices_alpaca(symbols)
    if not out:
        out = _prices_yfinance(symbols)
    return out


def get_positions_df(prices: dict | None = None, portfolio: float | None = None) -> pd.DataFrame:
    _empty = pd.DataFrame(columns=_POSITION_COLS)
    con = _con()
    if con is None:
        return _empty
    try:
        df = pd.read_sql_query(
            "SELECT symbol, entry_price, high_water_mark, atr_at_entry, opened_at FROM position_state", con
        )
        # Net shares held = total bought − total sold per symbol (position_state has no qty).
        net = dict(con.execute(
            "SELECT symbol, "
            "SUM(CASE WHEN action='BUY' THEN shares WHEN action LIKE 'SELL%' THEN -shares ELSE 0 END) "
            "FROM trades GROUP BY symbol"
        ).fetchall())
        if portfolio is None:
            portfolio = _latest_portfolio_value(con)
    except Exception:
        con.close()
        return _empty
    con.close()
    if df.empty:
        return _empty

    symbols = list(df["symbol"])
    if prices is None:
        prices = _live_prices(symbols)

    now = datetime.now(timezone.utc)
    opened = pd.to_datetime(df["opened_at"], utc=True, errors="coerce")

    rows = []
    for i, r in df.iterrows():
        sym   = r["symbol"]
        entry = float(r["entry_price"] or 0)
        shares = float(net.get(sym) or 0)
        cur   = prices.get(sym)
        d_held = (now - opened[i]).total_seconds() / 86400 if pd.notna(opened[i]) else None
        days   = (f"{int(d_held)}d" if d_held is not None and d_held >= 1
                  else f"{int(d_held * 24)}h" if d_held is not None
                  else "–")
        if cur is not None and entry > 0:
            unreal_usd = shares * (cur - entry)
            unreal_pct = (cur - entry) / entry
            value      = shares * cur
            pct_port   = (value / portfolio) if portfolio else 0.0
            rows.append([sym, round(shares, 3), round(entry, 2), round(cur, 2),
                         f"{'+' if unreal_usd >= 0 else '-'}${abs(unreal_usd):,.2f}",
                         f"{unreal_pct:+.2%}", round(value, 2),
                         f"{pct_port:.1%}", days])
        else:
            # Price unavailable (off-Space or fetch failed) — show what we have.
            value = shares * entry if entry > 0 else 0.0
            rows.append([sym, round(shares, 3), round(entry, 2), "—", "—", "—",
                         round(value, 2) if value else "—", "—", days])
    return pd.DataFrame(rows, columns=_POSITION_COLS)


# ── Holdings & Returns (open + sold in one table for easy comparison) ──────────

_RETURNS_COLS = ["Symbol", "Invested $", "Value $", "Return $", "Return %",
                 "Status", "Date"]


def get_returns_summary_df(prices: dict | None = None) -> pd.DataFrame:
    """Per-stock investment vs total return, covering BOTH open and sold positions.

    For each symbol the bot has traded:
      Invested $   = total spent buying (cost basis)
      Return $     = realized P&L (from sells) + unrealized P&L (open shares)
      Value $      = market value if open; total proceeds if sold
      Return %     = Return / Invested
      Status       = Open (still holding) or Sold (fully exited)
      Date         = "Opened Jun 09" for open, "Closed Jun 12" for sold
    """
    con = _con()
    if con is None:
        return pd.DataFrame(columns=_RETURNS_COLS)
    try:
        agg = con.execute(
            "SELECT symbol, "
            " SUM(CASE WHEN action='BUY' THEN notional ELSE 0 END)        AS invested, "
            " SUM(CASE WHEN action='BUY' THEN shares ELSE 0 END)          AS bought_sh, "
            " SUM(CASE WHEN action LIKE 'SELL%' THEN shares ELSE 0 END)    AS sold_sh, "
            " COALESCE(SUM(realized_pnl), 0)                               AS realized, "
            " MIN(CASE WHEN action='BUY' THEN timestamp END)              AS first_buy, "
            " MAX(CASE WHEN action LIKE 'SELL%' THEN timestamp END)        AS last_sell "
            "FROM trades GROUP BY symbol"
        ).fetchall()
        entries = dict(con.execute("SELECT symbol, entry_price FROM position_state").fetchall())
    except Exception:
        con.close()
        return pd.DataFrame(columns=_RETURNS_COLS)
    con.close()
    if not agg:
        return pd.DataFrame(columns=_RETURNS_COLS)

    # Live prices only needed for symbols still open
    open_syms = [r[0] for r in agg if (float(r[2] or 0) - float(r[3] or 0)) > 1e-6]
    if prices is None:
        prices = _live_prices(open_syms)

    rows = []
    for sym, buy_notional, bought_sh, sold_sh, realized, first_buy, last_sell in agg:
        buy_notional = float(buy_notional or 0)
        net_sh       = float(bought_sh or 0) - float(sold_sh or 0)
        realized     = float(realized or 0)
        is_open      = net_sh > 1e-6
        entry        = float(entries.get(sym) or 0)
        cur          = prices.get(sym)
        status       = "🟢 Open" if is_open else "⚪ Sold"
        when         = (first_buy if is_open else last_sell) or first_buy or last_sell
        # Subscriber-friendly date: "Opened Jun 09" / "Closed Jun 12"
        try:
            when_dt  = pd.to_datetime(when)
            when_str = ("Opened " if is_open else "Closed ") + when_dt.strftime("%b %d")
        except Exception:
            when_str = "–"

        if is_open:
            # Open: use the same entry-price basis as the Positions table so the two
            # tables agree on a stock's value/return (cost basis = net_sh × entry).
            if cur is not None and entry > 0:
                invested     = net_sh * entry
                total_return = net_sh * (cur - entry) + realized
                value        = invested + total_return          # = net_sh*cur + realized
                ret_pct      = total_return / invested if invested else None
                rows.append([sym, round(invested, 2), round(value, 2),
                             round(total_return, 2),
                             f"{ret_pct:+.2%}" if ret_pct is not None else "—",
                             status, when_str])
            else:
                # Price/entry unavailable — show cost basis, leave return unknown.
                invested = net_sh * entry if entry > 0 else buy_notional
                rows.append([sym, round(invested, 2), "—", "—", "—", status, when_str])
        else:
            # Sold: cost basis = what was paid; proceeds = invested + realized P&L.
            # Label Value $ as "proceeds" so subscribers don't think position is still live.
            invested     = buy_notional
            total_return = realized
            proceeds     = invested + realized
            ret_pct      = (realized / invested) if invested else None
            rows.append([sym, round(invested, 2), f"${proceeds:,.2f} (proceeds)",
                         round(total_return, 2),
                         f"{ret_pct:+.2%}" if ret_pct is not None else "—",
                         status, when_str])

    # Sort: open first, then by absolute return size
    df = pd.DataFrame(rows, columns=_RETURNS_COLS)
    return df


# ── Trade Log (Subscriber+) ───────────────────────────────────────────────────

def get_trades_df(days: int = 30) -> pd.DataFrame:
    con = _con()
    if con is None:
        return pd.DataFrame()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    df = pd.read_sql_query(
        "SELECT timestamp, symbol, action, shares, price, notional, pnl_pct "
        "FROM trades WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT 200",
        con, params=(since,),
    )
    con.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%m-%d %H:%M")
    df["pnl_pct"]   = df["pnl_pct"].apply(lambda x: f"{x:+.2%}" if x else "–")
    df["notional"]  = df["notional"].apply(lambda x: f"${x:,.2f}" if x else "–")
    df["price"]     = df["price"].apply(lambda x: f"${x:.2f}" if x else "–")
    df.columns = ["Time", "Symbol", "Action", "Shares", "Price", "Notional", "P&L"]
    return df


# ── Performance metrics (Pro+) ────────────────────────────────────────────────

# Minimum 5-min-bar return observations before an annualized Sharpe is meaningful.
_MIN_SHARPE_OBS  = 20   # minimum return observations
_MIN_SHARPE_DAYS = 20   # AND minimum distinct trading days — a Sharpe from a few
                        # hours of intraday snapshots is meaningless (annualization blows up)


def get_performance_metrics(days: int = 60) -> dict:
    con = _con()
    if con is None:
        return {}
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    _empty = {"sharpe": None, "win_rate": 0.0, "max_drawdown": 0.0,
              "total_return": 0.0, "trade_count": 0, "closed_trades": 0}

    # Portfolio-value trajectory: union trade rows AND heartbeat snapshots, ordered
    # by time — same source as Overview and the equity chart. Using trades alone
    # showed 0% return whenever the only trade rows shared one portfolio_value
    # (e.g. legacy rows all at the starting balance), ignoring the account's real
    # growth captured by snapshots. Falls back to trades-only on older DBs.
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

    # Win rate is over CLOSED trades only (SELL rows) — open BUYs have no outcome yet.
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
    # Annualized Sharpe assumes ~uniform 5-min bar returns (sqrt(252*78)). With only
    # a few hours of intraday snapshots it produces absurd values (e.g. 24), so it's
    # reported only once there are enough observations AND enough distinct trading
    # days for the annualization to mean something.
    distinct_days = len({ts[:10] for ts, _ in pv_rows if ts})
    if (len(rets) >= _MIN_SHARPE_OBS and distinct_days >= _MIN_SHARPE_DAYS
            and np.std(rets) > 0):
        sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252 * 78))
    else:
        sharpe = None
    peak = vals[0]; max_dd = 0.0
    for v in vals:
        peak   = max(peak, v)
        max_dd = max(max_dd, (peak - v) / (peak + 1e-8))
    closed   = len(sells)
    wins     = sum(1 for r in sells if r[0] is not None and r[0] > 0)
    win_rate = wins / closed if closed else 0.0
    return {
        "sharpe":        round(sharpe, 2) if sharpe is not None else None,
        "win_rate":      round(win_rate, 4),
        "max_drawdown":  round(max_dd, 4),
        "total_return":  round((vals[-1] - vals[0]) / (vals[0] + 1e-8), 4),
        "trade_count":   trade_count,
        "closed_trades": closed,
    }


def performance_md(m: dict) -> str:
    if not m:
        return f"No performance data yet. {_EMPTY_HINT}"
    closed = m.get("closed_trades", 0)
    # 0% win rate with no closed trades isn't "losing" — make that explicit.
    win_str = f"{m['win_rate']:.1%}" if closed else "n/a (no closed trades yet)"
    # Sharpe is None until there's enough history to annualize meaningfully.
    sharpe_str = f"{m['sharpe']:.2f}" if m.get("sharpe") is not None else "n/a (need more history)"
    return (
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Sharpe Ratio | **{sharpe_str}** |\n"
        f"| Win Rate | **{win_str}** |\n"
        f"| Max Drawdown | **{m['max_drawdown']:.1%}** |\n"
        f"| Total Return | **{m['total_return']:+.2%}** |\n"
        f"| Trades Analysed | {m['trade_count']} ({closed} closed) |"
    )


# ── Charts (Pro+) ─────────────────────────────────────────────────────────────

def portfolio_chart(days: int = 60) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(_BG); _ax_style(ax)
    con = _con()
    if con is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color=_MUTED); return fig
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    # Union trade rows and heartbeat snapshots so the curve is populated even on
    # no-trade days; falls back to trades-only on older DBs lacking the snapshot table.
    try:
        df = pd.read_sql_query(
            "SELECT timestamp, portfolio_value FROM trades WHERE timestamp >= ? "
            "UNION ALL "
            "SELECT timestamp, portfolio_value FROM portfolio_snapshots WHERE timestamp >= ? "
            "ORDER BY timestamp",
            con, params=(since, since),
        )
    except Exception:
        df = pd.read_sql_query(
            "SELECT timestamp, portfolio_value FROM trades WHERE timestamp >= ? ORDER BY timestamp",
            con, params=(since,),
        )
    con.close()
    if df.empty:
        ax.text(0.5, 0.5, f"No data yet\n{_EMPTY_HINT}", ha="center", va="center",
                color=_MUTED, fontsize=9, wrap=True); return fig
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    ax.plot(df["timestamp"], df["portfolio_value"], color=_ACCENT, lw=1.8, label="Portfolio")
    ax.fill_between(df["timestamp"], df["portfolio_value"], df["portfolio_value"].min() * 0.999,
                    alpha=0.12, color=_ACCENT)
    ax.set_title("Portfolio Value", color=_TEXT, fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    fig.autofmt_xdate(); fig.tight_layout(); return fig


def signals_chart(days: int = 30) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    fig.patch.set_facecolor(_BG)
    for ax in axes.flatten(): _ax_style(ax)
    con = _con()
    if con is None:
        axes[0][0].text(0.5, 0.5, "No data", ha="center", va="center", color=_MUTED); return fig
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    df = pd.read_sql_query(
        "SELECT xgb_prob, lstm_prob, sentiment_score, ensemble_score FROM trades "
        "WHERE action='BUY' AND timestamp >= ?", con, params=(since,),
    )
    con.close()
    pairs = [(axes[0][0], "xgb_prob", "XGB Probability", "#2563eb"),
             (axes[0][1], "lstm_prob", "LSTM Probability", _POS),
             (axes[1][0], "sentiment_score", "Sentiment Score", "#d97706"),
             (axes[1][1], "ensemble_score", "Ensemble Score", "#7c3aed")]
    for ax, col, title, color in pairs:
        data = df[col].dropna() if not df.empty and col in df.columns else pd.Series(dtype=float)
        if data.empty:
            ax.text(0.5, 0.5, "No BUY data yet", ha="center", va="center", color=_MUTED, fontsize=9)
        else:
            ax.hist(data, bins=20, color=color, alpha=0.85, edgecolor="none")
        ax.set_title(title, color=_TEXT, fontsize=10, fontweight="bold")
    fig.suptitle("Signal Score Distributions  (BUY entries)", color=_TEXT, fontsize=12, fontweight="bold")
    fig.tight_layout(); return fig


def monthly_chart(days: int = 180) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(_BG); _ax_style(ax)
    con = _con()
    if con is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color=_MUTED); return fig
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        df = pd.read_sql_query(
            "SELECT timestamp, portfolio_value FROM trades WHERE timestamp >= ? "
            "UNION ALL "
            "SELECT timestamp, portfolio_value FROM portfolio_snapshots WHERE timestamp >= ? "
            "ORDER BY timestamp",
            con, params=(since, since),
        )
    except Exception:
        df = pd.read_sql_query(
            "SELECT timestamp, portfolio_value FROM trades WHERE timestamp >= ? ORDER BY timestamp",
            con, params=(since,),
        )
    con.close()
    if df.empty:
        ax.text(0.5, 0.5, f"No trades yet\n{_EMPTY_HINT}", ha="center", va="center",
                color=_MUTED, fontsize=9, wrap=True); return fig
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df["month"]     = df["timestamp"].dt.to_period("M")
    m = df.groupby("month")["portfolio_value"].agg(["first", "last"])
    m["ret"] = (m["last"] - m["first"]) / (m["first"] + 1e-8) * 100
    colors = [_POS if r >= 0 else _NEG for r in m["ret"]]
    ax.bar([str(p) for p in m.index], m["ret"], color=colors, edgecolor="none")
    ax.axhline(0, color=_MUTED, lw=0.8)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.set_title("Monthly Returns", color=_TEXT, fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=25, colors=_MUTED)
    fig.tight_layout(); return fig


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
    # PDT is a rolling 5-business-day window — count all dates stored, not just today
    # (risk_state already prunes dates older than 5 business days)
    day_trades_rolling = len(dtd)
    con.close()
    return {
        "portfolio":           portfolio,
        # Guard against stale daily/weekly anchors (same logic as get_overview)
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
    # Only LOSSES consume the risk limit — gains don't, so use max(0, -pnl)
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


# ── Visual risk gauges — HTML progress bars (Institutional) ──────────────────

def compliance_gauges_html(c: dict) -> str:
    if not c:
        return f"<p style='color:{_MUTED};font-family:{_FONT}'>No compliance data yet. {_EMPTY_HINT}</p>"

    def _bar(label: str, value_str: str, limit_str: str, pct: float) -> str:
        pct     = min(pct * 100, 100)
        color   = _NEG if pct >= 100 else "#d97706" if pct >= 50 else _POS
        return (
            f"<div style='margin:14px 0'>"
            f"<div style='display:flex;justify-content:space-between;color:{_MUTED};font-size:13px;margin-bottom:4px'>"
            f"<span>{label}</span><span>{value_str} &nbsp;/&nbsp; limit {limit_str}</span></div>"
            f"<div style='background:{_GRID};border-radius:6px;height:18px'>"
            f"<div style='background:{color};width:{pct:.1f}%;height:100%;border-radius:6px;"
            f"transition:width .3s;display:flex;align-items:center;padding-left:6px;"
            f"font-size:11px;color:#fff;font-weight:bold'>"
            f"{'&nbsp;' + f'{pct:.0f}%' if pct > 10 else ''}"
            f"</div></div></div>"
        )

    # Gauges fill toward the limit only when LOSING — gains don't consume the limit
    daily_pct  = max(0.0, -c["day_pnl_pct"])  / (c["daily_limit_pct"]  or 1)
    weekly_pct = max(0.0, -c["week_pnl_pct"]) / (c["weekly_limit_pct"] or 1)
    pdt_pct    = c["day_trades_used"]          / (c["day_trades_limit"]  or 1)

    flags = ""
    if c["daily_warning_sent"]:
        flags += f"<span style='background:#d97706;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;margin-right:6px'>⚠ Daily warning sent</span>"
    if c["weekly_halt_alerted"]:
        flags += f"<span style='background:{_NEG};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px'>🔴 Weekly halt alerted</span>"

    return (
        f"<div style='background:{_BG};border:1px solid {_GRID};padding:18px;border-radius:10px;font-family:{_FONT}'>"
        f"<h3 style='color:{_TEXT};margin-top:0'>Risk Limit Gauges</h3>"
        + _bar("Daily Loss",   f"{c['day_pnl_pct']:+.2%}",  f"-{c['daily_limit_pct']:.0%}",  daily_pct)
        + _bar("Weekly Loss",  f"{c['week_pnl_pct']:+.2%}", f"-{c['weekly_limit_pct']:.0%}", weekly_pct)
        + _bar("Day Trades",   f"{c['day_trades_used']}/{c['day_trades_limit']}", "3 max / rolling 5 business days", pdt_pct)
        + (f"<div style='margin-top:12px'>{flags}</div>" if flags else "")
        + "</div>"
    )


# ── Color-coded Trade Log HTML (Subscriber+) ──────────────────────────────────

_ACTION_COLOR = {
    "BUY":                "#388e3c",   # green
    "SELL":               "#1565c0",   # blue — planned signal exit
    "SELL_TAKE_PROFIT":   "#00838f",   # teal
    "SELL_TRAILING_STOP": "#e65100",   # deep orange
    "SELL_TIME_EXIT":     "#6a1b9a",   # purple
    "SELL_STOP":          "#b71c1c",   # red
    "SELL_GAP_DOWN":      "#7f0000",   # dark red
}

# Plain-language reason per exit type (the action IS the reason for sells).
_SELL_REASON = {
    "SELL":               "Signal exit",
    "SELL_TAKE_PROFIT":   "Took profit",
    "SELL_TRAILING_STOP": "Trailing stop",
    "SELL_TIME_EXIT":     "Max hold reached",
    "SELL_STOP":          "Stop-loss hit",
    "SELL_GAP_DOWN":      "Gap-down protection",
}


def _trade_rationale(row) -> str:
    """One-line 'why' for a trade.

    Exit rows: plain-English reason (already stored as the action type).
    Entry rows: subscriber-friendly signal summary — avoids raw model names
    (XGB/LSTM) that mean nothing to a non-technical user.
    """
    action = str(row["action"])
    if action.startswith("SELL"):
        return _SELL_REASON.get(action, "Exit")
    # Buy: summarise signal strength in plain language + market regime
    xgb    = float(row.get("xgb_prob")        or 0)
    lstm   = float(row.get("lstm_prob")       or 0)
    sent   = float(row.get("sentiment_score") or 0)
    regime = str(row.get("regime") or "").strip()
    avg_model = (xgb + lstm) / 2 if (xgb or lstm) else 0
    if avg_model >= 0.75:
        strength = "Strong buy signal"
    elif avg_model >= 0.55:
        strength = "Buy signal"
    else:
        strength = "AI signal"
    if sent > 0.15:
        strength += " · positive news"
    elif sent < -0.15:
        strength += " · negative news"
    if regime and regime not in ("", "Unknown"):
        strength += f" · {regime.replace('_', ' ').title()}"
    return strength


def trades_html_table(days: int = 30) -> str:
    con = _con()
    if con is None:
        return f"<p style='color:{_MUTED};font-family:{_FONT}'>No trades yet. {_EMPTY_HINT}</p>"
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    df = pd.read_sql_query(
        "SELECT timestamp, symbol, action, shares, price, notional, pnl_pct, "
        "realized_pnl, xgb_prob, lstm_prob, sentiment_score, regime "
        "FROM trades WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT 200",
        con, params=(since,),
    )
    con.close()
    if df.empty:
        return (f"<p style='color:{_MUTED};font-family:{_FONT}'>No trades in the selected window. "
                f"Try a longer range, or check back after the next market session.</p>")

    # Subscriber-friendly display names for action codes
    _ACTION_DISPLAY = {
        "BUY":                "BUY",
        "SELL":               "SELL",
        "SELL_TAKE_PROFIT":   "SELL",
        "SELL_TRAILING_STOP": "SELL",
        "SELL_TIME_EXIT":     "SELL",
        "SELL_STOP":          "SELL",
        "SELL_GAP_DOWN":      "SELL",
    }

    rows_html = ""
    for _, row in df.iterrows():
        action  = str(row["action"])
        color   = _ACTION_COLOR.get(action, _MUTED)
        # Show clean "BUY" / "SELL" label — the Why column explains the specific reason
        display_action = _ACTION_DISPLAY.get(action, action)
        # Glyph makes buy/sell distinguishable without relying on colour (colourblind-safe).
        glyph   = "▲" if action == "BUY" else "▼"
        ts      = pd.to_datetime(row["timestamp"]).strftime("%m-%d %H:%M ET")
        # Show realized P&L as "$X.XX (Y.Y%)" on exits; "–" on entries (not yet closed).
        rlz_pct = row.get("pnl_pct")
        rlz_usd = row.get("realized_pnl")
        if action.startswith("SELL") and (rlz_pct or rlz_usd):
            pnl_usd_str = f"${rlz_usd:+,.2f}" if rlz_usd else ""
            pnl_pct_str = f"{rlz_pct:+.2%}"   if rlz_pct else ""
            pnl_str = f"{pnl_usd_str} ({pnl_pct_str})" if pnl_usd_str and pnl_pct_str else pnl_usd_str or pnl_pct_str
            pnl_col = _POS if (rlz_usd or rlz_pct or 0) > 0 else _NEG
        else:
            pnl_str = "–"
            pnl_col = _MUTED
        notional_str = f"${row['notional']:,.2f}" if row["notional"] else "–"
        why = _trade_rationale(row)
        rows_html += (
            f"<tr style='border-bottom:1px solid {_GRID}'>"
            f"<td style='color:{_MUTED};padding:6px'>{ts}</td>"
            f"<td style='color:{_TEXT};font-weight:bold;padding:6px'>{row['symbol']}</td>"
            f"<td style='padding:6px'><span style='background:{color};color:#fff;padding:2px 7px;border-radius:4px;"
            f"font-size:11px;white-space:nowrap'>{glyph} {display_action}</span></td>"
            f"<td style='color:{_TEXT};padding:6px'>{row['shares']:.3f}</td>"
            f"<td style='color:{_TEXT};padding:6px'>${row['price']:.2f}</td>"
            f"<td style='color:{_TEXT};padding:6px'>{notional_str}</td>"
            f"<td style='color:{pnl_col};font-weight:bold;padding:6px'>{pnl_str}</td>"
            f"<td style='color:{_MUTED};padding:6px;font-size:12px'>{why}</td>"
            f"</tr>"
        )

    return (
        "<div class='cf-table' style='overflow-x:auto'>"
        f"<table style='width:100%;border-collapse:collapse;font-family:{_FONT};font-size:13px'>"
        f"<thead><tr style='color:{_MUTED};border-bottom:2px solid {_GRID}'>"
        "<th style='text-align:left;padding:6px'>Time</th>"
        "<th style='text-align:left;padding:6px'>Symbol</th>"
        "<th style='text-align:left;padding:6px'>Action</th>"
        "<th style='text-align:left;padding:6px'>Shares</th>"
        "<th style='text-align:left;padding:6px'>Price</th>"
        "<th style='text-align:left;padding:6px'>Amount $</th>"
        "<th style='text-align:left;padding:6px'>Realized P&amp;L</th>"
        "<th style='text-align:left;padding:6px'>Why</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table></div>"
    )


# ── Live signal feed (all symbols, latest cycle) ──────────────────────────────

# Maps ensemble action string to a display colour for the signal table.
_ACTION_BADGE: dict[str, str] = {
    "STRONG_BUY":  _POS,
    "BUY":         "#15803d",
    "HOLD":        _MUTED,
    "SELL":        _NEG,
    "STRONG_SELL": "#7f1d1d",
}


_SIGNAL_COLS = ["Symbol", "As Of", "Signal", "Score", "XGB", "LSTM", "Sentiment", "Macro", "Regime"]


def _empty_signals_df() -> pd.DataFrame:
    """Typed empty DataFrame so Gradio renders column headers instead of 'undefined'."""
    return pd.DataFrame(columns=_SIGNAL_COLS)


def get_latest_signals_df() -> pd.DataFrame:
    """Return the most recent signal log row per symbol (latest cycle).

    Pulls from signal_log table written every bot cycle for all evaluated symbols,
    so this stays current even when no trade fires.
    """
    con = _con()
    if con is None:
        return _empty_signals_df()
    # Ensure the table exists — it may be missing on older DBs pulled from HF
    # before the migration landed. CREATE IF NOT EXISTS is a safe no-op otherwise.
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS signal_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, symbol TEXT NOT NULL,
                xgb_prob REAL, lstm_prob REAL, sentiment_score REAL,
                macro_score REAL, ensemble_score REAL,
                ensemble_action TEXT, regime TEXT
            )
        """)
        con.commit()
    except Exception:
        pass
    try:
        df = pd.read_sql_query(
            """
            SELECT s.symbol, s.timestamp, s.ensemble_action, s.ensemble_score,
                   s.xgb_prob, s.lstm_prob, s.sentiment_score, s.macro_score, s.regime
            FROM signal_log s
            INNER JOIN (
                SELECT symbol, MAX(timestamp) AS ts FROM signal_log GROUP BY symbol
            ) latest ON s.symbol = latest.symbol AND s.timestamp = latest.ts
            ORDER BY s.ensemble_score DESC
            """,
            con,
        )
    except Exception:
        con.close()
        return _empty_signals_df()
    con.close()
    if df.empty:
        return _empty_signals_df()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%m-%d %H:%M")
    df = df.rename(columns={
        "symbol":          "Symbol",
        "timestamp":       "As Of",
        "ensemble_action": "Signal",
        "ensemble_score":  "Score",
        "xgb_prob":        "XGB",
        "lstm_prob":       "LSTM",
        "sentiment_score": "Sentiment",
        "macro_score":     "Macro",
        "regime":          "Regime",
    })
    for col in ("Score", "XGB", "LSTM", "Sentiment", "Macro"):
        df[col] = df[col].round(3)
    return df


_SCREENER_COLS = ["Rank", "Symbol", "Sector", "Score", "Analyst", "ETF Mom", "Regime", "Screened At"]


def _empty_screener_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_SCREENER_COLS)


def get_screener_df() -> pd.DataFrame:
    """Return today's screener picks with factor scores, sorted by rank."""
    con = _con()
    if con is None:
        return _empty_screener_df()
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS screener_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                screened_at TEXT NOT NULL, symbol TEXT NOT NULL,
                rank INTEGER, composite_score REAL, analyst_signal REAL,
                etf_momentum REAL, regime TEXT, sector TEXT
            )
        """)
        con.commit()
    except Exception:
        pass
    try:
        # Most recent screener run only
        df = pd.read_sql_query("""
            SELECT s.rank, s.symbol, s.sector, s.composite_score,
                   s.analyst_signal, s.etf_momentum, s.regime, s.screened_at
            FROM screener_log s
            INNER JOIN (
                SELECT MAX(screened_at) AS latest FROM screener_log
            ) r ON s.screened_at = r.latest
            ORDER BY s.rank
        """, con)
    except Exception:
        con.close()
        return _empty_screener_df()
    con.close()
    if df.empty:
        return _empty_screener_df()
    df["screened_at"] = pd.to_datetime(df["screened_at"]).dt.strftime("%m-%d %H:%M")
    # Format analyst signal as readable label
    def _fmt_analyst(v):
        if v is None or v != v:
            return "—"
        if v > 0.1:
            return f"+{v:.2f} ▲"
        if v < -0.1:
            return f"{v:.2f} ▼"
        return f"{v:.2f}"
    df["analyst_signal"] = df["analyst_signal"].apply(_fmt_analyst)
    df["etf_momentum"] = df["etf_momentum"].apply(
        lambda v: "↑ Above SMA" if v is not None and v >= 0.5 else ("↓ Below SMA" if v is not None else "—")
    )
    df["composite_score"] = df["composite_score"].round(3)
    df = df.rename(columns={
        "rank":            "Rank",
        "symbol":          "Symbol",
        "sector":          "Sector",
        "composite_score": "Score",
        "analyst_signal":  "Analyst",
        "etf_momentum":    "ETF Mom",
        "regime":          "Regime",
        "screened_at":     "Screened At",
    })
    return df


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
