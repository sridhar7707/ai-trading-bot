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

    # DB sync status line
    sync_ok  = d.get("sync_ok")
    db_age   = d.get("db_age_s")
    sync_err = d.get("sync_err", "")
    if sync_ok is True:
        sync_line = f"🟢 DB synced {_fmt_age(d.get('sync_age_s'))} · file updated {_fmt_age(db_age)}"
    elif sync_ok is False:
        err_hint = f" — {sync_err}" if sync_err else ""
        sync_line = f"🔴 DB sync FAILED{err_hint} · file last updated {_fmt_age(db_age)}"
    else:
        sync_line = f"⚪ DB not yet synced · file last updated {_fmt_age(db_age)}"

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
        vs_spy  = f"{bot_ret:+.2%} since inception"

    return (
        f"### Bot Status: {status}\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Portfolio Value | **${d['portfolio']:,.2f}** |\n"
        f"| Day P&L | **{day_str}** |\n"
        f"| Week P&L | **{week_str}** |\n"
        f"| Return vs S&P 500 | **{vs_spy}** |\n"
        f"| Trades Today | {d['trades_today']} |\n"
        f"| Open Positions | {d['open_positions']} |\n"
        f"| Day Trades Used | {d['day_trades_used']}/3 (PDT) |\n"
        f"| Macro Score | {d['macro_score']:.2f} |\n\n"
        f"*{sync_line}*\n"
    )


# ── Positions (Subscriber+) ───────────────────────────────────────────────────

_POSITION_COLS = ["Symbol", "Shares", "Entry $", "Current $", "Unrealized %",
                  "Value $", "% Port", "Days"]


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
        days  = f"{(now - opened[i]).total_seconds()/86400:.1f}d" if pd.notna(opened[i]) else "–"
        if cur is not None and entry > 0:
            unreal_pct = (cur - entry) / entry
            value      = shares * cur
            pct_port   = (value / portfolio) if portfolio else 0.0
            rows.append([sym, round(shares, 3), round(entry, 2), round(cur, 2),
                         f"{unreal_pct:+.2%}", round(value, 2),
                         f"{pct_port:.1%}", days])
        else:
            # Price unavailable (off-Space or fetch failed) — show what we have.
            value = shares * entry if entry > 0 else 0.0
            rows.append([sym, round(shares, 3), round(entry, 2), "—", "—",
                         round(value, 2) if value else "—", "—", days])
    return pd.DataFrame(rows, columns=_POSITION_COLS)


# ── Holdings & Returns (open + sold in one table for easy comparison) ──────────

_RETURNS_COLS = ["Symbol", "Invested $", "Value $", "Return $", "Return %",
                 "Status", "Since / Sold"]


def get_returns_summary_df(prices: dict | None = None) -> pd.DataFrame:
    """Per-stock investment vs total return, covering BOTH open and sold positions.

    For each symbol the bot has traded:
      Invested $   = total spent buying (cost basis)
      Return $     = realized P&L (from sells) + unrealized P&L (open shares)
      Value $      = Invested + Return  (current worth incl. realized gains)
      Return %     = Return / Invested
      Status       = Open (still holding) or Sold (fully exited)
      Since / Sold = first buy date (open) or last sell date (sold)
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
    for sym, invested, bought_sh, sold_sh, realized, first_buy, last_sell in agg:
        invested  = float(invested or 0)
        net_sh    = float(bought_sh or 0) - float(sold_sh or 0)
        realized  = float(realized or 0)
        is_open   = net_sh > 1e-6

        unrealized = 0.0
        entry = float(entries.get(sym) or 0)
        cur   = prices.get(sym)
        if is_open and cur is not None and entry > 0:
            unrealized = net_sh * (cur - entry)

        total_return = realized + unrealized
        value        = invested + total_return
        ret_pct      = (total_return / invested) if invested else None
        status       = "🟢 Open" if is_open else "⚪ Sold"
        when         = (first_buy if is_open else last_sell) or first_buy or last_sell
        when_str     = pd.to_datetime(when).strftime("%Y-%m-%d") if when else "–"

        # If open but price is unavailable, the return is only partially known.
        if is_open and cur is None and net_sh > 1e-6:
            ret_str = "—"
            val_disp = "—"
            tot_disp = "—"
        else:
            ret_str  = f"{ret_pct:+.2%}" if ret_pct is not None else "—"
            val_disp = round(value, 2)
            tot_disp = round(total_return, 2)

        rows.append([sym, round(invested, 2), val_disp, tot_disp, ret_str, status, when_str])

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
_MIN_SHARPE_OBS = 20


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
    # a handful of irregular points it produces absurd values (e.g. 39), so report
    # None until there's enough history to be meaningful.
    if len(rets) >= _MIN_SHARPE_OBS and np.std(rets) > 0:
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
    df = pd.read_sql_query(
        "SELECT timestamp, portfolio_value FROM trades WHERE timestamp >= ? ORDER BY timestamp",
        con, params=(since,),
    )
    con.close()
    if df.empty:
        ax.text(0.5, 0.5, f"No trades yet\n{_EMPTY_HINT}", ha="center", va="center",
                color=_MUTED, fontsize=9, wrap=True); return fig
    df["timestamp"] = pd.to_datetime(df["timestamp"])
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
    daily_start  = float(rs.get("daily_start_value",  0) or 0)
    weekly_start = float(rs.get("weekly_start_value", 0) or 0)
    try:
        dtd = json.loads(rs.get("day_trade_dates", "[]"))
    except Exception:
        dtd = []
    con.close()
    return {
        "portfolio":           portfolio,
        "day_pnl_pct":         (portfolio - daily_start)  / daily_start  if daily_start  else 0.0,
        "week_pnl_pct":        (portfolio - weekly_start) / weekly_start if weekly_start else 0.0,
        "daily_limit_pct":     DAILY_LOSS_LIMIT_PCT,
        "weekly_limit_pct":    WEEKLY_LOSS_LIMIT_PCT,
        "day_trades_used":     dtd.count(today),
        "day_trades_limit":    3,
        "daily_warning_sent":  rs.get("daily_warning_sent_date") == today,
        "weekly_halt_alerted": rs.get("weekly_halt_alerted_week") == wk,
    }


def compliance_md(c: dict) -> str:
    if not c:
        return f"No compliance data yet. {_EMPTY_HINT}"
    daily_used  = abs(c["day_pnl_pct"])  / c["daily_limit_pct"]  * 100
    weekly_used = abs(c["week_pnl_pct"]) / c["weekly_limit_pct"] * 100
    pdt_used    = c["day_trades_used"]   / c["day_trades_limit"]  * 100
    return (
        f"### Risk Limits\n\n"
        f"| Limit | Used | Threshold | Status |\n|-------|------|-----------|--------|\n"
        f"| Daily Loss | {c['day_pnl_pct']:+.2%} | -{c['daily_limit_pct']:.0%} | "
        f"{'🔴 HIT' if daily_used >= 100 else f'🟡 {daily_used:.0f}%' if daily_used >= 50 else f'🟢 {daily_used:.0f}%'} |\n"
        f"| Weekly Loss | {c['week_pnl_pct']:+.2%} | -{c['weekly_limit_pct']:.0%} | "
        f"{'🔴 HIT' if weekly_used >= 100 else f'🟡 {weekly_used:.0f}%' if weekly_used >= 50 else f'🟢 {weekly_used:.0f}%'} |\n"
        f"| PDT Day Trades | {c['day_trades_used']}/{c['day_trades_limit']} | 3/rolling 5d | "
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

    daily_pct  = abs(c["day_pnl_pct"])  / (c["daily_limit_pct"]  or 1)
    weekly_pct = abs(c["week_pnl_pct"]) / (c["weekly_limit_pct"] or 1)
    pdt_pct    = c["day_trades_used"]   / (c["day_trades_limit"]  or 1)

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
        + _bar("PDT Trades",   f"{c['day_trades_used']}/{c['day_trades_limit']}", "3 / 5-day window", pdt_pct)
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
    """One-line 'why' for a trade: exit reason for sells, top signal drivers for buys."""
    action = str(row["action"])
    if action.startswith("SELL"):
        return _SELL_REASON.get(action, "Exit")
    xgb    = float(row.get("xgb_prob")        or 0)
    lstm   = float(row.get("lstm_prob")       or 0)
    sent   = float(row.get("sentiment_score") or 0)
    regime = str(row.get("regime") or "").strip()
    parts = []
    if xgb:  parts.append(f"XGB {xgb:.2f}")
    if lstm: parts.append(f"LSTM {lstm:.2f}")
    if sent: parts.append(f"news {sent:+.2f}")
    if regime and regime not in ("", "Unknown"):
        parts.append(regime.replace("_", " ").title())
    return " · ".join(parts) if parts else "—"


def trades_html_table(days: int = 30) -> str:
    con = _con()
    if con is None:
        return f"<p style='color:{_MUTED};font-family:{_FONT}'>No trades yet. {_EMPTY_HINT}</p>"
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    df = pd.read_sql_query(
        "SELECT timestamp, symbol, action, shares, price, notional, pnl_pct, "
        "xgb_prob, lstm_prob, sentiment_score, regime "
        "FROM trades WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT 200",
        con, params=(since,),
    )
    con.close()
    if df.empty:
        return (f"<p style='color:{_MUTED};font-family:{_FONT}'>No trades in the selected window. "
                f"Try a longer range, or check back after the next market session.</p>")

    rows_html = ""
    for _, row in df.iterrows():
        action  = str(row["action"])
        color   = _ACTION_COLOR.get(action, _MUTED)
        # Glyph makes buy/sell distinguishable without relying on colour (colourblind-safe).
        glyph   = "▲" if action == "BUY" else "▼"
        ts      = pd.to_datetime(row["timestamp"]).strftime("%m-%d %H:%M")
        pnl_str = f"{row['pnl_pct']:+.2%}" if row["pnl_pct"] else "–"
        pnl_col = _POS if row["pnl_pct"] and row["pnl_pct"] > 0 else (_NEG if row["pnl_pct"] and row["pnl_pct"] < 0 else _MUTED)
        notional_str = f"${row['notional']:,.2f}" if row["notional"] else "–"
        why = _trade_rationale(row)
        rows_html += (
            f"<tr style='border-bottom:1px solid {_GRID}'>"
            f"<td style='color:{_MUTED};padding:6px'>{ts}</td>"
            f"<td style='color:{_TEXT};font-weight:bold;padding:6px'>{row['symbol']}</td>"
            f"<td style='padding:6px'><span style='background:{color};color:#fff;padding:2px 7px;border-radius:4px;"
            f"font-size:11px;white-space:nowrap'>{glyph} {action}</span></td>"
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
        "<th style='text-align:left;padding:6px'>Notional</th>"
        "<th style='text-align:left;padding:6px'>P&amp;L</th>"
        "<th style='text-align:left;padding:6px'>Why</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table></div>"
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
