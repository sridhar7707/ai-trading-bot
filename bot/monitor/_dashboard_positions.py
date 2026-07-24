"""Position, returns, and trade log data functions extracted from dashboard_data.py."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from loguru import logger as _logger

_MUTED = "#6b7280"
_POS   = "#15803d"
_NEG   = "#dc2626"
_FONT  = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

_ALPACA_DATA = "https://data.alpaca.markets"

_POSITION_COLS = ["Symbol", "Shares", "Avg Cost $", "Current $",
                  "Unrealized $", "Unrealized %", "Value $", "% of Portfolio", "Days Held"]

_RETURNS_COLS = ["Symbol", "Invested $", "Value $", "Return $", "Return %", "Status", "Date"]


def _con():
    import bot.monitor.dashboard_data as _dd
    db = _dd._DB
    if not Path(db).exists():
        return None
    return sqlite3.connect(db, check_same_thread=False)


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


def _prices_alpaca(symbols: list[str]) -> dict:
    """Latest prices from Alpaca snapshots (official IEX feed). {} on failure."""
    headers = _alpaca_headers()
    if headers is None:
        return {}
    try:
        import requests
        from loguru import logger
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
            p = ((snap.get("latestTrade") or {}).get("p")
                 or (snap.get("dailyBar") or {}).get("c")
                 or (snap.get("minuteBar") or {}).get("c"))
            if p:
                out[sym] = float(p)
        return out
    except Exception as exc:
        from loguru import logger
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
            if hasattr(close, "columns"):
                for s in symbols:
                    try:
                        out[s] = float(close[s].dropna().iloc[-1])
                    except Exception as exc:
                        _logger.debug(f"get_open_positions: price parse for {s}: {exc}")
            else:
                try:
                    out[symbols[0]] = float(close.dropna().iloc[-1])
                except Exception as exc:
                    _logger.debug(f"get_open_positions: single-symbol price parse: {exc}")
        return out
    except Exception as exc:
        _logger.warning(f"get_open_positions: price fetch failed, returning empty: {exc}")
        return {}


def _live_prices(symbols: list[str]) -> dict:
    """Latest prices for `symbols` — Alpaca official feed first, yfinance fallback."""
    if not symbols or not os.environ.get("SPACE_ID"):
        return {}
    import bot.monitor.dashboard_data as _dd
    out = _dd._prices_alpaca(symbols)
    if not out:
        out = _dd._prices_yfinance(symbols)
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
            value = shares * entry if entry > 0 else 0.0
            rows.append([sym, round(shares, 3), round(entry, 2), "—", "—", "—",
                         round(value, 2) if value else "—", "—", days])
    return pd.DataFrame(rows, columns=_POSITION_COLS)


def get_returns_summary_df(prices: dict | None = None) -> pd.DataFrame:
    """Per-stock investment vs total return, covering BOTH open and sold positions."""
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
        try:
            when_dt  = pd.to_datetime(when)
            when_str = ("Opened " if is_open else "Closed ") + when_dt.strftime("%b %d")
        except Exception:
            when_str = "–"

        if is_open:
            if cur is not None and entry > 0:
                invested     = net_sh * entry
                total_return = net_sh * (cur - entry) + realized
                value        = invested + total_return
                ret_pct      = total_return / invested if invested else None
                rows.append([sym, round(invested, 2), round(value, 2),
                             round(total_return, 2),
                             f"{ret_pct:+.2%}" if ret_pct is not None else "—",
                             status, when_str])
            else:
                invested = net_sh * entry if entry > 0 else buy_notional
                rows.append([sym, round(invested, 2), "—", "—", "—", status, when_str])
        else:
            invested     = buy_notional
            total_return = realized
            proceeds     = invested + realized
            ret_pct      = (realized / invested) if invested else None
            rows.append([sym, round(invested, 2), f"${proceeds:,.2f} (proceeds)",
                         round(total_return, 2),
                         f"{ret_pct:+.2%}" if ret_pct is not None else "—",
                         status, when_str])

    df = pd.DataFrame(rows, columns=_RETURNS_COLS)
    return df


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
