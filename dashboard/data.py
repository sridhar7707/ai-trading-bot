"""Shared data cache, DB sync, price fetch, and time utilities."""
from __future__ import annotations

import datetime
import os
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Generator

import pandas as pd
from loguru import logger

from dashboard.design_system import GAIN, NEURAL, TEXT2

DB_PATH    = "trades.db"
HF_TOKEN   = os.getenv("HF_TOKEN", "")
HF_REPO_ID = os.getenv("HF_DB_REPO_ID", os.getenv("HF_REPO_ID", "ksri77/ai-trading-bot-db"))

# ── Shared data cache (55-second TTL) ─────────────────────────────────────────
_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TS: float = 0.0
_CACHE_TTL: float = 55.0
_price_cache: dict = {}
_price_cache_time: dict = {}
_PRICE_CACHE_TTL: float = 3600.0

_EMPTY_CACHE: dict = {
    "open_pos": {}, "prices": {}, "trades_df": pd.DataFrame(),
    "portfolio": "—", "regime_raw": "Unknown",
    "total_trades": 0, "buy_count": 0, "sell_count": 0, "win_count": 0,
    "recent_trades": [],
    "vix": 0.0, "avg_confidence": 0.0, "sentiment_avg": 0.0,
    "latest_buy_signal": {}, "today_buy_signals": [],
}

# ── Thread-safe SQLite helpers ─────────────────────────────────────────────────
_db_locks: dict[str, threading.Lock] = {}
_db_locks_meta = threading.Lock()


def _get_db_lock(path: str) -> threading.Lock:
    with _db_locks_meta:
        if path not in _db_locks:
            _db_locks[path] = threading.Lock()
        return _db_locks[path]


@contextmanager
def get_db_conn(db_path: str | None = None, timeout: float = 5.0) -> Generator[sqlite3.Connection, None, None]:
    """Thread-safe SQLite connection context manager.

    Creates a new connection per call (new-connection-per-call = thread-safe).
    check_same_thread=False avoids spurious errors when Gradio's thread pool
    re-enters a function from a different thread between creation and use.
    """
    path = db_path or DB_PATH
    conn = sqlite3.connect(path, check_same_thread=False, timeout=timeout)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def safe_query(sql: str, params: tuple = (), db_path: str | None = None, default: Any = None) -> Any:
    """Run a SELECT and return fetchall(), or `default` on any error."""
    try:
        with get_db_conn(db_path) as conn:
            return conn.execute(sql, params).fetchall()
    except Exception as exc:
        logger.warning(f"safe_query: {exc}")
        return default


def safe_execute(sql: str, params: tuple = (), db_path: str | None = None) -> bool:
    """Run an INSERT/UPDATE/DELETE, commit, and return True on success."""
    path = db_path or DB_PATH
    lock = _get_db_lock(path)
    try:
        with lock:
            with get_db_conn(path) as conn:
                conn.execute(sql, params)
                conn.commit()
                return True
    except Exception as exc:
        logger.warning(f"safe_execute: {exc}")
        return False


def _init_db() -> None:
    """Enable WAL mode on the dashboard DB so readers don't block the bot writer."""
    if not os.path.exists(DB_PATH):
        return
    try:
        with get_db_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
    except Exception as exc:
        logger.warning(f"_init_db WAL: {exc}")


_init_db()


def _sync_db() -> None:
    if not HF_REPO_ID:
        return
    try:
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(repo_id=HF_REPO_ID, filename="trades.db",
                                  repo_type="dataset",
                                  token=HF_TOKEN or None,  # None = public repos work without token
                                  force_download=True)
        shutil.copy(cached, DB_PATH)
    except Exception as e:
        msg = str(e).lower()
        if any(x in msg for x in ("404", "not found", "entry", "does not exist")):
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)
                logger.info("DB sync: trades.db deleted from HF — local copy removed")
        else:
            logger.opt(exception=True).warning(f"DB sync: {e}")
    for filename, dest in [
        ("validation_report.json",  "models/validation_report.json"),
        ("feature_importance.json", "models/feature_importance.json"),
    ]:
        try:
            from huggingface_hub import hf_hub_download
            cached = hf_hub_download(repo_id=HF_REPO_ID, filename=filename,
                                      repo_type="dataset", token=HF_TOKEN or None, force_download=True)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(cached, dest)
        except Exception as exc:
            logger.debug(f"hf_artifact_download: {exc}")


def _current_prices(symbols: list) -> dict:
    if not symbols:
        return {}
    try:
        import yfinance as yf
        df = yf.download(" ".join(symbols), period="2d", progress=False, auto_adjust=True)
        if df.empty:
            return {s: 0.0 for s in symbols}
        close = df["Close"]
        prices = {}
        for sym in symbols:
            try:
                col = close[sym] if isinstance(close, pd.DataFrame) else close
                prices[sym] = float(col.dropna().iloc[-1])
            except Exception:
                prices[sym] = 0.0
        return prices
    except Exception as e:
        logger.warning(f"Price fetch: {e}")
        return {s: 0.0 for s in symbols}


def _refresh_cache() -> dict:
    """One DB read + one yfinance call; derives everything all render fns need."""
    _sync_db()
    result = dict(_EMPTY_CACHE)
    if not os.path.exists(DB_PATH):
        return result
    try:
        with get_db_conn() as con:
            for _col in (
                "xgb_prob REAL DEFAULT 0.0",
                "lstm_prob REAL DEFAULT 0.0",
                "sentiment_score REAL DEFAULT 0.0",
                "macro_score REAL DEFAULT 0.0",
                "ensemble_score REAL DEFAULT 0.0",
                "realized_pnl REAL DEFAULT 0.0",
                "order_id TEXT DEFAULT NULL",
                "holding_days INTEGER DEFAULT 0",
                "feature_drivers TEXT DEFAULT NULL",
            ):
                try:
                    con.execute(f"ALTER TABLE trades ADD COLUMN {_col}")
                    con.commit()
                except sqlite3.OperationalError:
                    pass
            try:
                df = pd.read_sql(
                    "SELECT id,timestamp,symbol,action,shares,price,notional,"
                    "pnl_pct,portfolio_value,regime,"
                    "COALESCE(ensemble_score,0.0) AS ensemble_score,"
                    "COALESCE(sentiment_score,0.0) AS sentiment_score,"
                    "COALESCE(xgb_prob,0.0)       AS xgb_prob,"
                    "COALESCE(lstm_prob,0.0)       AS lstm_prob,"
                    "feature_drivers "
                    "FROM trades ORDER BY id", con)
            except Exception as _e:
                logger.opt(exception=True).warning(f"Extended trades query failed (missing columns?): {_e} — falling back to base schema")
                df = pd.read_sql(
                    "SELECT id,timestamp,symbol,action,shares,price,notional,"
                    "pnl_pct,portfolio_value,regime FROM trades ORDER BY id", con)
                df["ensemble_score"] = 0.0
                df["sentiment_score"] = 0.0
                df["xgb_prob"]        = 0.0
                df["lstm_prob"]       = 0.0
                df["feature_drivers"] = None
    except Exception as e:
        logger.opt(exception=True).warning(f"DB read: {e}")
        return result

    if df.empty:
        return result

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.date
    result["trades_df"] = df

    result["total_trades"] = len(df)
    result["buy_count"]    = int((df["action"] == "BUY").sum())
    sells_mask             = df["action"].str.startswith("SELL") & (df["action"] != "SELL_RECONCILE")
    result["sell_count"]   = int(sells_mask.sum())
    result["win_count"]    = int((sells_mask & (df["pnl_pct"] > 0)).sum())

    last = df.iloc[-1]
    result["portfolio"]  = (f"${last['portfolio_value']:,.2f}"
                            if pd.notna(last["portfolio_value"]) else "—")
    result["regime_raw"] = (str(last["regime"] or "Unknown")).replace("_", " ")

    pos: dict = {}
    for _, row in df.iterrows():
        sym = row["symbol"]
        shares   = row["shares"]   or 0.0
        notional = row["notional"] or 0.0
        if row["action"] == "BUY":
            if sym not in pos:
                pos[sym] = {"shares": 0.0, "invested": 0.0}
            pos[sym]["shares"]   += shares
            pos[sym]["invested"] += notional
        elif row["action"].startswith("SELL") and sym in pos and pos[sym]["shares"] > 0:
            avg = pos[sym]["invested"] / pos[sym]["shares"]
            pos[sym]["shares"]   = max(0.0, pos[sym]["shares"] - shares)
            pos[sym]["invested"] = max(0.0, pos[sym]["invested"] - avg * shares)
    result["open_pos"] = {s: d for s, d in pos.items() if d["shares"] > 0.001}

    recent = df.tail(15).iloc[::-1][
        ["timestamp", "symbol", "action", "shares", "price", "notional", "pnl_pct", "regime"]
    ]
    result["recent_trades"] = list(recent.itertuples(index=False, name=None))

    fetch_syms = list(result["open_pos"].keys()) + ["^VIX"]
    all_prices = _current_prices(fetch_syms)
    result["prices"] = {k: v for k, v in all_prices.items() if k != "^VIX"}
    result["vix"]    = all_prices.get("^VIX", 0.0)

    buys_df = df[df["action"] == "BUY"]
    if not buys_df.empty:
        result["avg_confidence"] = float(buys_df.tail(5)["ensemble_score"].mean())
        result["sentiment_avg"]  = float(buys_df.tail(20)["sentiment_score"].mean())
        result["latest_buy_signal"] = buys_df.iloc[-1].to_dict()
        today_str  = str(datetime.date.today())
        today_buys = buys_df[buys_df["date"].astype(str) == today_str]
        if today_buys.empty:
            today_buys = buys_df.tail(10)
        result["today_buy_signals"] = today_buys.iloc[::-1].to_dict("records")

    return result


def get_data() -> dict:
    """Return cached data, refreshing if TTL has elapsed."""
    global _CACHE, _CACHE_TS
    now = time.time()
    with _CACHE_LOCK:
        if _CACHE and (now - _CACHE_TS) < _CACHE_TTL:
            return _CACHE
        _CACHE = _refresh_cache()
        _CACHE_TS = now
    return _CACHE


def _now_ct() -> str:
    try:
        from zoneinfo import ZoneInfo
        ct = datetime.datetime.now(datetime.timezone.utc).astimezone(ZoneInfo("America/Chicago"))
        label = "CDT" if ct.dst() and ct.dst().total_seconds() else "CST"
        return ct.strftime(f"%b %d, %Y &nbsp;%H:%M {label}")
    except Exception as exc:
        logger.debug(f"_ct_now timezone: {exc}")
        return datetime.datetime.utcnow().strftime("%H:%M UTC")

def _to_ct(ts) -> str:
    try:
        from zoneinfo import ZoneInfo
        if isinstance(ts, datetime.datetime):
            dt = ts
        else:
            dt = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        ct = dt.astimezone(ZoneInfo("America/Chicago"))
        label = "CDT" if ct.dst() and ct.dst().total_seconds() else "CST"
        return ct.strftime(f"%Y-%m-%d %H:%M {label}")
    except Exception as exc:
        logger.debug(f"_to_ct parse: {exc}")
        return str(ts)[:16].replace("T", " ")

def _market_status() -> tuple:
    """Returns (label, color)."""
    try:
        from zoneinfo import ZoneInfo
        et = datetime.datetime.now(ZoneInfo("America/New_York"))
        if et.weekday() >= 5:
            return "Weekend", TEXT2
        open_t  = et.replace(hour=9,  minute=30, second=0, microsecond=0)
        close_t = et.replace(hour=16, minute=0,  second=0, microsecond=0)
        if open_t <= et < close_t:
            return "Market Open", GAIN
        elif et < open_t:
            return "Pre-Market", NEURAL
        return "After Hours", TEXT2
    except Exception as exc:
        logger.debug(f"_market_status: {exc}")
        return "—", TEXT2
