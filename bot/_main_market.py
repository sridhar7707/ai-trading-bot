"""Market data, universe, and sentiment helpers extracted from bot/main.py."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from loguru import logger

from bot.strategy.reddit_sentiment import get_wsb_sentiment
from config import EARNINGS_WINDOW_DAYS, MARKET_OPEN_BUFFER_MINS, MARKET_CLOSE_BUFFER_MINS, SYMBOLS

_UNIVERSE_PATH   = "data/universe_today.json"
# All ETF-like instruments in the universe — no earnings dates, skip earnings prefetch.
# Keep in sync with config.SYMBOLS whenever new ETFs are added.
_ETF_SYMBOLS     = {"VOO", "QQQ", "SPY", "VTI", "ARKK", "IWM", "GLD", "XLE", "XLF", "XLV"}
_EARNINGS_DB_TTL = 12 * 3600

_wsb_cache: dict[str, tuple[float, dict]] = {}
_WSB_CACHE_TTL = 300  # seconds — matches the trading cycle interval
_market_holiday_cache: dict[str, bool] = {}  # date_str → is_holiday (one Alpaca API call per day)

_US_MARKET_HOLIDAYS = {
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18",
    "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
    "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
    "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-12-25",
}


def _load_today_universe() -> tuple[list[str], dict]:
    """Return (symbols, payload) for today's screened universe, or (config.SYMBOLS, {})."""
    if not os.path.exists(_UNIVERSE_PATH):
        return list(SYMBOLS), {}
    try:
        with open(_UNIVERSE_PATH) as f:
            payload = json.load(f)
        if payload.get("date") != date.today().isoformat():
            logger.info("Universe file is from a prior day — using config.SYMBOLS")
            return list(SYMBOLS), {}
        syms = payload.get("symbols", [])
        if not syms:
            return list(SYMBOLS), {}
        logger.info(f"Loaded screened universe: {len(syms)} symbols ({syms[:5]}...)")
        return syms, payload
    except Exception as exc:
        logger.warning(f"Failed to load screened universe: {exc} — using config.SYMBOLS")
        return list(SYMBOLS), {}


def _import_screener_picks(con, payload: dict) -> None:
    """Write pre-market screener factor scores from universe_today.json into screener_log.

    Called once per session (first cycle) — subsequent calls are no-ops because we
    check whether the screened_at timestamp is already present.  This is the only
    path for screener data to reach the bot's trades.db: the premarket runner's
    local DB is discarded; only universe_today.json crosses the job boundary via cache.
    """
    picks      = payload.get("picks")
    screened_at = payload.get("screened_at")
    if not picks or not screened_at:
        return
    try:
        existing = con.execute(
            "SELECT COUNT(*) FROM screener_log WHERE screened_at = ?", (screened_at,)
        ).fetchone()[0]
        if existing:
            return
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        con.execute("DELETE FROM screener_log WHERE screened_at < ?", (cutoff,))
        rows = [
            (screened_at, p["symbol"], p.get("rank"), p.get("composite_score"),
             p.get("analyst_signal", 0.0), p.get("etf_momentum"),
             p.get("regime"), p.get("sector"))
            for p in picks
        ]
        con.executemany(
            "INSERT INTO screener_log "
            "(screened_at,symbol,rank,composite_score,analyst_signal,etf_momentum,regime,sector) "
            "VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        con.commit()
        logger.info(f"Imported {len(rows)} screener picks into screener_log")
    except Exception as exc:
        logger.warning(f"screener_log import failed (non-fatal): {exc}")


def _log_buy_skip(symbol: str, reason: str) -> None:
    """Log a standardized reason why a candidate buy was skipped."""
    logger.info(f"BUY {symbol} skipped — {reason}")


def _is_market_hours(alpaca_api=None) -> bool:
    import zoneinfo
    et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return False
    today_str = et.strftime("%Y-%m-%d")
    # Cache the holiday result per day — only one Alpaca API call per session
    is_holiday = _market_holiday_cache.get(today_str)
    if is_holiday is None:
        if alpaca_api is not None:
            try:
                cal = alpaca_api.get_calendar(start=today_str, end=today_str)
                is_holiday = len(cal) == 0
            except Exception as e:
                logger.warning(f"Alpaca calendar check failed — using hardcoded holidays: {e}")
                is_holiday = today_str in _US_MARKET_HOLIDAYS
        else:
            is_holiday = today_str in _US_MARKET_HOLIDAYS
        _market_holiday_cache[today_str] = is_holiday
    if is_holiday:
        logger.info("NYSE holiday — skipping cycle.")
        return False
    base = et.replace(second=0, microsecond=0)
    tradeable_open  = base.replace(hour=9,  minute=30) + timedelta(minutes=MARKET_OPEN_BUFFER_MINS)
    tradeable_close = base.replace(hour=16, minute=0)  - timedelta(minutes=MARKET_CLOSE_BUFFER_MINS)
    in_window = tradeable_open <= et < tradeable_close
    if not in_window:
        logger.info(
            f"Outside tradeable window ({tradeable_open.strftime('%H:%M')}–"
            f"{tradeable_close.strftime('%H:%M')} ET) — skipping cycle."
        )
    return in_window


def _is_near_earnings(con, symbol: str) -> bool:
    if symbol in _ETF_SYMBOLS:
        return False
    now = time.time()
    row = con.execute(
        "SELECT near_earnings, cached_at FROM earnings_cache WHERE symbol=?", (symbol,)
    ).fetchone()
    if row:
        try:
            if now - datetime.fromisoformat(row[1]).timestamp() < _EARNINGS_DB_TTL:
                return bool(row[0])
        except (ValueError, TypeError):
            pass
    try:
        import yfinance as yf
        cal = yf.Ticker(symbol).calendar
        if cal is None:
            near = False
        else:
            dates = cal.get("Earnings Date", []) if isinstance(cal, dict) else (
                cal.loc["Earnings Date"].tolist() if "Earnings Date" in cal.index else []
            )
            if not dates:
                near = False
            else:
                nearest = pd.to_datetime(dates[0]).date()
                near = abs((nearest - date.today()).days) <= EARNINGS_WINDOW_DAYS
                if near:
                    logger.info(f"Earnings guard: {symbol} — {nearest} within {EARNINGS_WINDOW_DAYS}d")
    except Exception as e:
        logger.warning(f"Earnings check failed for {symbol} — assuming safe: {e}")
        near = False
    con.execute(
        "INSERT OR REPLACE INTO earnings_cache (symbol, near_earnings, cached_at) VALUES (?,?,?)",
        (symbol, int(near), datetime.now(timezone.utc).isoformat())
    )
    con.commit()
    return near


def _prefetch_earnings_parallel(con, symbols: list[str]) -> dict[str, bool]:
    """Bulk-fetch earnings proximity: one SQL read, parallel yfinance for misses, one batch write.

    Replaces 25 sequential yfinance HTTP calls (1–5 s each on cache miss) with a
    single parallel burst capped at 8 threads.
    """
    now = time.time()
    placeholders = ",".join("?" * len(symbols))
    rows = con.execute(
        f"SELECT symbol, near_earnings, cached_at FROM earnings_cache WHERE symbol IN ({placeholders})",
        symbols,
    ).fetchall()

    result: dict[str, bool] = {}
    for sym, near, cached_at in rows:
        try:
            if now - datetime.fromisoformat(cached_at).timestamp() < _EARNINGS_DB_TTL:
                result[sym] = bool(near)
        except (ValueError, TypeError):
            pass

    for sym in symbols:
        if sym in _ETF_SYMBOLS:
            result[sym] = False

    stale = [s for s in symbols if s not in result and s not in _ETF_SYMBOLS]
    if not stale:
        return {s: result.get(s, False) for s in symbols}

    def _fetch_one(symbol: str) -> tuple[str, bool]:
        try:
            import yfinance as yf
            cal = yf.Ticker(symbol).calendar
            if cal is None:
                return symbol, False
            dates = cal.get("Earnings Date", []) if isinstance(cal, dict) else (
                cal.loc["Earnings Date"].tolist() if "Earnings Date" in cal.index else []
            )
            if not dates:
                return symbol, False
            nearest = pd.to_datetime(dates[0]).date()
            near = abs((nearest - date.today()).days) <= EARNINGS_WINDOW_DAYS
            if near:
                logger.info(f"Earnings guard: {symbol} — {nearest} within {EARNINGS_WINDOW_DAYS}d")
            return symbol, near
        except Exception as e:
            logger.warning(f"Earnings prefetch failed for {symbol}: {e}")
            return symbol, False

    logger.info(f"Earnings prefetch: {len(stale)} cache misses — fetching in parallel")
    with ThreadPoolExecutor(max_workers=min(len(stale), 8)) as pool:
        fresh = dict(pool.map(_fetch_one, stale))

    ts = datetime.now(timezone.utc).isoformat()
    con.executemany(
        "INSERT OR REPLACE INTO earnings_cache (symbol, near_earnings, cached_at) VALUES (?,?,?)",
        [(sym, int(near), ts) for sym, near in fresh.items()],
    )
    con.commit()
    result.update(fresh)
    return {s: result.get(s, False) for s in symbols}


def _wsb(symbol: str) -> tuple[str, dict]:
    """Fetch WSB sentiment with a 5-min module-level cache to avoid Reddit rate limits."""
    now = time.time()
    cached_ts, cached_result = _wsb_cache.get(symbol, (0.0, None))
    if cached_result is not None and now - cached_ts < _WSB_CACHE_TTL:
        return symbol, cached_result
    try:
        result = get_wsb_sentiment(symbol)
    except Exception as exc:
        logger.debug(f"wsb_sentiment_fetch {symbol}: {exc}")
        result = {"mentions": 0, "sentiment": 0.0}
    _wsb_cache[symbol] = (now, result)
    return symbol, result


def _load_premarket_sentiment() -> dict[str, float]:
    """Load pre-computed FinBERT scores from today's prefetch run, if available."""
    path = "data/sentiment_today.json"
    try:
        if os.path.exists(path):
            with open(path) as f:
                payload = json.load(f)
            if payload.get("date") == date.today().isoformat():
                scores = payload.get("scores", {})
                if scores:
                    logger.info(f"Loaded pre-market sentiment: {len(scores)} symbols")
                    return scores
    except Exception as e:
        logger.warning(f"Failed to load pre-market sentiment: {e}")
    return {}
