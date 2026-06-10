"""Main trading loop — runs every 5 minutes via GitHub Actions."""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
import traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date, datetime, timezone
from loguru import logger

_UNIVERSE_PATH  = "data/universe_today.json"
_HALT_FILE      = "data/HALT_TRADING"   # create this file to pause the bot without canceling the workflow


def _load_today_universe() -> list[str]:
    """Return today's screened universe if available, else config.SYMBOLS."""
    from config import SYMBOLS as _fallback
    if not os.path.exists(_UNIVERSE_PATH):
        return list(_fallback)
    try:
        with open(_UNIVERSE_PATH) as f:
            payload = json.load(f)
        if payload.get("date") != date.today().isoformat():
            logger.info("Universe file is from a prior day — using config.SYMBOLS")
            return list(_fallback)
        syms = payload.get("symbols", [])
        if not syms:
            return list(_fallback)
        logger.info(f"Loaded screened universe: {len(syms)} symbols ({syms[:5]}...)")
        return syms
    except Exception as exc:
        logger.warning(f"Failed to load screened universe: {exc} — using config.SYMBOLS")
        return list(_fallback)

from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from config import (
    SYMBOLS, TRADE_DB_PATH,
    MARKET_OPEN_BUFFER_MINS, MARKET_CLOSE_BUFFER_MINS,
    EARNINGS_WINDOW_DAYS,
    MAX_HOLD_DAYS, KELLY_LOOKBACK_TRADES, KELLY_FRACTION_MAX,
    CORRELATION_THRESHOLD, RS_LOOKBACK_BARS, ENTRY_REGIMES,
)
from bot.execution.alpaca_client import AlpacaClient
from bot.strategy.features import compute_features, FEATURE_COLS
from bot.strategy.regime_classifier import RegimeClassifier
from bot.strategy.xgb_predictor import XGBPredictor
from bot.strategy.lstm_predictor import LSTMPredictor
from bot.strategy.sentiment import batch_sentiment_scores
from bot.strategy.macro import _fetch_macro_raw, _compute_from_raw
from bot.strategy.reddit_sentiment import get_wsb_sentiment
from bot.strategy.ensemble import ensemble_signal, action_to_int, BUY_FRACTION, WEIGHTS
from bot.risk.risk_manager import RiskManager
import bot.monitor.telegram_bot as tg

os.makedirs("logs", exist_ok=True)
logger.add("logs/trading.log", rotation="1 week", retention="4 weeks", level="INFO")

_US_MARKET_HOLIDAYS = {
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18",
    "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
    "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
    "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-12-25",
}

_ETF_SYMBOLS      = {"VOO", "QQQ", "SPY", "VTI", "ARKK"}
_EARNINGS_DB_TTL  = 12 * 3600
_MACRO_DB_TTL     = 4  * 3600


def _is_market_hours(alpaca_api=None) -> bool:
    import zoneinfo
    from datetime import timedelta
    et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return False
    today_str = et.strftime("%Y-%m-%d")
    is_holiday = False
    if alpaca_api is not None:
        try:
            cal = alpaca_api.get_calendar(start=today_str, end=today_str)
            is_holiday = len(cal) == 0  # empty list = market closed today
        except Exception as e:
            logger.warning(f"Alpaca calendar check failed — using hardcoded holidays: {e}")
            is_holiday = today_str in _US_MARKET_HOLIDAYS
    else:
        is_holiday = today_str in _US_MARKET_HOLIDAYS
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


def init_db():
    con = sqlite3.connect(TRADE_DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, symbol TEXT, action TEXT,
            shares REAL, price REAL, notional REAL,
            regime TEXT, portfolio_value REAL, pnl_pct REAL,
            xgb_prob REAL DEFAULT 0.0,
            lstm_prob REAL DEFAULT 0.0,
            sentiment_score REAL DEFAULT 0.0,
            macro_score REAL DEFAULT 0.0,
            ensemble_score REAL DEFAULT 0.0
        )
    """)
    # Migration: add signal audit columns to existing DBs (safe no-op if already present)
    for _col in (
        "xgb_prob REAL DEFAULT 0.0",
        "lstm_prob REAL DEFAULT 0.0",
        "sentiment_score REAL DEFAULT 0.0",
        "macro_score REAL DEFAULT 0.0",
        "ensemble_score REAL DEFAULT 0.0",
    ):
        try:
            con.execute(f"ALTER TABLE trades ADD COLUMN {_col}")
        except sqlite3.OperationalError:
            pass
    con.execute("""
        CREATE TABLE IF NOT EXISTS position_state (
            symbol TEXT PRIMARY KEY, entry_price REAL,
            high_water_mark REAL, atr_at_entry REAL, opened_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS risk_state (
            key TEXT PRIMARY KEY, value TEXT, updated_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS earnings_cache (
            symbol TEXT PRIMARY KEY, near_earnings INTEGER, cached_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS macro_cache (
            key TEXT PRIMARY KEY, value REAL, cached_at TEXT
        )
    """)
    con.commit()
    return con


def log_trade(con, symbol, action, shares, price, notional, regime, portfolio_value, pnl_pct,
              xgb_prob: float = 0.0, lstm_prob: float = 0.0,
              sentiment_score: float = 0.0, macro_score: float = 0.0):
    sentiment_norm  = (sentiment_score + 1.0) / 2.0
    ensemble_score  = (WEIGHTS["xgb"]  * xgb_prob + WEIGHTS["lstm"] * lstm_prob
                       + WEIGHTS["sentiment"] * sentiment_norm + WEIGHTS["macro"] * macro_score)
    con.execute(
        "INSERT INTO trades VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(),
         symbol, action, shares, price, notional, regime, portfolio_value, pnl_pct,
         xgb_prob, lstm_prob, sentiment_score, macro_score, ensemble_score),
    )
    con.commit()


def _opened_today(con, symbol: str) -> bool:
    today = date.today().isoformat()
    row = con.execute(
        "SELECT 1 FROM trades WHERE symbol=? AND action='BUY' AND timestamp LIKE ? LIMIT 1",
        (symbol, today + "%"),
    ).fetchone()
    return row is not None


# ── Risk state persistence ────────────────────────────────────────────────────

def _week_key() -> str:
    """Return current ISO year-week string, e.g. '2026-W24'."""
    return date.today().strftime("%G-W%V")


def _load_risk_state(con) -> tuple[float | None, list[str], float | None, bool, bool]:
    """Returns (daily_start, day_trade_dates, weekly_start, daily_warning_sent, weekly_halt_alerted)."""
    today = date.today().isoformat()
    wk    = _week_key()
    rows  = {r[0]: r[1] for r in con.execute("SELECT key, value FROM risk_state")}

    daily_start: float | None = None
    if rows.get("daily_start_date") == today:
        try:
            daily_start = float(rows["daily_start_value"])
        except (KeyError, ValueError, TypeError):
            pass

    day_trade_dates: list[str] = []
    try:
        day_trade_dates = json.loads(rows.get("day_trade_dates", "[]"))
    except (json.JSONDecodeError, TypeError):
        pass

    weekly_start: float | None = None
    if rows.get("weekly_start_week") == wk:
        try:
            weekly_start = float(rows["weekly_start_value"])
        except (KeyError, ValueError, TypeError):
            pass

    daily_warning_sent  = rows.get("daily_warning_sent_date") == today
    weekly_halt_alerted = rows.get("weekly_halt_alerted_week") == wk

    return daily_start, day_trade_dates, weekly_start, daily_warning_sent, weekly_halt_alerted


def _save_risk_state(con, risk: RiskManager):
    today = date.today().isoformat()
    wk    = _week_key()
    trades  = json.dumps([d.isoformat() for d in risk.day_trade_log])
    start   = str(risk.daily_start_value)  if risk.daily_start_value  is not None else ""
    weekly  = str(risk.weekly_start_value) if risk.weekly_start_value is not None else ""
    entries = [
        ("daily_start_value",       start),
        ("daily_start_date",        today),
        ("day_trade_dates",         trades),
        ("weekly_start_value",      weekly),
        ("weekly_start_week",       wk),
        ("daily_warning_sent_date", today if risk.daily_warning_sent else ""),
        ("weekly_halt_alerted_week", wk   if risk.weekly_halt_alerted else ""),
    ]
    for key, val in entries:
        con.execute(
            "INSERT OR REPLACE INTO risk_state (key, value, updated_at) VALUES (?,?,?)",
            (key, val, datetime.now(timezone.utc).isoformat())
        )
    con.commit()


# ── Macro cache (DB-backed, survives subprocess restarts) ─────────────────────

def _get_macro_from_db(con) -> tuple[float, float, bool]:
    """Returns (score, cap, halt). halt=True means VIX >= MACRO_HALT_VIX — block all new buys."""
    now  = time.time()
    rows = {r[0]: (float(r[1]), r[2])
            for r in con.execute("SELECT key, value, cached_at FROM macro_cache")}
    if "score" in rows and "cap" in rows:
        try:
            cached_ts = datetime.fromisoformat(rows["score"][1]).timestamp()
            if now - cached_ts < _MACRO_DB_TTL:
                halt = bool(rows["halt"][0]) if "halt" in rows else False
                return rows["score"][0], rows["cap"][0], halt
        except (ValueError, TypeError):
            pass
    try:
        result = _compute_from_raw(_fetch_macro_raw())
    except Exception as e:
        logger.warning(f"Macro fetch failed — using neutral defaults: {e}")
        result = {"score": 0.5, "cap": 1.0, "halt": False}
    ts = datetime.now(timezone.utc).isoformat()
    for key in ("score", "cap", "halt"):
        con.execute(
            "INSERT OR REPLACE INTO macro_cache (key, value, cached_at) VALUES (?,?,?)",
            (key, float(result[key]), ts)
        )
    con.commit()
    return result["score"], result["cap"], bool(result["halt"])


# ── Earnings calendar ─────────────────────────────────────────────────────────

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


# ── Position state helpers ────────────────────────────────────────────────────

def _load_position_state(con, symbol: str) -> dict | None:
    row = con.execute(
        "SELECT entry_price, high_water_mark, atr_at_entry, opened_at FROM position_state WHERE symbol=?",
        (symbol,),
    ).fetchone()
    return ({"entry_price": row[0], "high_water_mark": row[1],
              "atr_at_entry": row[2], "opened_at": row[3]} if row else None)


def _upsert_position_state(con, symbol: str, entry_price: float,
                            high_water_mark: float, atr: float):
    con.execute("""
        INSERT INTO position_state (symbol, entry_price, high_water_mark, atr_at_entry, opened_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            high_water_mark = MAX(high_water_mark, excluded.high_water_mark),
            atr_at_entry    = excluded.atr_at_entry
    """, (symbol, entry_price, high_water_mark, atr, datetime.now(timezone.utc).isoformat()))
    con.commit()


def _delete_position_state(con, symbol: str):
    con.execute("DELETE FROM position_state WHERE symbol=?", (symbol,))
    con.commit()


# ── Advanced entry helpers ────────────────────────────────────────────────────

def _kelly_fraction(con, symbol: str, default: float = BUY_FRACTION) -> float:
    """
    Half-Kelly position fraction estimated from recent closed trades for this symbol.
    Returns `default` when fewer than 10 observations exist (not enough data).
    """
    rows = con.execute(
        "SELECT pnl_pct FROM trades WHERE symbol=? AND action LIKE 'SELL%' "
        "ORDER BY timestamp DESC LIMIT ?",
        (symbol, KELLY_LOOKBACK_TRADES),
    ).fetchall()
    if len(rows) < 10:
        return default
    pnls   = [r[0] for r in rows]
    wins   = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p <= 0]
    if not wins or not losses:
        return default
    win_rate = len(wins) / len(pnls)
    b        = (sum(wins) / len(wins)) / (sum(losses) / len(losses))
    kelly    = (win_rate * b - (1 - win_rate)) / b
    half_k   = max(0.02, min(KELLY_FRACTION_MAX, kelly * 0.5))
    logger.debug(f"Kelly {symbol}: win_rate={win_rate:.2f}, b={b:.2f}, half_f={half_k:.3f}")
    return half_k


def _passes_correlation_gate(symbol: str, positions: dict, bars_map: dict) -> bool:
    """Block buy if any held position has > CORRELATION_THRESHOLD 5-min return correlation."""
    bars_sym = bars_map.get(symbol)
    if bars_sym is None or bars_sym.empty:
        return True
    ret_sym = bars_sym["close"].pct_change().dropna()
    for held in positions:
        if held == symbol:
            continue
        bars_held = bars_map.get(held)
        if bars_held is None or bars_held.empty:
            continue
        ret_held  = bars_held["close"].pct_change().dropna()
        common    = ret_sym.index.intersection(ret_held.index)
        if len(common) < 20:
            continue
        corr = float(ret_sym.loc[common].corr(ret_held.loc[common]))
        if not math.isnan(corr) and corr > CORRELATION_THRESHOLD:
            logger.info(
                f"Correlation gate: {symbol} blocked — {corr:.2f} correlation with held {held}"
            )
            return False
    return True


def _check_time_exit(pos_state: dict | None, pnl_pct: float) -> bool:
    """Return True if position has been held too long with insufficient gain."""
    if not pos_state or not pos_state.get("opened_at"):
        return False
    try:
        opened  = datetime.fromisoformat(pos_state["opened_at"]).replace(tzinfo=timezone.utc)
        days    = (datetime.now(timezone.utc) - opened).days
        if days >= MAX_HOLD_DAYS and pnl_pct < 0.01:
            logger.info(
                f"Time exit: position held {days}d, pnl={pnl_pct:.2%} — freeing capital"
            )
            return True
    except (ValueError, TypeError):
        pass
    return False


def _maybe_record_day_trade(con, risk: RiskManager, symbol: str, sell_success: bool):
    """Record PDT day trade for ANY exit on a position opened today."""
    if sell_success and _opened_today(con, symbol):
        risk.record_day_trade()
        _save_risk_state(con, risk)


def _reconcile_positions(con, alpaca_positions: dict):
    """Sync position_state table with Alpaca's live positions at startup.
    Removes stale DB entries for positions closed externally;
    seeds DB entries for positions opened manually/outside the bot.
    """
    db_syms = {r[0] for r in con.execute("SELECT symbol FROM position_state").fetchall()}
    for sym in db_syms - set(alpaca_positions.keys()):
        logger.warning(f"Reconcile: {sym} in DB but not Alpaca — removing stale state")
        _delete_position_state(con, sym)
    for sym, pos in alpaca_positions.items():
        if sym not in db_syms:
            entry = float(getattr(pos, "avg_entry_price", 0) or 0)
            logger.warning(f"Reconcile: {sym} in Alpaca but not DB — seeding position state")
            _upsert_position_state(con, sym, entry, entry, 0.0)


def _signal_sell(con, client, symbol, pos_qty, current_price,
                 regime_name, portfolio_value, is_from_stop=False,
                 reason="stop-loss", pnl_pct=0.0) -> bool:
    sell_result = client.sell(symbol, qty=pos_qty, limit_price=current_price)
    if sell_result:
        filled = client.wait_for_fill(sell_result["order_id"], timeout_secs=12)
        if not filled and is_from_stop:
            # Stop-loss must execute — escalate to market order immediately
            logger.warning(f"Stop limit timed out for {symbol} — escalating to market sell")
            sell_result = client.sell_market(symbol, pos_qty)
            if sell_result:
                client.wait_for_fill(sell_result["order_id"], timeout_secs=10)
    if sell_result:
        sell_notional = pos_qty * current_price  # actual sell value for audit trail
        if is_from_stop:
            tg.alert_stop_loss(symbol, pnl_pct, notional=sell_notional)
            log_trade(con, symbol, "SELL_STOP", pos_qty, current_price, sell_notional,
                      regime_name, portfolio_value, pnl_pct)
        else:
            action_tag = "SELL" if reason == "signal" else f"SELL_{reason.upper().replace('-','_')}"
            tg.alert_sell(symbol, pos_qty, current_price, pnl_pct, reason=reason, notional=sell_notional)
            log_trade(con, symbol, action_tag, pos_qty, current_price, sell_notional,
                      regime_name, portfolio_value, pnl_pct)
        _delete_position_state(con, symbol)
        return True
    if is_from_stop:
        tg.alert_sell_failed(symbol, reason=reason)
    logger.error(f"SELL ({reason}) failed for {symbol} — will retry next cycle")
    return False


# ── End-of-day summary ────────────────────────────────────────────────────────

def end_of_day_summary():
    import zoneinfo
    con    = init_db()
    client = AlpacaClient()
    today  = date.today().isoformat()
    trades_count = con.execute(
        "SELECT COUNT(*) FROM trades WHERE timestamp LIKE ?", (today + "%",)
    ).fetchone()[0]
    daily_start, day_trade_dates, weekly_start, _, __ = _load_risk_state(con)
    day_trade_count = day_trade_dates.count(today)
    portfolio_value, available_cash = client.get_account_summary()
    positions = client.get_positions()
    day_return = ((portfolio_value - daily_start) / daily_start) if daily_start else 0.0
    try:
        spy_bars = client.get_bars("SPY", timeframe="1Day", limit=2)
        vs_spy   = float(spy_bars["close"].pct_change().iloc[-1]) if len(spy_bars) > 1 else 0.0
    except Exception:
        vs_spy = 0.0
    tg.alert_daily_summary(
        day_return=day_return,
        vs_spy=vs_spy,
        positions=list(positions.keys()),
        cash=available_cash,
        trades=trades_count,
        day_trades=day_trade_count,
    )
    logger.info(f"End-of-day summary sent: return={day_return:.2%}, trades={trades_count}")

    # Friday weekly report — Portfolio Manager visibility into week performance
    et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    if et.weekday() == 4:  # Friday
        try:
            from datetime import timedelta
            monday = date.today() - timedelta(days=date.today().weekday())
            week_rows = con.execute(
                "SELECT action, pnl_pct, portfolio_value FROM trades WHERE timestamp >= ?",
                (monday.isoformat(),)
            ).fetchall()
            week_sells  = [r for r in week_rows if r[0].startswith("SELL")]
            week_wins   = [r for r in week_sells if r[1] > 0]
            week_wr     = len(week_wins) / len(week_sells) if week_sells else 0.0
            week_return = ((portfolio_value - weekly_start) / weekly_start) if weekly_start else 0.0
            pv_week     = [r[2] for r in week_rows if r[2] > 0]
            week_dd     = 0.0
            if pv_week:
                pk = pv_week[0]
                for v in pv_week:
                    pk = max(pk, v)
                    week_dd = max(week_dd, (pk - v) / (pk + 1e-8))
            spy_wk = 0.0
            try:
                spy_wk_bars = client.get_bars("SPY", timeframe="1Day", limit=6)
                if len(spy_wk_bars) > 1:
                    spy_wk = float(spy_wk_bars["close"].iloc[-1] / spy_wk_bars["close"].iloc[-6] - 1)
            except Exception:
                pass
            tg.alert_weekly_report(
                week_return=week_return,
                vs_spy=spy_wk,
                win_rate=week_wr,
                sharpe=0.0,   # requires full intraday history; omitted for simplicity
                drawdown=week_dd,
            )
            logger.info(f"Weekly report sent: return={week_return:.2%}, win_rate={week_wr:.1%}")
        except Exception as e:
            logger.warning(f"Weekly report failed: {e}")

    con.close()


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


# ── Main trading cycle ────────────────────────────────────────────────────────

def run(mode: str = "paper"):
    logger.info(f"=== Trading cycle start | mode={mode} ===")

    # Emergency override: create data/HALT_TRADING file to pause without canceling the workflow.
    # Portfolio Manager can push this file to HuggingFace or add it via GitHub manual trigger.
    if os.path.exists(_HALT_FILE):
        logger.warning("HALT_TRADING file detected — cycle skipped. Remove file to resume.")
        tg._send("⛔ <b>EMERGENCY HALT ACTIVE</b> — bot paused. Delete data/HALT_TRADING to resume.")
        return

    client = AlpacaClient()
    if not _is_market_hours(client.api):
        return

    con = init_db()

    active_symbols = _load_today_universe()

    daily_start, day_trade_dates, weekly_start, daily_warning_sent, weekly_halt_alerted = _load_risk_state(con)
    risk = RiskManager(
        daily_start_value=daily_start,
        day_trade_dates=day_trade_dates,
        weekly_start_value=weekly_start,
        daily_warning_sent=daily_warning_sent,
        weekly_halt_alerted=weekly_halt_alerted,
    )

    regime_clf = RegimeClassifier()
    xgb        = XGBPredictor()
    lstm       = LSTMPredictor()

    portfolio_value, available_cash = client.get_account_summary()

    # Compliance: check Alpaca account status before trading
    try:
        acct = client.get_account()
        if getattr(acct, "pattern_day_trader", False):
            logger.warning("Alpaca account is flagged as Pattern Day Trader — PDT limits apply.")
        pdt_equity = float(getattr(acct, "equity", 0) or 0)
        pdt_exempt = pdt_equity >= 25_000
        if pdt_exempt:
            logger.info(f"Account equity ${pdt_equity:,.2f} ≥ $25,000 — PDT day-trade limits waived.")
    except Exception as e:
        logger.warning(f"Account compliance check failed: {e}")
        pdt_exempt = False

    positions       = client.get_positions()
    _reconcile_positions(con, positions)
    open_order_syms = client.get_open_order_symbols()

    # First cycle of the day — daily_start was None before reset_daily sets it
    if daily_start is None:
        tg.alert_bot_started(mode, portfolio_value)

    risk.reset_daily(portfolio_value)
    _save_risk_state(con, risk)

    logger.info(
        f"Portfolio: ${portfolio_value:.2f} | Cash: ${available_cash:.2f} | "
        f"Open positions: {list(positions.keys())} | Pending orders: {open_order_syms}"
    )

    # Early warning: once per day when portfolio crosses 50% of daily loss limit
    if risk.check_daily_loss_warning(portfolio_value):
        pnl_warn = (portfolio_value - risk.daily_start_value) / risk.daily_start_value
        tg.alert_risk_warning(portfolio_value, pnl_warn)
        risk.daily_warning_sent = True
        _save_risk_state(con, risk)

    macro_score, macro_cap, macro_halt = _get_macro_from_db(con)
    logger.info(f"Macro: score={macro_score:.2f}, cap={macro_cap:.1f}x, halt={macro_halt}")
    if macro_halt:
        logger.warning("VIX emergency halt active — no new buys this cycle")
        tg.alert_vix_halt()  # fires every cycle — VIX crisis events warrant repeated alerts

    # Weekly loss circuit breaker alert — sent once per week when limit is first hit
    if not risk.check_weekly_loss(portfolio_value) and not risk.weekly_halt_alerted:
        wk_pnl = (portfolio_value - risk.weekly_start_value) / risk.weekly_start_value
        tg.alert_weekly_loss_limit(portfolio_value, wk_pnl)
        risk.weekly_halt_alerted = True
        _save_risk_state(con, risk)

    premarket_sentiment = _load_premarket_sentiment()
    if not premarket_sentiment:
        logger.warning(
            "Pre-market sentiment unavailable — sentiment defaults to neutral (0.0) this cycle. "
            "NewsAPI quota (100 req/day) is not consumed in-cycle."
        )

    def _fetch_symbol(symbol: str) -> tuple[str, pd.DataFrame]:
        try:
            bars = compute_features(client.get_bars(symbol, timeframe="5Min", limit=200))
            # Staleness guard: skip symbols whose last bar is >15 min old during market hours
            if not bars.empty:
                last_ts = bars.index[-1]
                now_utc = pd.Timestamp.now(tz="UTC")
                last_utc = last_ts.tz_localize("UTC") if last_ts.tzinfo is None else last_ts.tz_convert("UTC")
                age_mins = (now_utc - last_utc).total_seconds() / 60
                if age_mins > 15:
                    logger.warning(f"Stale bars for {symbol}: last bar is {age_mins:.0f}m old — skipping")
                    bars = pd.DataFrame()
        except Exception as e:
            logger.warning(f"Bar fetch failed for {symbol}: {e}")
            bars = pd.DataFrame()
        return symbol, bars

    with ThreadPoolExecutor(max_workers=min(len(active_symbols), 10)) as pool:
        fetched = list(pool.map(_fetch_symbol, active_symbols))

    bars_map = {sym: bars for sym, bars in fetched}

    if premarket_sentiment:
        finbert_scores = premarket_sentiment
        logger.info("Using pre-market FinBERT sentiment — skipping in-cycle BERT pass")
    else:
        finbert_scores = {sym: 0.0 for sym in active_symbols}

    def _wsb(symbol: str) -> tuple[str, dict]:
        try:
            return symbol, get_wsb_sentiment(symbol)
        except Exception:
            return symbol, {"mentions": 0, "sentiment": 0.0}

    with ThreadPoolExecutor(max_workers=6) as pool:
        wsb_map = dict(pool.map(_wsb, active_symbols))

    sentiments: dict[str, float] = {}
    for symbol in active_symbols:
        wsb   = wsb_map[symbol]
        score = finbert_scores.get(symbol, 0.0)
        if wsb["mentions"] > 0:
            wsb_weight = min(0.50, math.log1p(wsb["mentions"]) / 10)
            sentiments[symbol] = score * (1 - wsb_weight) + wsb["sentiment"] * wsb_weight
        else:
            sentiments[symbol] = score

    # Pre-compute SPY 5-bar return for relative strength gate
    spy_bars_5m = bars_map.get("SPY", pd.DataFrame())
    spy_5bar_return: float | None = None
    if not spy_bars_5m.empty and len(spy_bars_5m) > RS_LOOKBACK_BARS:
        v = spy_bars_5m["close"].pct_change(RS_LOOKBACK_BARS).iloc[-1]
        if not math.isnan(v):
            spy_5bar_return = float(v)

    try:
        spy_day_bars = client.get_bars("SPY", timeframe="1Day", limit=2)
        vs_spy_today = float(spy_day_bars["close"].pct_change().iloc[-1]) if len(spy_day_bars) > 1 else 0.0
    except Exception:
        vs_spy_today = 0.0

    # ── Per-symbol decision loop ──────────────────────────────────────────────
    for symbol in active_symbols:
        try:
            bars = bars_map.get(symbol, pd.DataFrame())
            if bars is None or bars.empty:
                continue

            latest        = bars.iloc[-1]
            current_price = float(latest["close"])
            current_atr   = float(latest.get("atr", 0) or 0)
            volume_ratio  = float(latest.get("volume_ratio", 1.0) or 1.0)
            regime_code   = regime_clf.predict(latest)
            regime_name   = regime_clf.regime_name(regime_code)

            xgb_prob          = xgb.predict_proba(latest)
            lstm_prob         = lstm.predict_proba(bars)
            sentiment         = sentiments.get(symbol, 0.0)
            action_str, ensemble_size = ensemble_signal(
                xgb_prob, lstm_prob, sentiment, regime_name, macro_score=macro_score
            )
            action = action_to_int(action_str)

            # ── Exit / management for held positions ──────────────────────────
            if symbol in positions:
                pos_state   = _load_position_state(con, symbol)
                entry_price = float(getattr(positions[symbol], "avg_entry_price", 0) or 0)
                pos_qty     = float(positions[symbol].qty)
                pnl_pct     = float(positions[symbol].unrealized_plpc or 0)

                # ⓪ Gap-down hard floor — bypass limit/ATR logic, market-sell immediately
                if pnl_pct < -0.10:
                    logger.warning(f"Gap-down floor: {symbol} pnl={pnl_pct:.1%} — immediate market sell")
                    sell_result = client.sell_market(symbol, pos_qty)
                    if sell_result:
                        client.wait_for_fill(sell_result["order_id"], timeout_secs=10)
                        tg.alert_stop_loss(symbol, pnl_pct, notional=pos_qty * current_price)
                        log_trade(con, symbol, "SELL_GAP_DOWN", pos_qty, current_price,
                                  pos_qty * current_price, regime_name, portfolio_value, pnl_pct)
                        _delete_position_state(con, symbol)
                        _maybe_record_day_trade(con, risk, symbol, True)
                    continue

                if pos_state:
                    new_hwm = max(pos_state["high_water_mark"], current_price)
                    if new_hwm > pos_state["high_water_mark"]:
                        _upsert_position_state(con, symbol, entry_price, new_hwm, current_atr)
                    hwm = new_hwm
                else:
                    _upsert_position_state(con, symbol, entry_price, current_price, current_atr)
                    hwm = current_price

                # ① Take-profit: max(6%, 3×ATR), capped at 8%
                if entry_price > 0 and current_atr > 0:
                    tp_pct = max(0.06, min(0.08, (3 * current_atr) / entry_price))
                    if pnl_pct >= tp_pct:
                        success = _signal_sell(
                            con, client, symbol, pos_qty, current_price,
                            regime_name, portfolio_value,
                            reason="take-profit", pnl_pct=pnl_pct
                        )
                        _maybe_record_day_trade(con, risk, symbol, success)
                        continue

                # ② ATR stop-loss
                if risk.check_stop_loss(symbol, current_price, entry_price,
                                        atr=current_atr, pnl_pct=pnl_pct):
                    success = _signal_sell(
                        con, client, symbol, pos_qty, current_price,
                        regime_name, portfolio_value,
                        is_from_stop=True, reason="stop-loss", pnl_pct=pnl_pct
                    )
                    _maybe_record_day_trade(con, risk, symbol, success)
                    continue

                # ③ Trailing stop (armed after 0.5% gain)
                if hwm > entry_price * 1.005 and risk.check_trailing_stop(
                        symbol, current_price, hwm, current_atr):
                    success = _signal_sell(
                        con, client, symbol, pos_qty, current_price,
                        regime_name, portfolio_value,
                        is_from_stop=True, reason="trailing-stop", pnl_pct=pnl_pct
                    )
                    _maybe_record_day_trade(con, risk, symbol, success)
                    continue

                # ④ Time-based forced exit — free capital from stale positions
                if _check_time_exit(pos_state, pnl_pct):
                    success = _signal_sell(
                        con, client, symbol, pos_qty, current_price,
                        regime_name, portfolio_value,
                        reason="time-exit", pnl_pct=pnl_pct
                    )
                    _maybe_record_day_trade(con, risk, symbol, success)
                    continue

                # ⑤ Ensemble sell signal
                if action == 2:
                    is_day_trade = _opened_today(con, symbol)
                    if is_day_trade and not pdt_exempt and not risk.check_pdt(is_day_trade=True):
                        logger.warning(f"PDT limit — skipping signal sell of {symbol}")
                    else:
                        success = _signal_sell(
                            con, client, symbol, pos_qty, current_price,
                            regime_name, portfolio_value,
                            reason="signal", pnl_pct=pnl_pct
                        )
                        if success and is_day_trade:
                            risk.record_day_trade()
                            _save_risk_state(con, risk)
                continue

            # ── Entry gates (applied in order of cheapness) ───────────────────
            if action != 1:
                continue

            # Gate 0 — VIX emergency halt: no new positions when VIX >= 40
            if macro_halt:
                continue

            # Gate 1 — Regime: only buy in trending or ranging markets
            if regime_name not in ENTRY_REGIMES:
                logger.info(f"BUY {symbol} skipped — regime={regime_name} (allowed: {ENTRY_REGIMES})")
                continue

            # Gate 2 — Volume: confirm institutional participation
            if volume_ratio < 1.0:
                logger.info(f"BUY {symbol} skipped — volume ratio {volume_ratio:.2f} < 1.0")
                continue

            # Gate 3 — 15-min RSI: multi-timeframe momentum must be bullish
            rsi_15m = float(latest.get("rsi_15m", 50) or 50)
            if rsi_15m < 50:
                logger.info(f"BUY {symbol} skipped — 15min RSI {rsi_15m:.1f} < 50")
                continue

            # Gate 4 — Relative strength: stock must be outperforming SPY over last N bars
            if spy_5bar_return is not None and symbol != "SPY":
                stock_5bar = bars["close"].pct_change(RS_LOOKBACK_BARS).iloc[-1]
                if not math.isnan(stock_5bar) and float(stock_5bar) < spy_5bar_return:
                    logger.info(
                        f"BUY {symbol} skipped — RS weak ({stock_5bar:.2%} vs SPY {spy_5bar_return:.2%})"
                    )
                    continue

            # Gate 5 — Open order: no duplicate limit submissions
            if symbol in open_order_syms:
                logger.info(f"BUY {symbol} skipped — open order already pending")
                continue

            # Gate 6 — Earnings proximity
            if _is_near_earnings(con, symbol):
                continue

            # Gate 7 — Correlation: avoid adding a position highly correlated with existing holdings
            if not _passes_correlation_gate(symbol, positions, bars_map):
                continue

            # Gate 8 — Cash and risk approval
            # ensemble_size: STRONG_BUY=0.20, BUY=0.12 — use as confidence multiplier on Kelly
            kelly_f      = _kelly_fraction(con, symbol)
            confidence   = ensemble_size / BUY_FRACTION  # 1.0 for BUY, 1.67 for STRONG_BUY
            pos_fraction = min(kelly_f * macro_cap * confidence, KELLY_FRACTION_MAX)
            notional     = portfolio_value * pos_fraction

            if notional > available_cash * 0.95:
                logger.warning(
                    f"BUY {symbol} skipped — need ${notional:.2f}, "
                    f"running cash ${available_cash:.2f}"
                )
                continue
            if not risk.approve_buy(symbol, notional, portfolio_value,
                                    portfolio_value, positions):
                continue

            result = client.buy(symbol, notional, limit_price=current_price)
            if result:
                filled = client.wait_for_fill(result["order_id"], timeout_secs=15)
                if filled:
                    tg.alert_buy(symbol, notional / current_price, current_price,
                                 regime_name, portfolio_value, vs_spy_today * 100,
                                 notional=notional)
                    log_trade(con, symbol, "BUY", notional / current_price,
                              current_price, notional, regime_name, portfolio_value, 0,
                              xgb_prob=xgb_prob, lstm_prob=lstm_prob,
                              sentiment_score=sentiments.get(symbol, 0.0),
                              macro_score=macro_score)
                    _upsert_position_state(con, symbol, current_price, current_price, current_atr)
                    available_cash -= notional
                else:
                    logger.warning(f"BUY {symbol} order did not fill — position state NOT recorded")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    logger.info("=== Trading cycle complete ===")
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    default="paper", choices=["paper", "live"])
    parser.add_argument("--summary", action="store_true",
                        help="Send end-of-day Telegram summary and exit")
    args = parser.parse_args()
    try:
        if args.summary:
            end_of_day_summary()
        else:
            run(mode=args.mode)
    except Exception:
        tb = traceback.format_exc()
        logger.error("Bot crashed:\n" + tb)
        print(f"::error title=Trading Bot Crash::{tb.splitlines()[-1]} — see step log", flush=True)
        sys.exit(1)
