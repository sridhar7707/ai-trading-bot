"""Main trading loop — runs every 5 minutes via GitHub Actions."""
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

from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from config import (
    SYMBOLS, TRADE_DB_PATH,
    MARKET_OPEN_BUFFER_MINS, MARKET_CLOSE_BUFFER_MINS,
    EARNINGS_WINDOW_DAYS,
)
from bot.execution.alpaca_client import AlpacaClient
from bot.strategy.features import compute_features
from bot.strategy.regime_classifier import RegimeClassifier
from bot.strategy.xgb_predictor import XGBPredictor
from bot.strategy.lstm_predictor import LSTMPredictor
from bot.strategy.sentiment import collect_headlines, batch_sentiment_scores
from bot.strategy.macro import get_macro_position_cap
from bot.strategy.reddit_sentiment import get_wsb_sentiment
from bot.strategy.ensemble import ensemble_signal, action_to_int
from bot.risk.risk_manager import RiskManager
import bot.monitor.telegram_bot as tg

logger.add("logs/trading.log", rotation="1 week", retention="4 weeks", level="INFO")

# ETFs have no earnings dates — skip the calendar check for these symbols
_ETF_SYMBOLS = {"VOO", "QQQ", "SPY", "VTI", "ARKK"}
_EARNINGS_DB_TTL = 12 * 3600  # seconds


def _is_market_hours() -> bool:
    import zoneinfo
    from datetime import timedelta
    et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
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
            regime TEXT, portfolio_value REAL, pnl_pct REAL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS position_state (
            symbol TEXT PRIMARY KEY, entry_price REAL,
            high_water_mark REAL, atr_at_entry REAL, opened_at TEXT
        )
    """)
    # Persists RiskManager state across the 5-minute subprocess restarts
    con.execute("""
        CREATE TABLE IF NOT EXISTS risk_state (
            key TEXT PRIMARY KEY, value TEXT, updated_at TEXT
        )
    """)
    # Persists earnings calendar lookups across subprocess restarts (12-hour TTL)
    con.execute("""
        CREATE TABLE IF NOT EXISTS earnings_cache (
            symbol TEXT PRIMARY KEY, near_earnings INTEGER, cached_at TEXT
        )
    """)
    con.commit()
    return con


def log_trade(con, symbol, action, shares, price, notional, regime, portfolio_value, pnl_pct):
    con.execute(
        "INSERT INTO trades VALUES (NULL,?,?,?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(),
         symbol, action, shares, price, notional, regime, portfolio_value, pnl_pct),
    )
    con.commit()


def _opened_today(con, symbol: str) -> bool:
    today = date.today().isoformat()
    row = con.execute(
        "SELECT 1 FROM trades WHERE symbol=? AND action='BUY' AND timestamp LIKE ? LIMIT 1",
        (symbol, today + "%"),
    ).fetchone()
    return row is not None


# ── Risk state persistence (survives per-cycle subprocess restarts) ───────────

def _load_risk_state(con) -> tuple[float | None, list[str]]:
    """Returns (daily_start_value, day_trade_date_strings) from DB for today."""
    today = date.today().isoformat()
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
    return daily_start, day_trade_dates


def _save_risk_state(con, risk: RiskManager):
    """Write RiskManager state to DB so the next cycle can restore it."""
    today  = date.today().isoformat()
    trades = json.dumps([d.isoformat() for d in risk.day_trade_log])
    start  = str(risk.daily_start_value) if risk.daily_start_value is not None else ""
    for key, val in [("daily_start_value", start),
                     ("daily_start_date",  today),
                     ("day_trade_dates",   trades)]:
        con.execute(
            "INSERT OR REPLACE INTO risk_state (key, value, updated_at) VALUES (?,?,?)",
            (key, val, today)
        )
    con.commit()


# ── Earnings calendar (DB-backed cache, persists across subprocess restarts) ──

def _is_near_earnings(con, symbol: str) -> bool:
    """
    True if an earnings announcement is within EARNINGS_WINDOW_DAYS.
    ETFs are always False. Result cached in DB for 12 hours.
    Fails open (False) if the yfinance fetch fails.
    """
    if symbol in _ETF_SYMBOLS:
        return False
    now = time.time()
    row = con.execute(
        "SELECT near_earnings, cached_at FROM earnings_cache WHERE symbol=?", (symbol,)
    ).fetchone()
    if row:
        try:
            cached_ts = datetime.fromisoformat(row[1]).timestamp()
            if now - cached_ts < _EARNINGS_DB_TTL:
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
        logger.debug(f"Earnings check failed for {symbol}: {e}")
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
        "SELECT entry_price, high_water_mark, atr_at_entry FROM position_state WHERE symbol=?",
        (symbol,),
    ).fetchone()
    return {"entry_price": row[0], "high_water_mark": row[1], "atr_at_entry": row[2]} if row else None


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


def _signal_sell(con, client, symbol, pos_qty, current_price,
                 regime_name, portfolio_value, is_from_stop=False,
                 reason="stop-loss", pnl_pct=0.0) -> bool:
    sell_result = client.sell(symbol, qty=pos_qty, limit_price=current_price)
    if sell_result:
        if is_from_stop:
            tg.alert_stop_loss(symbol, pnl_pct)
            log_trade(con, symbol, "SELL_STOP", pos_qty, current_price, 0,
                      regime_name, portfolio_value, pnl_pct)
        else:
            action_tag = "SELL" if reason == "signal" else f"SELL_{reason.upper().replace('-','_')}"
            tg.alert_sell(symbol, pos_qty, current_price, pnl_pct, reason=reason)
            log_trade(con, symbol, action_tag, pos_qty, current_price, 0,
                      regime_name, portfolio_value, pnl_pct)
        _delete_position_state(con, symbol)
        return True
    if is_from_stop:
        tg.alert_sell_failed(symbol, reason=reason)
    logger.error(f"SELL ({reason}) failed for {symbol} — will retry next cycle")
    return False


# ── End-of-day summary ────────────────────────────────────────────────────────

def end_of_day_summary():
    con    = init_db()
    client = AlpacaClient()
    today  = date.today().isoformat()
    trades_count = con.execute(
        "SELECT COUNT(*) FROM trades WHERE timestamp LIKE ?", (today + "%",)
    ).fetchone()[0]
    day_trade_count = con.execute(
        "SELECT COUNT(*) FROM trades WHERE action='BUY' AND timestamp LIKE ?",
        (today + "%",)
    ).fetchone()[0]
    daily_start, _ = _load_risk_state(con)
    portfolio_value, available_cash = client.get_account_summary()
    positions = client.get_positions()
    day_return = ((portfolio_value - daily_start) / daily_start) if daily_start else 0.0
    try:
        spy_bars  = client.get_bars("SPY", timeframe="1Day", limit=2)
        vs_spy    = float(spy_bars["close"].pct_change().iloc[-1]) if len(spy_bars) > 1 else 0.0
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


# ── Main trading cycle ────────────────────────────────────────────────────────

def run(mode: str = "paper"):
    logger.info(f"=== Trading cycle start | mode={mode} ===")
    if not _is_market_hours():
        return

    con = init_db()

    # Restore RiskManager state from DB (persists across 5-min subprocess restarts)
    daily_start, day_trade_dates = _load_risk_state(con)
    risk = RiskManager(daily_start_value=daily_start, day_trade_dates=day_trade_dates)

    client     = AlpacaClient()
    regime_clf = RegimeClassifier()
    xgb        = XGBPredictor()
    lstm       = LSTMPredictor()

    portfolio_value, available_cash = client.get_account_summary()
    positions       = client.get_positions()
    open_order_syms = client.get_open_order_symbols()

    risk.reset_daily(portfolio_value)
    _save_risk_state(con, risk)   # persist so daily_start survives next restart

    logger.info(
        f"Portfolio: ${portfolio_value:.2f} | Cash: ${available_cash:.2f} | "
        f"Open positions: {list(positions.keys())} | Pending orders: {open_order_syms}"
    )
    macro_cap = get_macro_position_cap()
    logger.info(f"Macro position cap: {macro_cap:.1f}x")

    # Phase 1 — parallel fetch: bars + headlines
    def _fetch_symbol(symbol: str) -> tuple[str, pd.DataFrame, list[str]]:
        try:
            bars = compute_features(client.get_bars(symbol, timeframe="5Min", limit=200))
        except Exception as e:
            logger.warning(f"Bar fetch failed for {symbol}: {e}")
            bars = pd.DataFrame()
        try:
            headlines = collect_headlines(symbol)
        except Exception as e:
            logger.warning(f"Headline fetch failed for {symbol}: {e}")
            headlines = []
        return symbol, bars, headlines

    with ThreadPoolExecutor(max_workers=min(len(SYMBOLS), 10)) as pool:
        fetched = list(pool.map(_fetch_symbol, SYMBOLS))

    bars_map         = {sym: bars for sym, bars, _   in fetched}
    symbol_headlines = {sym: hdl  for sym, _,    hdl in fetched}

    finbert_scores = batch_sentiment_scores(symbol_headlines)

    def _wsb(symbol: str) -> tuple[str, dict]:
        try:
            return symbol, get_wsb_sentiment(symbol)
        except Exception:
            return symbol, {"mentions": 0, "sentiment": 0.0}

    with ThreadPoolExecutor(max_workers=6) as pool:
        wsb_map = dict(pool.map(_wsb, SYMBOLS))

    sentiments: dict[str, float] = {}
    for symbol in SYMBOLS:
        wsb   = wsb_map[symbol]
        score = finbert_scores.get(symbol, 0.0)
        if wsb["mentions"] > 0:
            wsb_weight = min(0.50, math.log1p(wsb["mentions"]) / 10)
            sentiments[symbol] = score * (1 - wsb_weight) + wsb["sentiment"] * wsb_weight
        else:
            sentiments[symbol] = score

    try:
        spy_bars     = client.get_bars("SPY", timeframe="1Day", limit=2)
        vs_spy_today = float(spy_bars["close"].pct_change().iloc[-1]) if len(spy_bars) > 1 else 0.0
    except Exception:
        vs_spy_today = 0.0

    # ── Per-symbol decision loop ──────────────────────────────────────────────
    for symbol in SYMBOLS:
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

            xgb_prob   = xgb.predict_proba(latest)
            lstm_prob  = lstm.predict_proba(bars)
            sentiment  = sentiments.get(symbol, 0.0)
            action_str, pos_fraction = ensemble_signal(xgb_prob, lstm_prob, sentiment, regime_name)
            action = action_to_int(action_str)

            # ── Exit / management for held positions ──────────────────────────
            if symbol in positions:
                pos_state   = _load_position_state(con, symbol)
                entry_price = float(getattr(positions[symbol], "avg_entry_price", 0) or 0)
                pos_qty     = float(positions[symbol].qty)
                # Read P&L from the already-fetched positions dict — no extra API call
                pnl_pct     = float(positions[symbol].unrealized_plpc)

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
                        _signal_sell(con, client, symbol, pos_qty, current_price,
                                     regime_name, portfolio_value,
                                     reason="take-profit", pnl_pct=pnl_pct)
                        continue

                # ② ATR stop-loss
                if risk.check_stop_loss(symbol, current_price, entry_price,
                                        atr=current_atr, pnl_pct=pnl_pct):
                    _signal_sell(con, client, symbol, pos_qty, current_price,
                                 regime_name, portfolio_value,
                                 is_from_stop=True, reason="stop-loss", pnl_pct=pnl_pct)
                    continue

                # ③ Trailing stop (armed once meaningfully in profit)
                if hwm > entry_price * 1.005 and risk.check_trailing_stop(
                        symbol, current_price, hwm, current_atr):
                    _signal_sell(con, client, symbol, pos_qty, current_price,
                                 regime_name, portfolio_value,
                                 is_from_stop=True, reason="trailing-stop", pnl_pct=pnl_pct)
                    continue

                # ④ Ensemble sell — exits when model conviction deteriorates
                if action == 2:
                    is_day_trade = _opened_today(con, symbol)
                    if is_day_trade and not risk.check_pdt(is_day_trade=True):
                        logger.warning(f"PDT limit — skipping signal sell of {symbol}")
                    else:
                        success = _signal_sell(con, client, symbol, pos_qty, current_price,
                                               regime_name, portfolio_value,
                                               reason="signal", pnl_pct=pnl_pct)
                        if success and is_day_trade:
                            risk.record_day_trade()
                            _save_risk_state(con, risk)
                continue

            # ── Entry logic for symbols not currently held ────────────────────
            if action != 1:
                continue

            if volume_ratio < 1.0:
                logger.info(f"BUY {symbol} skipped — volume ratio {volume_ratio:.2f} < 1.0")
                continue
            if symbol in open_order_syms:
                logger.info(f"BUY {symbol} skipped — open order already pending")
                continue
            if _is_near_earnings(con, symbol):
                continue

            pos_fraction = pos_fraction * macro_cap
            notional     = portfolio_value * pos_fraction

            if notional > available_cash * 0.95:
                logger.warning(
                    f"BUY {symbol} skipped — need ${notional:.2f}, "
                    f"available cash ${available_cash:.2f}"
                )
                continue
            if not risk.approve_buy(symbol, notional, portfolio_value,
                                    portfolio_value, positions):
                continue

            result = client.buy(symbol, notional, limit_price=current_price)
            if result:
                tg.alert_buy(symbol, notional / current_price, current_price,
                             regime_name, portfolio_value, vs_spy_today * 100)
                log_trade(con, symbol, "BUY", notional / current_price,
                          current_price, notional, regime_name, portfolio_value, 0)
                _upsert_position_state(con, symbol, current_price, current_price, current_atr)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    logger.info("=== Trading cycle complete ===")


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
