"""Main trading loop — runs every 5 minutes via GitHub Actions."""
import argparse
import math
import os
import sqlite3
import sys
import time
import traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timezone
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

# ── Earnings calendar cache (12-hour TTL — refreshed once per trading day) ────
_EARNINGS_CACHE: dict[str, tuple[bool, float]] = {}
_EARNINGS_TTL   = 12 * 3600


def _is_near_earnings(symbol: str) -> bool:
    """
    Returns True if an earnings announcement falls within EARNINGS_WINDOW_DAYS.
    Blocks new entries to avoid binary event risk. Cached 12 hours.
    Fails open (returns False) if the calendar fetch fails.
    """
    now = time.time()
    if symbol in _EARNINGS_CACHE:
        result, ts = _EARNINGS_CACHE[symbol]
        if now - ts < _EARNINGS_TTL:
            return result
    try:
        import yfinance as yf
        from datetime import date
        cal = yf.Ticker(symbol).calendar
        if cal is None:
            _EARNINGS_CACHE[symbol] = (False, now)
            return False
        # yfinance 0.2.x returns a dict; older versions return a DataFrame
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
        else:
            dates = cal.loc["Earnings Date"].tolist() if "Earnings Date" in cal.index else []
        if not dates:
            _EARNINGS_CACHE[symbol] = (False, now)
            return False
        nearest = pd.to_datetime(dates[0]).date()
        near = abs((nearest - date.today()).days) <= EARNINGS_WINDOW_DAYS
        _EARNINGS_CACHE[symbol] = (near, now)
        if near:
            logger.info(f"Earnings guard: {symbol} earnings {nearest} — blocking entry")
        return near
    except Exception as e:
        logger.debug(f"Earnings check failed for {symbol}: {e}")
        _EARNINGS_CACHE[symbol] = (False, now)
        return False


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
            timestamp TEXT,
            symbol TEXT,
            action TEXT,
            shares REAL,
            price REAL,
            notional REAL,
            regime TEXT,
            portfolio_value REAL,
            pnl_pct REAL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS position_state (
            symbol           TEXT PRIMARY KEY,
            entry_price      REAL,
            high_water_mark  REAL,
            atr_at_entry     REAL,
            opened_at        TEXT
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
    today = datetime.now(timezone.utc).date().isoformat()
    row = con.execute(
        "SELECT 1 FROM trades WHERE symbol=? AND action='BUY' AND timestamp LIKE ? LIMIT 1",
        (symbol, today + "%"),
    ).fetchone()
    return row is not None


# ── Position state helpers ────────────────────────────────────────────────────

def _load_position_state(con, symbol: str) -> dict | None:
    row = con.execute(
        "SELECT entry_price, high_water_mark, atr_at_entry FROM position_state WHERE symbol=?",
        (symbol,),
    ).fetchone()
    if row:
        return {"entry_price": row[0], "high_water_mark": row[1], "atr_at_entry": row[2]}
    return None


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


def _signal_sell(con, client, positions, symbol, pos_qty, current_price,
                 regime_name, portfolio_value, risk, is_from_stop=False,
                 stop_reason="stop-loss", pnl_pct=0.0) -> bool:
    """Execute a sell order and record it. Returns True on success."""
    sell_result = client.sell(symbol, qty=pos_qty, limit_price=current_price)
    if sell_result:
        if is_from_stop:
            tg.alert_stop_loss(symbol, pnl_pct)
            log_trade(con, symbol, "SELL_STOP", pos_qty, current_price, 0,
                      regime_name, portfolio_value, pnl_pct)
        else:
            tg.alert_sell(symbol, pos_qty, current_price, pnl_pct, reason=stop_reason)
            log_trade(con, symbol, "SELL" if stop_reason == "signal" else f"SELL_{stop_reason.upper()}",
                      pos_qty, current_price, 0, regime_name, portfolio_value, pnl_pct)
        _delete_position_state(con, symbol)
        return True
    else:
        if is_from_stop:
            tg.alert_sell_failed(symbol, reason=stop_reason)
        logger.error(f"SELL ({stop_reason}) failed for {symbol} — will retry next cycle")
        return False


# ── Main trading cycle ────────────────────────────────────────────────────────

def run(mode: str = "paper"):
    logger.info(f"=== Trading cycle start | mode={mode} ===")
    if not _is_market_hours():
        return

    con    = init_db()
    client = AlpacaClient()
    regime_clf = RegimeClassifier()
    xgb    = XGBPredictor()
    lstm   = LSTMPredictor()
    risk   = RiskManager()

    portfolio_value, available_cash = client.get_account_summary()
    positions        = client.get_positions()        # {symbol: position_obj}
    open_order_syms  = client.get_open_order_symbols()
    risk.reset_daily(portfolio_value)

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

    # Phase 2 — single batch FinBERT pass
    finbert_scores = batch_sentiment_scores(symbol_headlines)

    # Phase 3 — parallel WSB sentiment
    def _wsb(symbol: str) -> tuple[str, dict]:
        try:
            return symbol, get_wsb_sentiment(symbol)
        except Exception:
            return symbol, {"mentions": 0, "sentiment": 0.0}

    with ThreadPoolExecutor(max_workers=6) as pool:
        wsb_map = dict(pool.map(_wsb, SYMBOLS))

    # Confidence-weighted blend: WSB weight scales with log(mentions) up to 50%
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

            # Compute ensemble score once — used in both held-position and entry paths
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
                pnl_pct     = client.get_position_pnl_pct(symbol)

                # Update high-water-mark
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
                        _signal_sell(con, client, positions, symbol, pos_qty, current_price,
                                     regime_name, portfolio_value, risk,
                                     stop_reason="take-profit", pnl_pct=pnl_pct)
                        continue

                # ② ATR stop-loss
                if risk.check_stop_loss(symbol, current_price, entry_price,
                                        atr=current_atr, pnl_pct=pnl_pct):
                    _signal_sell(con, client, positions, symbol, pos_qty, current_price,
                                 regime_name, portfolio_value, risk,
                                 is_from_stop=True, stop_reason="stop-loss", pnl_pct=pnl_pct)
                    continue

                # ③ Trailing stop (only once meaningfully in profit)
                if hwm > entry_price * 1.005 and risk.check_trailing_stop(
                        symbol, current_price, hwm, current_atr):
                    _signal_sell(con, client, positions, symbol, pos_qty, current_price,
                                 regime_name, portfolio_value, risk,
                                 is_from_stop=True, stop_reason="trailing-stop", pnl_pct=pnl_pct)
                    continue

                # ④ Ensemble signal exit — fires when model conviction shifts to SELL
                #    This closes positions on signal deterioration, not just on stops.
                if action == 2:
                    is_day_trade = _opened_today(con, symbol)
                    if is_day_trade and not risk.check_pdt(is_day_trade=True):
                        logger.warning(f"PDT limit — skipping signal sell of {symbol}")
                    else:
                        success = _signal_sell(con, client, positions, symbol, pos_qty,
                                               current_price, regime_name, portfolio_value, risk,
                                               stop_reason="signal", pnl_pct=pnl_pct)
                        if success and is_day_trade:
                            risk.record_day_trade()
                continue  # don't fall through to entry logic

            # ── Entry logic for symbols not currently held ────────────────────

            if action != 1:
                continue  # HOLD or SELL — no entry

            # Gate 1 — volume confirmation: skip thin/low-conviction bars
            if volume_ratio < 1.0:
                logger.info(f"BUY {symbol} skipped — volume ratio {volume_ratio:.2f} < 1.0")
                continue

            # Gate 2 — open order guard: don't stack limit orders
            if symbol in open_order_syms:
                logger.info(f"BUY {symbol} skipped — open order already pending")
                continue

            # Gate 3 — earnings calendar: avoid binary event risk
            if _is_near_earnings(symbol):
                continue

            pos_fraction = pos_fraction * macro_cap
            notional     = portfolio_value * pos_fraction

            # Gate 4 — cash availability
            if notional > available_cash * 0.95:
                logger.warning(
                    f"BUY {symbol} skipped — need ${notional:.2f}, "
                    f"available cash ${available_cash:.2f}"
                )
                continue

            # Gate 5 — risk manager (sector, daily loss, position count, size)
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
    parser.add_argument("--mode", default="paper", choices=["paper", "live"])
    args = parser.parse_args()
    try:
        run(mode=args.mode)
    except Exception:
        tb = traceback.format_exc()
        logger.error("Trading cycle crashed:\n" + tb)
        print(f"::error title=Trading Bot Crash::{tb.splitlines()[-1]} — see step log", flush=True)
        sys.exit(1)
