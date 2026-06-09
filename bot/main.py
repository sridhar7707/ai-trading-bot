"""Main trading loop — runs every 5 minutes via GitHub Actions."""
import argparse
import os
import sqlite3
import sys
import traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timezone
from loguru import logger

from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from config import (
    SYMBOLS, TRADE_DB_PATH,
    MARKET_OPEN_BUFFER_MINS, MARKET_CLOSE_BUFFER_MINS,
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


def _is_market_hours() -> bool:
    """
    True only during NYSE regular session with timing buffers applied.
    Skips the first MARKET_OPEN_BUFFER_MINS (volatile open) and the last
    MARKET_CLOSE_BUFFER_MINS (illiquid close) to improve fill quality.
    """
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
    # Tracks per-position state across cycles for trailing stops
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


# ── Position state helpers (for trailing stop across cycles) ──────────────────

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

    portfolio_value = client.get_portfolio_value()
    positions       = client.get_positions()          # {symbol: position_obj}
    risk.reset_daily(portfolio_value)

    logger.info(f"Portfolio: ${portfolio_value:.2f} | Open positions: {list(positions.keys())}")
    macro_cap = get_macro_position_cap()
    logger.info(f"Macro position cap: {macro_cap:.1f}x")

    # Phase 1 — parallel fetch: bars + headlines for all symbols
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

    bars_map       = {sym: bars for sym, bars, _    in fetched}
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

    sentiments: dict[str, float] = {}
    for symbol in SYMBOLS:
        wsb   = wsb_map[symbol]
        score = finbert_scores.get(symbol, 0.0)
        sentiments[symbol] = (score + wsb["sentiment"]) / 2 if wsb["mentions"] > 0 else score

    # SPY benchmark for buy alert context
    try:
        spy_bars    = client.get_bars("SPY", timeframe="1Day", limit=2)
        vs_spy_today = float(spy_bars["close"].pct_change().iloc[-1]) if len(spy_bars) > 1 else 0.0
    except Exception:
        vs_spy_today = 0.0

    # ── Per-symbol decision loop ──────────────────────────────────────────────
    for symbol in SYMBOLS:
        try:
            bars = bars_map.get(symbol, pd.DataFrame())
            if bars is None or bars.empty:
                continue

            latest      = bars.iloc[-1]
            current_price = float(latest["close"])
            current_atr   = float(latest.get("atr", 0) or 0)
            regime_code   = regime_clf.predict(latest)
            regime_name   = regime_clf.regime_name(regime_code)

            # ── Exit checks for held positions ────────────────────────────────
            if symbol in positions:
                pos_state   = _load_position_state(con, symbol)
                entry_price = float(getattr(positions[symbol], "avg_entry_price", 0) or 0)
                pnl_pct     = client.get_position_pnl_pct(symbol)

                # Update high-water-mark in DB
                if pos_state:
                    new_hwm = max(pos_state["high_water_mark"], current_price)
                    if new_hwm > pos_state["high_water_mark"]:
                        _upsert_position_state(con, symbol, entry_price, new_hwm, current_atr)
                    hwm = new_hwm
                else:
                    # First time seeing this position — seed the state
                    _upsert_position_state(con, symbol, entry_price, current_price, current_atr)
                    hwm = current_price

                # ATR stop-loss check
                stop_triggered = risk.check_stop_loss(
                    symbol, current_price, entry_price,
                    atr=current_atr, pnl_pct=pnl_pct,
                )
                # Trailing stop check (only if we're above entry)
                trail_triggered = risk.check_trailing_stop(
                    symbol, current_price, hwm, current_atr
                ) if hwm > entry_price * 1.005 else False

                if stop_triggered or trail_triggered:
                    reason = "stop-loss" if stop_triggered else "trailing-stop"
                    sell_result = client.sell(symbol)
                    if sell_result:
                        tg.alert_stop_loss(symbol, pnl_pct)
                        log_trade(con, symbol, "SELL_STOP", 0, current_price, 0,
                                  regime_name, portfolio_value, pnl_pct)
                        _delete_position_state(con, symbol)
                    else:
                        logger.error(f"SELL_STOP ({reason}) failed for {symbol} — will retry")
                        tg.alert_sell_failed(symbol, reason=reason)
                continue  # already holding — wait for sell signal or stop

            # ── Entry decision for symbols not held ───────────────────────────
            xgb_prob  = xgb.predict_proba(latest)
            lstm_prob = lstm.predict_proba(bars)
            sentiment = sentiments.get(symbol, 0.0)

            action_str, pos_fraction = ensemble_signal(xgb_prob, lstm_prob, sentiment, regime_name)
            action = action_to_int(action_str)

            # Apply macro cap and ATR-based position sizing
            pos_fraction = pos_fraction * macro_cap

            # Portfolio-relative sizing: use actual equity, not fixed DCA schedule
            # pos_fraction (0.12–0.20) × portfolio_value gives correct scale regardless
            # of how much the account has grown or shrunk.
            notional = portfolio_value * pos_fraction

            if action == 1:  # Buy
                if risk.approve_buy(symbol, notional, portfolio_value,
                                    portfolio_value, positions):
                    result = client.buy(symbol, notional)
                    if result:
                        tg.alert_buy(symbol, notional / current_price, current_price,
                                     regime_name, portfolio_value, vs_spy_today * 100)
                        log_trade(con, symbol, "BUY", notional / current_price,
                                  current_price, notional, regime_name, portfolio_value, 0)
                        # Seed trailing-stop state for this new position
                        _upsert_position_state(con, symbol, current_price,
                                               current_price, current_atr)

            elif action == 2:  # Sell signal on a held position (checked above; handle missed case)
                if symbol in positions:
                    is_day_trade = _opened_today(con, symbol)
                    if is_day_trade and not risk.check_pdt(is_day_trade=True):
                        logger.warning(f"PDT limit reached — skipping sell of {symbol}")
                    else:
                        pnl_pct = client.get_position_pnl_pct(symbol)
                        sell_result = client.sell(symbol)
                        if sell_result:
                            if is_day_trade:
                                risk.record_day_trade()
                            tg.alert_sell(symbol, float(positions[symbol].qty),
                                          current_price, pnl_pct)
                            log_trade(con, symbol, "SELL",
                                      float(positions[symbol].qty), current_price,
                                      0, regime_name, portfolio_value, pnl_pct)
                            _delete_position_state(con, symbol)
                        else:
                            logger.error(f"SELL order rejected for {symbol} — NOT logged")

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
