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

from config import SYMBOLS, TRADE_DB_PATH
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


def _opened_today(con, symbol: str) -> bool:
    """Return True if there is a BUY for this symbol recorded today (UTC).
    log_trade also uses timezone.utc, so both sides are UTC-consistent."""
    today = datetime.now(timezone.utc).date().isoformat()
    row = con.execute(
        "SELECT 1 FROM trades WHERE symbol=? AND action='BUY' AND timestamp LIKE ? LIMIT 1",
        (symbol, today + "%"),
    ).fetchone()
    return row is not None


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
    con.commit()
    return con


def log_trade(con, symbol, action, shares, price, notional, regime, portfolio_value, pnl_pct):
    con.execute(
        "INSERT INTO trades VALUES (NULL,?,?,?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), symbol, action, shares, price, notional, regime, portfolio_value, pnl_pct),
    )
    con.commit()


def _is_market_hours() -> bool:
    """Return True only during NYSE regular session (9:30am–4:00pm ET, Mon–Fri)."""
    import zoneinfo
    et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return False
    open_time = et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = et.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_time <= et < close_time


def run(mode: str = "paper"):
    logger.info(f"=== Trading cycle start | mode={mode} ===")
    if not _is_market_hours():
        logger.info("Outside market hours — skipping cycle.")
        return
    con = init_db()
    client = AlpacaClient()
    regime_clf = RegimeClassifier()
    xgb = XGBPredictor()
    lstm = LSTMPredictor()
    risk = RiskManager()

    portfolio_value = client.get_portfolio_value()
    positions = client.get_positions()
    risk.reset_daily(portfolio_value)

    macro_cap = get_macro_position_cap()
    logger.info(f"Macro position cap: {macro_cap:.1f}x")

    # Phase 1 — parallel: fetch bars + headlines for all symbols simultaneously
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

    bars_map = {sym: bars for sym, bars, _ in fetched}
    symbol_headlines = {sym: headlines for sym, _, headlines in fetched}

    # Phase 2 — single batch FinBERT pass across all symbols
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
        wsb = wsb_map[symbol]
        score = finbert_scores.get(symbol, 0.0)
        sentiments[symbol] = (score + wsb["sentiment"]) / 2 if wsb["mentions"] > 0 else score

    # Pre-fetch SPY daily bar once for vs-SPY comparison on buys
    try:
        spy_bars = client.get_bars("SPY", timeframe="1Day", limit=2)
        vs_spy_today = float(spy_bars["close"].pct_change().iloc[-1]) if len(spy_bars) > 1 else 0.0
    except Exception:
        vs_spy_today = 0.0

    for symbol in SYMBOLS:
        try:
            bars = bars_map.get(symbol, pd.DataFrame())
            if bars is None or bars.empty:
                continue

            latest = bars.iloc[-1]
            regime_code = regime_clf.predict(latest)
            regime_name = regime_clf.regime_name(regime_code)

            # Stop-loss check on existing positions
            if symbol in positions:
                pnl_pct = client.get_position_pnl_pct(symbol)
                if risk.check_stop_loss(symbol, pnl_pct):
                    sell_result = client.sell(symbol)
                    if sell_result:
                        tg.alert_stop_loss(symbol, pnl_pct)
                        log_trade(con, symbol, "SELL_STOP", 0, latest["close"], 0, regime_name, portfolio_value, pnl_pct)
                    else:
                        logger.error(f"SELL_STOP failed for {symbol} — will retry next cycle")
                        tg.alert_sell_failed(symbol, reason="stop-loss")
                    continue

            # Ensemble signal
            xgb_prob = xgb.predict_proba(latest)
            lstm_prob = lstm.predict_proba(bars)
            sentiment = sentiments.get(symbol, 0.0)

            action_str, pos_fraction = ensemble_signal(xgb_prob, lstm_prob, sentiment, regime_name)
            action = action_to_int(action_str)

            # Apply macro position cap
            pos_fraction = pos_fraction * macro_cap

            price = float(latest["close"])
            notional = portfolio_value * pos_fraction

            if action == 1:  # Buy
                if risk.approve_buy(symbol, notional, portfolio_value, portfolio_value, len(positions)):
                    result = client.buy(symbol, notional)
                    if result:
                        tg.alert_buy(symbol, notional / price, price, regime_name, portfolio_value, vs_spy_today * 100)
                        log_trade(con, symbol, "BUY", notional / price, price, notional, regime_name, portfolio_value, 0)
            elif action == 2:  # Sell
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
                            tg.alert_sell(symbol, float(positions[symbol].qty), price, pnl_pct)
                            log_trade(con, symbol, "SELL", float(positions[symbol].qty), price, 0, regime_name, portfolio_value, pnl_pct)
                        else:
                            logger.error(f"SELL order rejected for {symbol} — trade NOT logged")

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
        # Emit a GitHub Actions annotation so it appears in the Annotations panel
        print(f"::error title=Trading Bot Crash::{tb.splitlines()[-1]} — see step log for full traceback", flush=True)
        sys.exit(1)
