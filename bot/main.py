"""Main trading loop — runs every 5 minutes via GitHub Actions."""
from __future__ import annotations

import argparse
import math
import os
import sqlite3
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date, datetime, timedelta, timezone
import pandas as pd
import yfinance as yf
from loguru import logger
from bot.core.error_logger import log_exception

from config import (
    SYMBOLS, TRADE_DB_PATH,
    MARKET_OPEN_BUFFER_MINS, MARKET_CLOSE_BUFFER_MINS,
    EARNINGS_WINDOW_DAYS,
    MAX_HOLD_DAYS, KELLY_LOOKBACK_TRADES, KELLY_FRACTION_MAX,
    CORRELATION_THRESHOLD, RS_LOOKBACK_BARS, ENTRY_REGIMES, MIN_VOLUME_RATIO,
    PDT_MAX_DAY_TRADES, PDT_WINDOW_DAYS, PAPER_SIM_CAPITAL,
    MAX_RISK_PER_TRADE_PCT,
    ATR_STOP_MULTIPLIER, ATR_MIN_STOP_PCT, ATR_MAX_STOP_PCT, STOP_LOSS_PCT,
    MIN_RR_RATIO, MIN_TP_PCT, RANGING_SIZE_FACTOR,
    MAX_SECTOR_EXPOSURE_PCT, MAX_POSITION_DRIFT_PCT, MIN_CASH_RESERVE_PCT,
    MAX_POSITION_PCT, SECTOR_MAP,
)
from bot.execution.alpaca_client import AlpacaClient
from bot.strategy.features import compute_features, FEATURE_COLS
from bot.strategy.regime_classifier import RegimeClassifier
from bot.strategy.xgb_predictor import XGBPredictor
from bot.strategy.lstm_predictor import LSTMPredictor
from bot.strategy.sentiment import batch_sentiment_scores
from bot.strategy.macro import _get_cached as _get_macro_cached
from bot.strategy.reddit_sentiment import get_wsb_sentiment
from bot.strategy.ensemble import ensemble_signal, action_to_int, BUY_FRACTION, WEIGHTS
from bot.risk.risk_manager import RiskManager, _business_days_between
import bot.monitor.telegram_bot as tg

# Sub-module imports (helpers extracted to keep this file under 500 lines)
from bot._main_db import (
    _anchor_daily_start, _enable_wal_mode,
    _get_macro_from_db, _load_risk_state, _log_signal, _record_snapshot,
    _save_risk_state, _week_key, log_trade,
    init_db as _init_db_core,
)


def init_db() -> sqlite3.Connection:
    """Wrapper so monkeypatching bot.main.TRADE_DB_PATH in tests still works."""
    return _init_db_core(TRADE_DB_PATH)


def _apply_sim_capital(portfolio_value: float, available_cash: float) -> tuple[float, float, bool]:
    """Cap equity to PAPER_SIM_CAPITAL for small-account dry-run mechanics."""
    if PAPER_SIM_CAPITAL and PAPER_SIM_CAPITAL > 0:
        return (min(portfolio_value, PAPER_SIM_CAPITAL),
                min(available_cash, PAPER_SIM_CAPITAL),
                True)
    return portfolio_value, available_cash, False
from bot._main_positions import (
    _check_time_exit, _delete_position_state, _is_wash_sale_risk,
    _kelly_fraction, _load_position_state, _maybe_record_day_trade,
    _opened_today, _passes_correlation_gate, _reconcile_positions,
    _signal_sell, _trim_position, _upsert_position_state,
)
from bot._main_market import (
    _import_screener_picks, _is_market_hours, _is_near_earnings,
    _load_premarket_sentiment, _load_today_universe, _log_buy_skip,
    _prefetch_earnings_parallel, _wsb,
)
from bot._main_cycle import _fetch_symbol, _handle_exits, _handle_entry
from bot._main_runner import (
    _do_clean_db, _do_reset_daily_start, end_of_day_summary, run_loop,
)

os.makedirs("logs", exist_ok=True)
if not os.getenv("_BOT_LOG_HANDLER_ADDED"):
    logger.add("logs/trading.log", rotation="1 week", retention="4 weeks", level="INFO")
    os.environ["_BOT_LOG_HANDLER_ADDED"] = "1"

_HALT_FILE        = "data/HALT_TRADING"
_last_hf_sync: float = 0.0
_HF_SYNC_INTERVAL: float = 900
_stop_fired_today: set[str] = set()
_stop_fired_date: str = ""


def run(
    mode: str = "paper",
    _regime_clf: RegimeClassifier | None = None,
    _xgb: XGBPredictor | None = None,
    _lstm: LSTMPredictor | None = None,
) -> None:
    logger.info(f"=== Trading cycle start | mode={mode} ===")

    # Emergency override: create data/HALT_TRADING file to pause without canceling the workflow.
    if os.path.exists(_HALT_FILE):
        logger.warning("HALT_TRADING file detected — cycle skipped. Remove file to resume.")
        tg._send("⛔ <b>EMERGENCY HALT ACTIVE</b> — bot paused. Delete data/HALT_TRADING to resume.")
        return

    global _last_hf_sync, _stop_fired_today, _stop_fired_date
    today_str = date.today().isoformat()
    if _stop_fired_date != today_str:
        _stop_fired_today = set()
        _stop_fired_date = today_str
    client = AlpacaClient()
    if not _is_market_hours(client.api):
        logger.info("Market is closed — cycle skipped (no trades, no DB write). "
                    "Dashboard will keep showing the last synced values.")
        return

    con = init_db()

    active_symbols, _universe_payload = _load_today_universe()
    _import_screener_picks(con, _universe_payload)

    daily_start, day_trade_dates, weekly_start, daily_warning_sent, weekly_halt_alerted, portfolio_high = _load_risk_state(con)

    # Anchor daily_start to the account's value at yesterday's close (not current
    # live price), so Day P&L means "today's gain" rather than gain-since-inception.
    if daily_start is None:
        daily_start, _src = _anchor_daily_start(con)
        if daily_start is not None:
            logger.info(f"Daily start anchored to {_src}: ${daily_start:.2f}")

    risk = RiskManager(
        daily_start_value=daily_start,
        day_trade_dates=day_trade_dates,
        weekly_start_value=weekly_start,
        daily_warning_sent=daily_warning_sent,
        weekly_halt_alerted=weekly_halt_alerted,
        portfolio_high=portfolio_high,
    )

    regime_clf = _regime_clf if _regime_clf is not None else RegimeClassifier()
    xgb        = _xgb        if _xgb        is not None else XGBPredictor()
    lstm       = _lstm       if _lstm       is not None else LSTMPredictor()

    real_portfolio_value, real_available_cash = client.get_account_summary()
    if real_portfolio_value <= 0:
        logger.error(
            f"Alpaca returned portfolio_value=${real_portfolio_value:.2f} — likely an auth/connection "
            "failure (check ALPACA_KEY/ALPACA_SECRET). Dashboard would show $0.00. Aborting cycle."
        )
        tg._send("🚨 Alpaca account value is $0.00 — check API credentials. Bot cycle aborted.")
        con.close()
        return
    logger.info(f"Alpaca connection OK — account value ${real_portfolio_value:,.2f}")
    # Paper sim-capital: size/risk-check as if the account were small (dry-run).
    # We keep the real values separately so the dashboard always shows the true account equity.
    portfolio_value, available_cash, _sim_capital = _apply_sim_capital(real_portfolio_value, real_available_cash)
    if _sim_capital:
        logger.warning(
            f"PAPER_SIM_CAPITAL active — sizing & risk as if account = "
            f"${portfolio_value:,.2f} (real account ${real_portfolio_value:,.2f})"
        )
    risk.update_portfolio_high(portfolio_value)

    # Brokerage compliance: validate account standing before placing any orders
    try:
        acct = client.get_account()
        # ① Account status gate — Alpaca can suspend accounts for policy violations
        acct_status     = getattr(acct, "status",          "ACTIVE")
        trading_blocked = getattr(acct, "trading_blocked", False)
        account_blocked = getattr(acct, "account_blocked", False)
        if acct_status != "ACTIVE" or trading_blocked or account_blocked:
            status_msg = (f"status={acct_status}, trading_blocked={trading_blocked}, "
                          f"account_blocked={account_blocked}")
            logger.error(f"Account not tradeable ({status_msg}) — aborting cycle")
            tg._send(f"🚨 Account not tradeable ({status_msg}) — bot halted. Check Alpaca dashboard.")
            con.close()
            return
        # ② PDT flag and equity check
        if getattr(acct, "pattern_day_trader", False):
            logger.warning("Alpaca account is flagged as Pattern Day Trader — PDT limits apply.")
        pdt_equity = float(getattr(acct, "equity", 0) or 0)
        # Under sim-capital, use the simulated equity so the PDT limit (under $25k)
        # actually applies — that's a key small-account behaviour to dry-run.
        if _sim_capital:
            pdt_equity = min(pdt_equity, PAPER_SIM_CAPITAL)
        pdt_exempt = pdt_equity >= 25_000
        if pdt_exempt:
            logger.info(f"Account equity ${pdt_equity:,.2f} ≥ $25,000 — PDT day-trade limits waived.")
        logger.info(f"Account standing verified — status=ACTIVE, equity=${pdt_equity:,.2f}")
    except Exception as e:
        logger.warning(f"Account compliance check failed: {e}")
        pdt_exempt = False

    positions       = client.get_positions()
    _reconcile_positions(con, positions, portfolio_value=portfolio_value, client=client)
    buy_order_syms, sell_order_syms = client.get_open_order_symbols()

    # Restore intraday halt — persists across 5-min cycles so a mid-day breach
    # can't be traded through when the risk object is reconstructed each cycle.
    _halt_row = con.execute(
        "SELECT value FROM risk_state WHERE key='trading_halted_date'"
    ).fetchone()
    if _halt_row and _halt_row[0] == date.today().isoformat():
        risk.halted = True
        logger.warning("Halt state restored from DB — daily loss limit was breached earlier today")

    # First cycle of the day — daily_start was None before reset_daily sets it
    if daily_start is None:
        tg.alert_bot_started(mode, real_portfolio_value)

    # Always reset daily using the REAL account value so the dashboard's Day P&L
    # baseline matches what Alpaca actually shows — not the sim-capped value.
    risk.reset_daily(real_portfolio_value)
    _save_risk_state(con, risk)

    logger.info(
        f"Portfolio: ${real_portfolio_value:.2f} (sim: ${portfolio_value:.2f}) | "
        f"Cash: ${real_available_cash:.2f} | "
        f"Open positions: {list(positions.keys())} | "
        f"Pending buys: {buy_order_syms} | Pending sells: {sell_order_syms}"
    )
    # Heartbeat snapshot — always stores the REAL account value so the dashboard
    # portfolio total is correct regardless of PAPER_SIM_CAPITAL.
    _record_snapshot(con, real_portfolio_value, real_available_cash, len(positions))
    if sell_order_syms:
        logger.warning(
            f"Open sell orders detected for {len(sell_order_syms)} symbol(s): {sell_order_syms} "
            "— exit management paused for these symbols this cycle"
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

    # Batch-fetch all daily bars via yfinance before the thread pool.
    # Per-symbol yf.download inside threads is not thread-safe and causes
    # data corruption (wrong shapes). One batch call also reduces Yahoo Finance
    # requests from N/cycle to 1/cycle (~78/day vs ~1560/day).
    _yf_batch: dict[str, pd.DataFrame] = {}
    try:
        _batch_syms = list(active_symbols)
        if "SPY" not in _batch_syms:
            _batch_syms.append("SPY")
        _raw_batch = yf.download(_batch_syms, period="2y", interval="1d",
                                 progress=False, auto_adjust=True, group_by="ticker")
        for _sym in _batch_syms:
            try:
                _sym_df = _raw_batch[_sym].copy()
                _sym_df.columns = [c.lower() for c in _sym_df.columns]
                _sym_df = _sym_df[["open", "high", "low", "close", "volume"]].dropna()
                if not _sym_df.empty:
                    _yf_batch[_sym] = _sym_df
            except Exception:
                pass
        loaded, total = len(_yf_batch), len(_batch_syms)
        logger.info(f"yfinance batch: {loaded}/{total} symbols loaded")
        if "SPY" not in _yf_batch:
            logger.warning("SPY missing from yfinance batch — relative strength gate disabled this cycle")
            tg.send("⚠️ <b>SPY data missing</b> — relative strength gate disabled. Buys not filtered by SPY comparison.")
        if loaded < total * 0.5:
            tg.send(
                f"⚠️ <b>yfinance data degraded</b> — only {loaded}/{total} symbols loaded.\n"
                "Yahoo Finance may have changed their API format. "
                "XGB/LSTM signals falling back to 5-min bars (out-of-distribution).\n"
                "Check: <code>pip install --upgrade yfinance</code>"
            )
    except Exception as _e:
        logger.warning(f"yfinance batch prefetch failed — daily bars unavailable: {_e}")
        tg.send(
            f"⚠️ <b>yfinance batch fetch failed</b> — {_e}\n"
            "Daily bars unavailable this cycle. Check if Yahoo Finance format changed."
        )

    # _fetch_symbol now lives in bot._main_cycle; pass client and _yf_batch explicitly
    with ThreadPoolExecutor(max_workers=len(active_symbols)) as pool:
        futures = [pool.submit(_fetch_symbol, sym, client, _yf_batch) for sym in active_symbols]
        fetched = [f.result() for f in futures]

    bars_map = {sym: (b5, bd) for sym, b5, bd in fetched}
    _n_5m = sum(1 for _, b5, _ in fetched if not b5.empty)
    if _n_5m < len(active_symbols) * 0.5:
        tg.send(
            f"⚠️ <b>Alpaca feed degraded</b> — only {_n_5m}/{len(active_symbols)} symbols "
            "have live 5-min bars. Most symbols will be skipped this cycle. "
            "Check Alpaca IEX status."
        )

    if premarket_sentiment:
        finbert_scores = premarket_sentiment
        logger.info("Using pre-market FinBERT sentiment — skipping in-cycle BERT pass")
    else:
        finbert_scores = {sym: 0.0 for sym in active_symbols}

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

    # Pre-compute SPY N-bar return for relative strength gate (daily bars so it matches sig_bars)
    _, spy_daily = bars_map.get("SPY", (pd.DataFrame(), pd.DataFrame()))
    spy_5bar_return: float | None = None
    if not spy_daily.empty and len(spy_daily) > RS_LOOKBACK_BARS:
        v = spy_daily["close"].pct_change(RS_LOOKBACK_BARS).iloc[-1]
        if not math.isnan(v):
            spy_5bar_return = float(v)

    # Use already-fetched SPY daily bars (yfinance) — avoids redundant Alpaca call that
    # returns only 1 bar on the IEX free tier
    vs_spy_today = 0.0
    if not spy_daily.empty and len(spy_daily) > 1:
        _v = spy_daily["close"].pct_change().iloc[-1]
        if not math.isnan(_v):
            vs_spy_today = float(_v)

    # Prefetch earnings proximity in parallel — avoids 25 sequential yfinance HTTP calls
    earnings_map = _prefetch_earnings_parallel(con, active_symbols)

    # ── Per-symbol decision loop ──────────────────────────────────────────────
    for symbol in active_symbols:
        try:
            bars_5m, bars_daily = bars_map.get(symbol, (pd.DataFrame(), pd.DataFrame()))
            # Use daily bars for XGB/LSTM/regime (matches training data; never < 60 rows).
            # Fall back to 5-min only when daily fetch fails.
            sig_bars = bars_daily if not bars_daily.empty else bars_5m
            if sig_bars.empty:
                continue

            latest = sig_bars.iloc[-1]
            # Prefer the freshest intraday close for price-sensitive calcs (limit orders, ATR stops).
            # Fall back to daily close when 5-min bars are not yet available (early morning).
            current_price = float(
                bars_5m.iloc[-1]["close"] if not bars_5m.empty else latest["close"]
            )
            current_atr   = float(latest.get("atr", 0) or 0)
            volume_ratio  = float(latest.get("volume_ratio", 1.0) or 1.0)
            regime_code   = regime_clf.predict(latest)
            regime_name   = regime_clf.regime_name(regime_code)

            xgb_prob          = xgb.predict_proba(latest)
            lstm_prob         = lstm.predict_proba(sig_bars)
            sentiment         = sentiments.get(symbol, 0.0)
            action_str, ensemble_size = ensemble_signal(
                xgb_prob, lstm_prob, sentiment, regime_name, macro_score=macro_score
            )
            action = action_to_int(action_str)

            # Log every evaluated signal so the dashboard can show live model
            # output even on cycles where no trade fires.
            _log_signal(con, symbol, xgb_prob, lstm_prob, sentiment,
                        macro_score, regime_name, action_str)

            if _handle_exits(con, client, risk, symbol, positions, sell_order_syms,
                             current_price, current_atr, regime_name, portfolio_value,
                             action, pdt_exempt, _stop_fired_today):
                continue

            # ── Entry gates (applied in order of cheapness) ───────────────────
            if action != 1:
                continue

            available_cash = _handle_entry(
                con, client, risk, symbol, positions, buy_order_syms,
                earnings_map, bars_map, sig_bars, latest, current_price,
                current_atr, regime_name, portfolio_value, available_cash,
                xgb_prob, lstm_prob, sentiment, macro_score, macro_cap,
                macro_halt, spy_5bar_return, vs_spy_today, sentiments,
                action, action_str, ensemble_size, pdt_exempt, xgb,
                _stop_fired_today, volume_ratio,
            )

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    con.commit()  # flush all batched signal_log inserts in one fsync (was 25 individual commits)

    # End-of-cycle DB summary — makes "why is the dashboard empty?" obvious from the logs
    try:
        n_trades = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        n_today  = con.execute(
            "SELECT COUNT(*) FROM trades WHERE timestamp LIKE ?", (date.today().isoformat() + "%",)
        ).fetchone()[0]
        n_pos    = con.execute("SELECT COUNT(*) FROM position_state").fetchone()[0]
        last     = con.execute(
            "SELECT portfolio_value FROM trades ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        last_pv  = f"${last[0]:,.2f}" if last else "NONE"
        logger.info(
            f"DB summary — trades total={n_trades} (today={n_today}), open_positions={n_pos}, "
            f"latest portfolio_value={last_pv}"
        )
        if n_trades == 0:
            logger.warning(
                "trades table is EMPTY — dashboard will show $0.00 because portfolio value is "
                "derived from the latest trade row. No trade has executed yet (gates blocking, "
                "no buy signal, or first run). This is expected until the first fill."
            )
    except Exception as _e:
        logger.warning(f"End-of-cycle DB summary failed: {_e}")

    # Prune old signal_log rows (keep last 7 days) to cap DB growth
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        deleted = con.execute(
            "DELETE FROM signal_log WHERE timestamp < ?", (cutoff,)
        ).rowcount
        con.commit()
        if deleted:
            logger.debug(f"signal_log: pruned {deleted} rows older than 7 days")
    except Exception as _e:
        logger.warning(f"signal_log prune failed: {_e}")

    logger.info("=== Trading cycle complete ===")
    con.close()

    try:
        from bot.monitor.sync_db import push_db
        if time.time() - _last_hf_sync > _HF_SYNC_INTERVAL:
            if push_db():
                _last_hf_sync = time.time()
            else:
                logger.warning("trades.db sync to HuggingFace FAILED — dashboard will show stale data")
        else:
            logger.debug(f"HF sync skipped — last push {time.time() - _last_hf_sync:.0f}s ago (<{_HF_SYNC_INTERVAL:.0f}s threshold)")
    except Exception as _e:
        logger.warning(f"HF DB sync skipped: {_e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    default="paper", choices=["paper", "live"])
    parser.add_argument("--summary", action="store_true",
                        help="Send end-of-day Telegram summary and exit")
    parser.add_argument("--loop",    action="store_true",
                        help="Long-running mode: load models once, loop until market close")
    parser.add_argument("--reset-daily-start", action="store_true",
                        help="Clear stale daily_start anchor so Day P&L resets on next cycle")
    parser.add_argument("--clean-db", action="store_true",
                        help="Wipe all bot data for a clean start (reset Alpaca paper account first)")
    args = parser.parse_args()
    try:
        if args.clean_db:
            _do_clean_db()
        elif args.reset_daily_start:
            _do_reset_daily_start()
        elif args.summary:
            end_of_day_summary()
        elif args.loop:
            run_loop(mode=args.mode)
        else:
            run(mode=args.mode)
    except Exception:
        tb = traceback.format_exc()
        logger.error("Bot crashed:\n" + tb)
        sys.stdout.write(f"::error title=Trading Bot Crash::{tb.splitlines()[-1]} — see step log\n")
        sys.stdout.flush()
        sys.exit(1)
