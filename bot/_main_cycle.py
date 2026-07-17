"""Per-symbol cycle logic extracted from bot/main.py: fetch, exits, and entries."""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

import bot.monitor.telegram_bot as tg
from bot.strategy.ensemble import WEIGHTS, BUY_FRACTION
from bot.strategy.features import compute_features
from config import (
    ATR_MAX_STOP_PCT, ATR_MIN_STOP_PCT, ATR_STOP_MULTIPLIER,
    ENTRY_REGIMES, KELLY_FRACTION_MAX,
    MAX_POSITION_DRIFT_PCT, MAX_POSITION_PCT,
    MAX_RISK_PER_TRADE_PCT, MAX_SECTOR_EXPOSURE_PCT,
    MIN_CASH_RESERVE_PCT, MIN_RR_RATIO, MIN_TP_PCT,
    MACD_CONFIRMATION_MIN, MIN_VOLUME_RATIO, RANGING_SIZE_FACTOR, XGB_MIN_CONFIDENCE,
    RS_LOOKBACK_BARS, SECTOR_MAP, STOP_LOSS_PCT,
)
from database.user_settings import get_setting as _get_setting
from bot._main_db import log_trade, _save_risk_state
from bot._main_market import _log_buy_skip
from bot._main_positions import (
    _check_time_exit, _delete_position_state, _is_wash_sale_risk,
    _kelly_fraction, _load_position_state, _maybe_record_day_trade,
    _opened_today, _passes_correlation_gate, _signal_sell, _trim_position,
    _upsert_position_state,
)

# TP ceiling is 12% so worst-case R:R at the stop ceiling (10%) stays at 1.2,
# keeping a 20% buffer above MIN_RR_RATIO=1.0 to absorb transaction costs.
_TP_FLOOR = 0.06
_TP_CEIL  = 0.12


def _atr_tp_pct(atr: float, price: float) -> float:
    return max(_TP_FLOOR, min(_TP_CEIL, (4.0 * atr) / price))


def _fetch_symbol(symbol: str, client, yf_batch: dict) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Return (symbol, bars_5m, bars_daily).

    bars_5m  — intraday 5-min bars; empty when not enough today yet (< 60 bars) or
               stale (feed broken).  Used for current price only.
    bars_daily — 1-year daily OHLCV from yfinance; used for XGB/LSTM/regime (matches training).
    Both empty → skip this symbol entirely (feed is stale/broken).
    """
    feed_stale = False
    bars_5m    = pd.DataFrame()

    try:
        raw = compute_features(client.get_bars(symbol, timeframe="5Min", limit=200))
        if not raw.empty:
            last_ts  = raw.index[-1]
            now_utc  = pd.Timestamp.now(tz="UTC")
            last_utc = (last_ts.tz_localize("UTC")
                        if last_ts.tzinfo is None else last_ts.tz_convert("UTC"))
            age_mins = (now_utc - last_utc).total_seconds() / 60
            if age_mins > 30:
                logger.warning(
                    f"Stale bars for {symbol}: last bar is {age_mins:.0f}m old — skipping"
                )
                feed_stale = True
            else:
                bars_5m = raw
    except ValueError:
        # Not enough intraday bars yet (normal early-day condition with IEX free tier)
        pass
    except Exception as e:
        logger.warning(f"5min bar fetch failed for {symbol}: {e}")
        feed_stale = True

    if feed_stale:
        return symbol, pd.DataFrame(), pd.DataFrame()

    # Daily bars from pre-fetched batch (thread-safe; computed before thread pool)
    bars_daily = pd.DataFrame()
    raw_d = yf_batch.get(symbol)
    if raw_d is not None and not raw_d.empty:
        try:
            bars_daily = compute_features(raw_d)
        except Exception as e:
            logger.warning(f"Daily bar features failed for {symbol}: {e}")

    return symbol, bars_5m, bars_daily


def _handle_exits(
    con: sqlite3.Connection, client, risk, symbol: str, positions: dict,
    sell_order_syms: set, current_price: float, current_atr: float,
    regime_name: str, portfolio_value: float, action: int, pdt_exempt: bool,
    stop_fired_today: set,
) -> bool:
    """Handle exit / management for a held position. Returns True when symbol was processed."""
    if symbol not in positions:
        return False

    # Brokerage guard: skip exit processing entirely when a sell order is already
    # open for this symbol. Submitting a second sell order while one is pending
    # could fill both, creating an unintended short position.
    if symbol in sell_order_syms:
        logger.info(
            f"Exit management skipped for {symbol} — open sell order pending"
        )
        return True

    pos_state   = _load_position_state(con, symbol)
    entry_price = float(getattr(positions[symbol], "avg_entry_price", 0) or 0)
    pos_qty     = float(positions[symbol].qty)
    pnl_pct     = float(positions[symbol].unrealized_plpc or 0)

    # Compute holding period for audit trail (SEC reconciliation)
    holding_days = 0
    if pos_state and pos_state.get("opened_at"):
        try:
            opened_dt = datetime.fromisoformat(pos_state["opened_at"]).replace(tzinfo=timezone.utc)
            holding_days = (datetime.now(timezone.utc) - opened_dt).days
        except (ValueError, TypeError):
            pass

    # ⓪ Gap-down hard floor — bypass limit/ATR logic, market-sell immediately
    if pnl_pct < -0.10:
        if symbol in sell_order_syms:
            logger.info(
                f"Gap-down exit skipped for {symbol} — open sell order pending "
                "(prevents duplicate fill → unintended short position)"
            )
            return True
        logger.warning(f"Gap-down floor: {symbol} pnl={pnl_pct:.1%} — immediate market sell")
        sell_result = client.sell_market(symbol, pos_qty)
        if sell_result:
            client.wait_for_fill(sell_result["order_id"], timeout_secs=10)
            tg.alert_stop_loss(symbol, pnl_pct, notional=pos_qty * current_price)
            log_trade(con, symbol, "SELL_GAP_DOWN", pos_qty, current_price,
                      pos_qty * current_price, regime_name, portfolio_value, pnl_pct,
                      entry_price=entry_price,
                      order_id=sell_result.get("order_id"),
                      holding_days=holding_days)
            _delete_position_state(con, symbol)
            _maybe_record_day_trade(con, risk, symbol, True, pdt_exempt=pdt_exempt)
        return True

    if pos_state:
        new_hwm = max(pos_state["high_water_mark"], current_price)
        if new_hwm > pos_state["high_water_mark"]:
            _upsert_position_state(con, symbol, entry_price, new_hwm, current_atr)
        hwm = new_hwm
    else:
        _upsert_position_state(con, symbol, entry_price, current_price, current_atr)
        hwm = current_price

    # ① Take-profit: 4×ATR clamped to [6%, 12%] — captures medium-term swing moves
    if entry_price > 0 and current_atr > 0:
        tp_pct = _atr_tp_pct(current_atr, entry_price)
        if pnl_pct >= tp_pct:
            success = _signal_sell(
                con, client, symbol, pos_qty, current_price,
                regime_name, portfolio_value,
                reason="take-profit", pnl_pct=pnl_pct, entry_price=entry_price,
                holding_days=holding_days
            )
            _maybe_record_day_trade(con, risk, symbol, success, pdt_exempt=pdt_exempt)
            return True

    # ② ATR stop-loss
    _stop_triggered = risk.check_stop_loss(symbol, current_price, entry_price,
                                           atr=current_atr, pnl_pct=pnl_pct)
    logger.debug(f"Stop-loss check {symbol}: pnl={pnl_pct:.1%} triggered={_stop_triggered}")
    if _stop_triggered:
        success = _signal_sell(
            con, client, symbol, pos_qty, current_price,
            regime_name, portfolio_value,
            is_from_stop=True, reason="stop-loss", pnl_pct=pnl_pct,
            entry_price=entry_price, holding_days=holding_days
        )
        if success:
            stop_fired_today.add(symbol)
        _maybe_record_day_trade(con, risk, symbol, success, pdt_exempt=pdt_exempt)
        return True

    # ③ Trailing stop (armed after 0.5% gain)
    if hwm > entry_price * 1.005 and risk.check_trailing_stop(
            symbol, current_price, hwm, current_atr):
        success = _signal_sell(
            con, client, symbol, pos_qty, current_price,
            regime_name, portfolio_value,
            is_from_stop=True, reason="trailing-stop", pnl_pct=pnl_pct,
            entry_price=entry_price, holding_days=holding_days
        )
        if success:
            stop_fired_today.add(symbol)
        _maybe_record_day_trade(con, risk, symbol, success, pdt_exempt=pdt_exempt)
        return True

    # ④ Drift trim — partial sell if position has grown above MAX_POSITION_DRIFT_PCT
    if portfolio_value > 0:
        position_pct = (pos_qty * current_price) / portfolio_value
        if position_pct > MAX_POSITION_DRIFT_PCT:
            target_notional = portfolio_value * MAX_POSITION_PCT
            trim_qty = (pos_qty * current_price - target_notional) / current_price
            if trim_qty >= 0.001:
                logger.info(
                    f"{symbol} at {position_pct:.1%} of portfolio "
                    f"(max {MAX_POSITION_DRIFT_PCT:.0%}) — trimming ${trim_qty * current_price:.0f}"
                )
                _trim_position(con, client, symbol, round(trim_qty, 3),
                               current_price, regime_name, portfolio_value,
                               pnl_pct, entry_price)
                return True  # re-evaluate next cycle with updated qty

    # ⑤ Time-based forced exit — free capital from stale positions
    if _check_time_exit(pos_state, pnl_pct):
        success = _signal_sell(
            con, client, symbol, pos_qty, current_price,
            regime_name, portfolio_value,
            reason="time-exit", pnl_pct=pnl_pct, entry_price=entry_price,
            holding_days=holding_days
        )
        _maybe_record_day_trade(con, risk, symbol, success, pdt_exempt=pdt_exempt)
        return True

    # ⑤ Ensemble sell signal
    if action == 2:
        is_day_trade = _opened_today(con, symbol)
        if is_day_trade and not pdt_exempt and not risk.check_pdt(is_day_trade=True):
            logger.warning(f"PDT limit — skipping signal sell of {symbol}")
        else:
            success = _signal_sell(
                con, client, symbol, pos_qty, current_price,
                regime_name, portfolio_value,
                reason="signal", pnl_pct=pnl_pct, entry_price=entry_price,
                holding_days=holding_days
            )
            if success and is_day_trade and not pdt_exempt:
                risk.record_day_trade()
                _save_risk_state(con, risk)
    return True


def compute_tradeable_capital(con: sqlite3.Connection, portfolio_value: float) -> float:
    """Return the capital available for new positions respecting the reinvestment setting.

    When reinvest_profits_only=true: tradeable = max(0, portfolio_value - initial_deposit).
    Computes once per cycle and is passed into _handle_entry, avoiding per-symbol DB reads.
    """
    if _get_setting("reinvest_profits_only", "false") != "true":
        return portfolio_value

    dep_str = _get_setting("initial_deposit", None)
    initial: float | None = None
    if dep_str:
        try:
            initial = float(dep_str)
        except Exception:
            pass

    if initial is None:
        try:
            row = con.execute(
                "SELECT portfolio_value FROM portfolio_snapshots "
                "WHERE portfolio_value > 0 ORDER BY timestamp ASC LIMIT 1"
            ).fetchone()
            if row:
                initial = float(row[0])
        except Exception:
            pass

    if initial is None:
        try:
            row = con.execute(
                "SELECT portfolio_value FROM trades "
                "WHERE portfolio_value > 0 ORDER BY id ASC LIMIT 1"
            ).fetchone()
            if row:
                initial = float(row[0])
        except Exception:
            pass

    if initial is None:
        logger.warning(
            "reinvest_profits_only=true but initial deposit unknown "
            "(no initial_deposit setting and no portfolio history) — trading full portfolio"
        )
        return portfolio_value

    tradeable = max(0.0, portfolio_value - initial)
    logger.debug(
        f"reinvest-profits-only: tradeable=${tradeable:.0f} "
        f"(portfolio=${portfolio_value:.0f}, deposit=${initial:.0f})"
    )
    return tradeable


def _handle_entry(
    con: sqlite3.Connection, client, risk, symbol: str, positions: dict,
    buy_order_syms: set, earnings_map: dict, bars_map: dict,
    sig_bars: pd.DataFrame, latest, current_price: float, current_atr: float,
    regime_name: str, portfolio_value: float, available_cash: float,
    xgb_prob: float, lstm_prob: float, sentiment: float, macro_score: float,
    macro_cap: float, macro_halt: bool, spy_5bar_return: float | None,
    vs_spy_today: float, sentiments: dict, action: int, action_str: str,
    ensemble_size: float, pdt_exempt: bool, xgb, stop_fired_today: set,
    volume_ratio: float, tradeable_capital: float,
) -> float:
    """Process entry gates and buy execution. Returns updated available_cash."""
    # Gate 0 — VIX emergency halt: no new positions when VIX >= 40
    if macro_halt:
        _log_buy_skip(symbol, "VIX emergency halt")
        return available_cash

    # Gate 1 — Regime: only buy in trending or ranging markets
    if regime_name not in ENTRY_REGIMES:
        _log_buy_skip(symbol, f"regime={regime_name} (allowed: {ENTRY_REGIMES})")
        return available_cash

    # Gate 2 — Volume: confirm institutional participation
    if volume_ratio < MIN_VOLUME_RATIO:
        _log_buy_skip(symbol, f"volume ratio {volume_ratio:.2f} < {MIN_VOLUME_RATIO}")
        return available_cash

    # Gate 3 — XGB minimum confidence: live trades show 62% WR at >=0.55 vs 25% below
    if xgb_prob < XGB_MIN_CONFIDENCE:
        _log_buy_skip(symbol, f"xgb_prob {xgb_prob:.3f} < min {XGB_MIN_CONFIDENCE:.2f}")
        return available_cash

    # Gate 4 — Relative strength: stock must be outperforming SPY over last N bars
    if spy_5bar_return is not None and symbol != "SPY":
        stock_5bar = sig_bars["close"].pct_change(RS_LOOKBACK_BARS).iloc[-1]
        if not math.isnan(stock_5bar) and float(stock_5bar) < spy_5bar_return:
            _log_buy_skip(
                symbol,
                f"RS weak ({stock_5bar:.2%} vs SPY {spy_5bar_return:.2%})"
            )
            return available_cash

    # Gate 5 — Open order: no duplicate limit buy submissions
    if symbol in buy_order_syms:
        _log_buy_skip(symbol, "open buy order already pending")
        return available_cash

    # Gate 6 — Earnings proximity (prefetched in parallel before loop)
    if earnings_map.get(symbol, False):
        _log_buy_skip(symbol, "earnings proximity")
        return available_cash

    # Gate 7 — Correlation: avoid adding a position highly correlated with existing holdings
    if not _passes_correlation_gate(symbol, positions, bars_map):
        return available_cash

    # Gate 7.5 — Wash-sale guard (IRS IRC §1091): block re-buy within 30 days of a loss sale
    if _is_wash_sale_risk(con, symbol):
        _log_buy_skip(symbol, "wash-sale guard active")
        return available_cash

    # Gate 7.7 — Stop re-entry block: don't re-buy a symbol whose stop fired today
    if symbol in stop_fired_today:
        _log_buy_skip(symbol, "stop-loss fired earlier today (re-entry blocked)")
        return available_cash

    # Gate 7.9 — MACD confirmation (daily bars only — intraday MACD oscillates too fast)
    # Disabled by default (MACD_CONFIRMATION_MIN=-inf). Set to 0.0 to require positive crossover.
    if not math.isinf(MACD_CONFIRMATION_MIN):
        _, _daily_bars = bars_map.get(symbol, (pd.DataFrame(), pd.DataFrame()))
        if not _daily_bars.empty:
            _macd_diff = float(latest.get("macd_diff", 0.0))
            if _macd_diff <= MACD_CONFIRMATION_MIN:
                _log_buy_skip(
                    symbol,
                    f"MACD gate: daily macd_diff={_macd_diff:.4f} <= min={MACD_CONFIRMATION_MIN}"
                )
                return available_cash

    # Gate 8 — Cash and risk approval
    # ensemble_size: STRONG_BUY=0.20, BUY=0.12 — use as confidence multiplier on Kelly
    kelly_f      = _kelly_fraction(con, symbol)
    confidence   = ensemble_size / BUY_FRACTION  # 1.0 for BUY, 1.67 for STRONG_BUY
    pos_fraction = min(kelly_f * macro_cap * confidence, KELLY_FRACTION_MAX)

    # Reinvestment guard: tradeable_capital is pre-computed once per cycle by main.py
    # using compute_tradeable_capital() so we avoid per-symbol DB reads here.
    if tradeable_capital <= 0.0:
        _log_buy_skip(symbol, "no tradeable capital (profits only mode, no profits yet)")
        return available_cash
    notional = tradeable_capital * pos_fraction
    if notional <= 0.0:
        _log_buy_skip(symbol, f"zero position size (kelly={kelly_f:.3f}, macro_cap={macro_cap:.2f})")
        return available_cash

    # Risk-per-trade cap: size so max dollar loss ≤ MAX_RISK_PER_TRADE_PCT of portfolio.
    # Derives the implied stop % from ATR (same formula as risk_manager), then back-calculates
    # max safe notional — volatile stocks get smaller positions automatically.
    if current_atr and current_atr > 0 and current_price > 0:
        stop_pct = max(ATR_MIN_STOP_PCT, min(ATR_MAX_STOP_PCT,
                       (ATR_STOP_MULTIPLIER * current_atr) / current_price))
        tp_target_pct = _atr_tp_pct(current_atr, current_price)
    else:
        stop_pct = STOP_LOSS_PCT
        tp_target_pct = _TP_FLOOR

    # Gate 8a — Minimum absolute profit target: not worth entering if upside < MIN_TP_PCT
    if tp_target_pct < MIN_TP_PCT:
        _log_buy_skip(symbol, f"TP target {tp_target_pct:.1%} < min {MIN_TP_PCT:.1%}")
        return available_cash

    # Gate 8b — Minimum risk/reward: require TP ≥ MIN_RR_RATIO × stop distance
    rr_ratio = tp_target_pct / stop_pct
    if rr_ratio < MIN_RR_RATIO:
        _log_buy_skip(
            symbol,
            f"R:R {rr_ratio:.2f} < min {MIN_RR_RATIO} (TP={tp_target_pct:.1%}, stop={stop_pct:.1%})"
        )
        return available_cash

    max_risk_notional = (portfolio_value * MAX_RISK_PER_TRADE_PCT) / stop_pct
    if notional > max_risk_notional:
        logger.info(
            f"BUY {symbol}: notional capped ${notional:.0f}→${max_risk_notional:.0f} "
            f"(stop_pct={stop_pct:.1%}, max_risk={MAX_RISK_PER_TRADE_PCT:.1%})"
        )
        notional = max_risk_notional

    # Gate 8c — Reduce position by RANGING_SIZE_FACTOR in sideways markets (lower conviction)
    if regime_name == "RANGING":
        notional *= RANGING_SIZE_FACTOR
        logger.debug(f"BUY {symbol}: RANGING regime — size reduced to ${notional:.0f}")

    # Gate 8d — Sector exposure cap: total portfolio value in this sector ≤ MAX_SECTOR_EXPOSURE_PCT
    _sym_sector = SECTOR_MAP.get(symbol, "Unknown")
    if _sym_sector not in ("Unknown", "Broad_ETF"):
        _sector_val = sum(
            float(getattr(pos, "market_value", 0) or 0)
            for sym, pos in positions.items()
            if SECTOR_MAP.get(sym, "Unknown") == _sym_sector
        )
        _sector_pct = _sector_val / portfolio_value if portfolio_value > 0 else 0
        if _sector_pct >= MAX_SECTOR_EXPOSURE_PCT:
            _log_buy_skip(
                symbol,
                f"{_sym_sector} sector at {_sector_pct:.1%} of portfolio (max {MAX_SECTOR_EXPOSURE_PCT:.0%})"
            )
            return available_cash

    # Gate 8e — Cash reserve: always keep MIN_CASH_RESERVE_PCT uninvested
    _min_reserve = portfolio_value * MIN_CASH_RESERVE_PCT
    if notional > available_cash * 0.95:
        logger.warning(
            f"BUY {symbol} skipped — need ${notional:.2f}, "
            f"running cash ${available_cash:.2f}"
        )
        return available_cash
    if available_cash - notional < _min_reserve:
        logger.info(
            f"BUY {symbol} skipped — would breach cash reserve "
            f"(need ${notional:.0f}, reserve=${_min_reserve:.0f}, cash=${available_cash:.0f})"
        )
        return available_cash
    if not risk.approve_buy(symbol, notional, portfolio_value,
                            portfolio_value, positions):
        return available_cash

    result = client.buy(symbol, notional, limit_price=current_price)
    if result:
        filled = client.wait_for_fill(result["order_id"], timeout_secs=15)
        if filled:
            # Use actual fill price for P&L accuracy; fall back to limit estimate
            _actual_fill = client.get_fill_price(result["order_id"])
            if _actual_fill is None:
                fill_price = current_price
                logger.warning(
                    f"BUY {symbol}: actual fill price unavailable — "
                    f"using limit estimate ${current_price:.2f} (audit: cost basis may differ)"
                )
            else:
                fill_price = _actual_fill
                slippage_bps = (_actual_fill - current_price) / current_price * 10_000
                logger.info(
                    f"BUY {symbol}: filled ${_actual_fill:.2f} "
                    f"({slippage_bps:+.1f} bps vs limit ${current_price:.2f})"
                )
            fill_shares = notional / fill_price
            _drivers = xgb.explain(latest)
            _sent_s = sentiments.get(symbol, 0.0)
            _ens_score = (
                WEIGHTS["xgb"]       * xgb_prob +
                WEIGHTS["lstm"]      * lstm_prob +
                WEIGHTS["sentiment"] * ((_sent_s + 1.0) / 2.0) +
                WEIGHTS["macro"]     * macro_score
            )
            _sym_sector_tg = SECTOR_MAP.get(symbol, "")
            _sect_pct_tg = 0.0
            if _sym_sector_tg and _sym_sector_tg not in ("Unknown", "Broad_ETF") and portfolio_value > 0:
                _existing_sv = sum(
                    float(getattr(p, "market_value", 0) or 0)
                    for s2, p in positions.items()
                    if SECTOR_MAP.get(s2, "") == _sym_sector_tg
                )
                _sect_pct_tg = (_existing_sv + notional) / portfolio_value * 100
            _cash_pct_tg = ((available_cash - notional) / portfolio_value * 100
                            if portfolio_value > 0 else 0.0)
            tg.alert_buy(symbol, fill_shares, fill_price,
                         regime_name, portfolio_value, vs_spy_today * 100,
                         notional=notional,
                         xgb_prob=xgb_prob, lstm_prob=lstm_prob,
                         sentiment_score=_sent_s,
                         ensemble_score=_ens_score,
                         drivers=_drivers,
                         sector=_sym_sector_tg,
                         sector_pct_after=_sect_pct_tg,
                         cash_pct_after=_cash_pct_tg)
            _ai_rsn = (
                f"Bought {symbol} at ${fill_price:.2f}. "
                f"XGB: {xgb_prob:.0%}, LSTM: {lstm_prob:.0%}, regime: {regime_name}. "
                f"Stop: {stop_pct:.1%}, target: {tp_target_pct:.1%}, R:R: {rr_ratio:.2f}x."
            )
            log_trade(con, symbol, "BUY", fill_shares,
                      fill_price, notional, regime_name, portfolio_value, 0,
                      xgb_prob=xgb_prob, lstm_prob=lstm_prob,
                      sentiment_score=sentiments.get(symbol, 0.0),
                      macro_score=macro_score,
                      order_id=result.get("order_id"),
                      feature_drivers=json.dumps(_drivers) if _drivers is not None else None,
                      ai_reasoning=_ai_rsn,
                      stop_loss=round(fill_price * (1 - stop_pct), 4),
                      take_profit=round(fill_price * (1 + tp_target_pct), 4),
                      risk_reward_ratio=round(rr_ratio, 4))
            _upsert_position_state(con, symbol, fill_price, fill_price, current_atr)
            available_cash -= notional
            buy_order_syms.discard(symbol)  # order is now filled, not pending
        else:
            logger.warning(f"BUY {symbol} order did not fill — position state NOT recorded")

    return available_cash
