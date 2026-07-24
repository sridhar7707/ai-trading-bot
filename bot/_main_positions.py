"""Position and trade management helpers extracted from bot/main.py."""
from __future__ import annotations

import math
import sqlite3
from datetime import date, datetime, timedelta, timezone

from loguru import logger

from bot.risk.risk_manager import RiskManager, _business_days_between
import bot.monitor.telegram_bot as tg
from config import (
    KELLY_LOOKBACK_TRADES, KELLY_FRACTION_MAX, CORRELATION_THRESHOLD,
    MAX_HOLD_DAYS, PDT_MAX_DAY_TRADES, PDT_WINDOW_DAYS,
    MAX_POSITION_DRIFT_PCT, MAX_POSITION_PCT,
)
from bot._main_db import log_trade, _save_risk_state
from bot.decision.daily_actions import record as _rec_action
from bot.capital.pool import CapitalPool, update_on_sell as _pool_sell
from bot.strategy.ensemble import BUY_FRACTION


def _opened_today(con: sqlite3.Connection, symbol: str) -> bool:
    today = date.today().isoformat()
    row = con.execute(
        "SELECT 1 FROM trades WHERE symbol=? AND action='BUY' AND timestamp LIKE ? LIMIT 1",
        (symbol, today + "%"),
    ).fetchone()
    return row is not None


def _load_position_state(con: sqlite3.Connection, symbol: str) -> dict | None:
    row = con.execute(
        "SELECT entry_price, high_water_mark, atr_at_entry, opened_at FROM position_state WHERE symbol=?",
        (symbol,),
    ).fetchone()
    return ({"entry_price": row[0], "high_water_mark": row[1],
              "atr_at_entry": row[2], "opened_at": row[3]} if row else None)


def _upsert_position_state(con: sqlite3.Connection, symbol: str, entry_price: float,
                            high_water_mark: float, atr: float) -> None:
    con.execute("""
        INSERT INTO position_state (symbol, entry_price, high_water_mark, atr_at_entry, opened_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            high_water_mark = MAX(high_water_mark, excluded.high_water_mark),
            atr_at_entry    = excluded.atr_at_entry
    """, (symbol, entry_price, high_water_mark, atr, datetime.now(timezone.utc).isoformat()))
    con.commit()


def _delete_position_state(con: sqlite3.Connection, symbol: str) -> None:
    con.execute("DELETE FROM position_state WHERE symbol=?", (symbol,))
    con.commit()


def _kelly_fraction(con: sqlite3.Connection, symbol: str, default: float = BUY_FRACTION) -> float:
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
    """Block buy if any held position has > CORRELATION_THRESHOLD daily-return correlation.
    bars_map values are (bars_5m, bars_daily) tuples; daily bars are used for correlation
    since they have consistent history regardless of time-of-day.
    """
    def _resolve(e):
        if isinstance(e, tuple):
            return e[1] if not e[1].empty else (e[0] if not e[0].empty else None)
        return e if e is not None and not e.empty else None

    entry = bars_map.get(symbol)
    bars_sym = _resolve(entry)
    if bars_sym is None or bars_sym.empty:
        return True
    ret_sym = bars_sym["close"].pct_change().dropna()
    for held in positions:
        if held == symbol:
            continue
        h_entry   = bars_map.get(held)
        bars_held = _resolve(h_entry)
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


def _is_wash_sale_risk(con: sqlite3.Connection, symbol: str) -> bool:
    """IRS wash-sale rule: if the same security was sold at a loss within the past 30 days,
    re-buying it disallows that loss deduction (IRC §1091). Block the buy to avoid the trap.
    We use realized_pnl < 0 as the loss indicator; falls back to pnl_pct < 0 if realised_pnl
    is zero (e.g., entry_price not recorded on older rows).
    """
    # Calendar-day cutoff (IRC §1091 is measured in calendar days, not to the second).
    # Using a date — not a precise datetime — means a loss sale from exactly 30 days
    # ago counts for the whole of that day, and the boundary is deterministic
    # (a microsecond-precise "now - 30d" made the 30-day edge race-dependent).
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=30)).isoformat()
    row = con.execute(
        "SELECT 1 FROM trades WHERE symbol=? AND action LIKE 'SELL%' "
        "AND (realized_pnl < 0 OR (realized_pnl = 0 AND pnl_pct < 0)) "
        "AND timestamp >= ? LIMIT 1",
        (symbol, cutoff),
    ).fetchone()
    if row:
        logger.warning(
            f"Wash-sale guard: {symbol} sold at a loss within 30 days — "
            "skipping buy to avoid IRS loss disallowance (IRC §1091)"
        )
        return True
    return False


def _maybe_record_day_trade(con: sqlite3.Connection, risk: RiskManager, symbol: str,
                             sell_success: bool, pdt_exempt: bool = False) -> None:
    """Record PDT day trade for exits on positions opened today (skipped when account is exempt)."""
    if not pdt_exempt and sell_success and _opened_today(con, symbol):
        today = date.today()
        recent = sum(
            1 for d in risk.day_trade_log
            if _business_days_between(d, today) < PDT_WINDOW_DAYS
        )
        if recent >= PDT_MAX_DAY_TRADES:
            logger.critical(
                f"PDT AUDIT: {symbol} protective exit is day trade #{recent + 1} "
                f"in the 5-business-day window (limit={PDT_MAX_DAY_TRADES}) — "
                "executed to protect capital but FINRA 4210 exposure exceeded. "
                "Review account immediately."
            )
        risk.record_day_trade()
        _save_risk_state(con, risk)


def _reconcile_positions(con: sqlite3.Connection, alpaca_positions: dict,
                          portfolio_value: float = 0.0, client=None) -> None:
    """Sync position_state table with Alpaca's live positions at startup.
    Removes stale DB entries for positions closed externally;
    seeds DB entries for positions opened manually/outside the bot.
    Logs a SELL_RECONCILE trade for any stale DB position so the dashboard
    position walk sees a matching close and no longer shows it as open.
    """
    db_syms = {r[0] for r in con.execute("SELECT symbol FROM position_state").fetchall()}
    for sym in db_syms - set(alpaca_positions.keys()):
        logger.warning(f"Reconcile: {sym} in DB but not Alpaca — closing open trades record")
        rows = con.execute(
            "SELECT action, shares FROM trades WHERE symbol=?", (sym,)
        ).fetchall()
        net_shares = sum(r[1] if r[0] == "BUY" else -r[1] for r in rows)
        if net_shares > 0.001:
            ps = con.execute(
                "SELECT entry_price, opened_at FROM position_state WHERE symbol=?", (sym,)
            ).fetchone()
            entry_price = float(ps[0]) if ps else 0.0
            # Fetch actual current market price so realized P&L is correct.
            sell_price = entry_price
            if client is not None:
                try:
                    sell_price = client.get_latest_price(sym)
                except Exception:
                    sell_price = entry_price
            pnl_pct = (sell_price - entry_price) / entry_price if entry_price > 0 else 0.0
            notional = net_shares * sell_price
            # Calculate holding days from when the position was opened.
            holding_days = 0
            if ps and ps[1]:
                try:
                    opened = datetime.fromisoformat(ps[1].replace("Z", "+00:00"))
                    holding_days = (datetime.now(timezone.utc) - opened).days
                except Exception:
                    holding_days = 0
            log_trade(con, sym, "SELL_RECONCILE", net_shares, sell_price,
                      notional, "reconcile", portfolio_value, pnl_pct,
                      entry_price=entry_price, holding_days=holding_days)
            logger.warning(
                f"Reconcile: logged SELL_RECONCILE for {sym} "
                f"({net_shares:.4f} shares @ ${sell_price:.2f}, entry=${entry_price:.2f}, "
                f"pnl={pnl_pct:+.2%}, {holding_days}d held)"
            )
        _delete_position_state(con, sym)
    for sym, pos in alpaca_positions.items():
        if sym not in db_syms:
            entry = float(getattr(pos, "avg_entry_price", 0) or 0)
            logger.warning(f"Reconcile: {sym} in Alpaca but not DB — seeding position state")
            _upsert_position_state(con, sym, entry, entry, 0.0)


def _trim_position(con: sqlite3.Connection, client, symbol: str, trim_qty: float,
                   current_price: float, regime_name: str, portfolio_value: float,
                   pnl_pct: float, entry_price: float,
                   pool: CapitalPool | None = None) -> bool:
    """Partial sell to reduce an oversized position back to MAX_POSITION_PCT.
    Unlike _signal_sell, does NOT delete position_state — the position still exists.
    """
    result = client.sell(symbol, qty=trim_qty, limit_price=current_price)
    if result:
        client.wait_for_fill(result["order_id"], timeout_secs=12)
        trim_notional = trim_qty * current_price
        log_trade(con, symbol, "SELL_TRIM", trim_qty, current_price, trim_notional,
                  regime_name, portfolio_value, pnl_pct, entry_price=entry_price,
                  order_id=result.get("order_id"))
        if pool:
            cost_basis = entry_price * trim_qty if entry_price > 0 else trim_notional
            _pool_sell(con, pool.id, cost_basis, trim_notional, symbol=symbol)
        _trim_freed_pct = trim_notional / portfolio_value * 100 if portfolio_value > 0 else 0.0
        tg.alert_sell(symbol, trim_qty, current_price, pnl_pct,
                      reason="drift-trim", notional=trim_notional,
                      cash_freed_pct=_trim_freed_pct)
        logger.info(f"TRIM {symbol}: sold {trim_qty:.3f} shares @ ${current_price:.2f} "
                    f"(position drifted above {MAX_POSITION_DRIFT_PCT:.0%} of portfolio)")
        return True
    logger.warning(f"TRIM {symbol}: partial sell order failed")
    return False


def _signal_sell(con: sqlite3.Connection, client, symbol: str, pos_qty: float,
                 current_price: float, regime_name: str, portfolio_value: float,
                 is_from_stop: bool = False, reason: str = "stop-loss",
                 pnl_pct: float = 0.0, entry_price: float = 0.0,
                 holding_days: int = 0,
                 pool: CapitalPool | None = None) -> bool:
    limit_order = client.sell(symbol, qty=pos_qty, limit_price=current_price)
    sell_result = limit_order
    if limit_order:
        filled_qty = client.wait_for_fill(limit_order["order_id"], timeout_secs=12)
        remaining_qty = pos_qty - filled_qty
        if remaining_qty > 0.001 and is_from_stop:
            logger.warning(
                f"Stop limit timed out for {symbol} (order {limit_order['order_id']}) "
                f"filled={filled_qty:.4f}/{pos_qty:.4f} — escalating {remaining_qty:.4f} to market"
            )
            market_order = client.sell_market(symbol, remaining_qty)
            if market_order:
                client.wait_for_fill(market_order["order_id"], timeout_secs=10)
                sell_result = market_order
            else:
                logger.error(
                    f"Market-sell escalation also failed for {symbol} "
                    f"— original limit order {limit_order['order_id']} partially filled "
                    f"({filled_qty:.4f} of {pos_qty:.4f} shares)"
                )
                sell_result = None if filled_qty == 0.0 else limit_order
    if sell_result:
        sell_notional = pos_qty * current_price
        order_id = sell_result.get("order_id")
        if is_from_stop:
            tg.alert_stop_loss(symbol, pnl_pct, notional=sell_notional)
            log_trade(con, symbol, "SELL_STOP", pos_qty, current_price, sell_notional,
                      regime_name, portfolio_value, pnl_pct, entry_price=entry_price,
                      order_id=order_id, holding_days=holding_days)
        else:
            action_tag = "SELL" if reason == "signal" else f"SELL_{reason.upper().replace('-','_')}"
            _freed_pct = sell_notional / portfolio_value * 100 if portfolio_value > 0 else 0.0
            tg.alert_sell(symbol, pos_qty, current_price, pnl_pct, reason=reason,
                          notional=sell_notional, cash_freed_pct=_freed_pct)
            log_trade(con, symbol, action_tag, pos_qty, current_price, sell_notional,
                      regime_name, portfolio_value, pnl_pct, entry_price=entry_price,
                      order_id=order_id, holding_days=holding_days)
        if pool:
            cost_basis = entry_price * pos_qty if entry_price > 0 else pos_qty * current_price
            _pool_sell(con, pool.id, cost_basis, pos_qty * current_price, symbol=symbol)
        _rec_action(con, "sell", symbol,
                    reasoning=f"Exit ({reason}): {pnl_pct:+.1%} P&L",
                    confidence=0, status="executed")
        _delete_position_state(con, symbol)
        return True
    if is_from_stop:
        tg.alert_sell_failed(symbol, reason=reason)
    logger.error(f"SELL ({reason}) failed for {symbol} — will retry next cycle")
    return False


# ── Exit orchestration (moved from _main_cycle.py to keep exit logic co-located) ─

_TP_FLOOR = 0.06
_TP_CEIL  = 0.25


def _atr_tp_pct(atr: float, price: float) -> float:
    return max(_TP_FLOOR, min(_TP_CEIL, (4.0 * atr) / price))


def _handle_exits(
    con: sqlite3.Connection, client, risk, symbol: str, positions: dict,
    sell_order_syms: set, current_price: float, current_atr: float,
    regime_name: str, portfolio_value: float, action: int, pdt_exempt: bool,
    stop_fired_today: set, pool: CapitalPool | None = None,
) -> bool:
    """Handle exit / management for a held position. Returns True when symbol was processed."""
    if symbol not in positions:
        return False

    if symbol in sell_order_syms:
        logger.info(f"Exit management skipped for {symbol} — open sell order pending")
        return True

    pos_state   = _load_position_state(con, symbol)
    entry_price = float(getattr(positions[symbol], "avg_entry_price", 0) or 0)
    pos_qty     = float(positions[symbol].qty)
    pnl_pct     = float(positions[symbol].unrealized_plpc or 0)

    holding_days = 0
    if pos_state and pos_state.get("opened_at"):
        try:
            opened_dt = datetime.fromisoformat(pos_state["opened_at"]).replace(tzinfo=timezone.utc)
            holding_days = (datetime.now(timezone.utc) - opened_dt).days
        except (ValueError, TypeError):
            pass

    # ⓪ Gap-down hard floor — bypass limit/ATR logic, market-sell immediately
    if pnl_pct < -0.10:
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
            if pool:
                cost_basis = entry_price * pos_qty if entry_price > 0 else pos_qty * current_price
                _pool_sell(con, pool.id, cost_basis, pos_qty * current_price, symbol=symbol)
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

    # ① Take-profit: 4×ATR clamped to [6%, 25%]
    if entry_price > 0 and current_atr > 0:
        tp_pct = _atr_tp_pct(current_atr, entry_price)
        if pnl_pct >= tp_pct:
            success = _signal_sell(
                con, client, symbol, pos_qty, current_price,
                regime_name, portfolio_value,
                reason="take-profit", pnl_pct=pnl_pct, entry_price=entry_price,
                holding_days=holding_days, pool=pool,
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
            entry_price=entry_price, holding_days=holding_days, pool=pool,
        )
        if success:
            stop_fired_today.add(symbol)
        _maybe_record_day_trade(con, risk, symbol, success, pdt_exempt=pdt_exempt)
        return True

    # ③ Trailing stop (armed after 3% gain)
    if hwm > entry_price * 1.03 and risk.check_trailing_stop(
            symbol, current_price, hwm, current_atr):
        success = _signal_sell(
            con, client, symbol, pos_qty, current_price,
            regime_name, portfolio_value,
            is_from_stop=True, reason="trailing-stop", pnl_pct=pnl_pct,
            entry_price=entry_price, holding_days=holding_days, pool=pool,
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
                               pnl_pct, entry_price, pool=pool)
                return True

    # ⑤ Time-based forced exit
    if _check_time_exit(pos_state, pnl_pct):
        success = _signal_sell(
            con, client, symbol, pos_qty, current_price,
            regime_name, portfolio_value,
            reason="time-exit", pnl_pct=pnl_pct, entry_price=entry_price,
            holding_days=holding_days, pool=pool,
        )
        _maybe_record_day_trade(con, risk, symbol, success, pdt_exempt=pdt_exempt)
        return True

    # ⑥ Ensemble sell signal
    if action == 2:
        is_day_trade = _opened_today(con, symbol)
        if is_day_trade and not pdt_exempt and not risk.check_pdt(is_day_trade=True):
            logger.warning(f"PDT limit — skipping signal sell of {symbol}")
        else:
            success = _signal_sell(
                con, client, symbol, pos_qty, current_price,
                regime_name, portfolio_value,
                reason="signal", pnl_pct=pnl_pct, entry_price=entry_price,
                holding_days=holding_days, pool=pool,
            )
            if success and is_day_trade and not pdt_exempt:
                risk.record_day_trade()
                _save_risk_state(con, risk)
    return True
