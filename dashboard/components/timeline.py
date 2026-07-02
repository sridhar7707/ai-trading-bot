"""Decision Timeline — chronological log of every decision per position (req 11.6)."""
from __future__ import annotations
import datetime
from loguru import logger
from dashboard.design_system import (
    SURFACE, SURFACE2, BORDER, TEXT1, TEXT2, TEXT3,
    ACTION_BUY, ACTION_SELL, ACTION_TRIM, ACTION_HOLD, ACTION_WATCH,
    GAIN, LOSS, NEURAL, PRIMARY,
    FONT_SECTION, FONT_VALUE, FONT_LABEL, WEIGHT_BOLD,
    _section, _wrap, _card, _empty_state,
    _sym,
)
from dashboard.data import get_data, get_db_conn, DB_PATH, safe_query
from bot.core.error_logger import safe_render, timed, log_exception
import os

_logger = logger

_TYPE_COLORS = {
    "buy":    GAIN,
    "add":    GAIN,
    "trim":   "#f59e0b",
    "sell":   LOSS,
    "review": NEURAL,
    "hold":   TEXT2,
}

_TYPE_ICONS = {
    "buy": "●", "add": "●", "trim": "◑",
    "sell": "○", "review": "◇", "hold": "·",
}


def log_decision(symbol: str, decision_type: str, price: float,
                 quantity: float, reasoning: str, confidence: int | None = None,
                 portfolio_value: float | None = None,
                 triggered_by: str = "ai") -> bool:
    """Insert a decision_log entry. Safe to call from trading engine."""
    if not os.path.exists(DB_PATH):
        return False
    today = datetime.date.today().isoformat()
    try:
        with get_db_conn() as con:
            con.execute(
                "INSERT INTO decision_log "
                "(symbol, decision_date, decision_type, price_at_decision,"
                "quantity_changed, reasoning, ai_confidence, portfolio_value_at_time, triggered_by)"
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (symbol, today, decision_type.lower(), price,
                 quantity, reasoning, confidence, portfolio_value, triggered_by)
            )
            con.commit()
        return True
    except Exception as exc:
        log_exception(_logger, "log_decision", exc, {"symbol": symbol})
        return False


def _load_timeline(symbol: str) -> list[dict]:
    if not os.path.exists(DB_PATH):
        return []
    try:
        with get_db_conn() as con:
            rows = con.execute(
                "SELECT decision_date, decision_type, price_at_decision,"
                "quantity_changed, reasoning, ai_confidence, portfolio_value_at_time,"
                "triggered_by FROM decision_log WHERE symbol=? ORDER BY decision_date ASC,"
                "decision_id ASC",
                (symbol,)
            ).fetchall()
        cols = ["decision_date", "decision_type", "price_at_decision",
                "quantity_changed", "reasoning", "ai_confidence",
                "portfolio_value_at_time", "triggered_by"]
        return [dict(zip(cols, r)) for r in rows]
    except Exception as exc:
        log_exception(_logger, "_load_timeline", exc, {"symbol": symbol})
        return []


def _sync_from_trades(symbol: str | None = None) -> None:
    """One-time backfill: create decision_log entries from existing trades rows."""
    if not os.path.exists(DB_PATH):
        return
    try:
        with get_db_conn() as con:
            # Per-symbol check when a symbol is given; global check for full backfill
            if symbol:
                count = con.execute(
                    "SELECT COUNT(*) FROM decision_log WHERE symbol=?", (symbol,)
                ).fetchone()[0]
            else:
                count = con.execute("SELECT COUNT(*) FROM decision_log").fetchone()[0]
            if count > 0:
                return
            where = "WHERE symbol=?" if symbol else ""
            params = (symbol,) if symbol else ()
            rows = con.execute(
                f"SELECT symbol, date(timestamp), action, price, shares,"
                f"ensemble_score, portfolio_value, ai_reasoning FROM trades "
                f"{where} ORDER BY id ASC",
                params
            ).fetchall()
            for r in rows:
                sym, dt, action, price, shares, ens, pv, reasoning = r
                if not dt:
                    continue
                dt_str = str(dt)[:10]
                dtype  = "buy"  if "BUY"  in str(action) else "sell"
                conf   = int(float(ens or 0) * 100) if ens else None
                note   = reasoning or f"{action} via AI bot"
                con.execute(
                    "INSERT OR IGNORE INTO decision_log "
                    "(symbol, decision_date, decision_type, price_at_decision,"
                    "quantity_changed, reasoning, ai_confidence,"
                    "portfolio_value_at_time, triggered_by) VALUES (?,?,?,?,?,?,?,?,?)",
                    (sym, dt_str, dtype, price, shares, note, conf, pv, "ai")
                )
            con.commit()
    except Exception as exc:
        log_exception(_logger, "_sync_from_trades", exc)


def render_decision_timeline(symbol: str | None = None) -> str:
    """Render decision timeline for a single symbol."""
    if not symbol:
        return (
            f'<div class="nt nt-wrap">'
            f'{_section("⏱", "Decision Timeline", "Full history per position")}'
            f'{_card(_empty_state("⏱", "Select a symbol", "Choose a symbol to view its decision timeline."))}'
            f'</div>'
        )
    _sync_from_trades(symbol)
    entries = _load_timeline(symbol)

    if not entries:
        return (
            f'<div class="nt nt-wrap">'
            f'{_section("⏱", "Decision Timeline", symbol)}'
            f'{_card(_empty_state("⏱", f"No history for {symbol}", "History will appear after trades are logged."))}'
            f'</div>'
        )

    # Calculate total return for closed positions
    buy_prices: list[float]  = [e["price_at_decision"] or 0.0 for e in entries if e["decision_type"] == "buy"]
    sell_prices: list[float] = [e["price_at_decision"] or 0.0 for e in entries if e["decision_type"] == "sell"]
    total_return_str = ""
    first_entry_date = entries[0]["decision_date"]
    last_exit_date   = None
    if buy_prices and sell_prices:
        avg_buy  = sum(buy_prices)  / len(buy_prices)
        avg_sell = sum(sell_prices) / len(sell_prices)
        ret = (avg_sell - avg_buy) / avg_buy * 100 if avg_buy > 0 else 0.0
        ret_c = GAIN if ret >= 0 else LOSS
        last_exit_date = max(e["decision_date"] for e in entries if e["decision_type"] == "sell")
        try:
            d0 = datetime.date.fromisoformat(first_entry_date)
            d1 = datetime.date.fromisoformat(last_exit_date)
            hold_days = (d1 - d0).days
        except Exception:
            hold_days = 0
        total_return_str = (
            f'<div style="background:{ret_c}11;border:1px solid {ret_c}44;border-radius:6px;'
            f'padding:10px 14px;margin-top:12px;display:flex;gap:20px;flex-wrap:wrap;">'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">Total Return</span>'
            f'<span style="font-weight:{WEIGHT_BOLD};color:{ret_c};font-size:{FONT_VALUE};">'
            f'{ret:+.1f}%</span>'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">Holding Period</span>'
            f'<span style="font-weight:{WEIGHT_BOLD};color:{TEXT1};">{hold_days} days</span>'
            f'</div>'
        )

    # Timeline entries
    items_html = ""
    for entry in entries:
        dtype   = entry["decision_type"] or "review"
        color   = _TYPE_COLORS.get(dtype, TEXT2)
        icon    = _TYPE_ICONS.get(dtype, "·")
        date_s  = entry["decision_date"] or "—"
        price   = entry["price_at_decision"]
        qty     = entry["quantity_changed"]
        reason  = entry["reasoning"] or "—"
        conf    = entry["ai_confidence"]
        by_who  = entry["triggered_by"] or "ai"

        price_str = f"@ ${price:.2f}" if price else ""
        qty_str   = f"  {qty:+.2f} shares" if qty else ""
        conf_str  = f"  · {conf}% confidence" if conf else ""
        by_str    = f"  · {by_who}" if by_who not in ("ai", "") else ""

        items_html += (
            f'<div style="display:flex;gap:14px;padding:10px 0;'
            f'border-bottom:1px solid {BORDER};">'
            f'<div style="flex:0 0 90px;font-size:{FONT_LABEL};color:{TEXT2};">{date_s}</div>'
            f'<div style="flex:0 0 12px;color:{color};font-size:18px;line-height:1;'
            f'margin-top:1px;">{icon}</div>'
            f'<div style="flex:1;">'
            f'<div style="font-weight:{WEIGHT_BOLD};color:{color};'
            f'text-transform:capitalize;font-size:{FONT_LABEL};">'
            f'{dtype.title()} {price_str}{qty_str}</div>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};margin-top:2px;line-height:1.5;">'
            f'&#8220;{reason}&#8221;'
            f'<span style="color:{TEXT3};">{conf_str}{by_str}</span></div>'
            f'</div>'
            f'</div>'
        )

    body = items_html + total_return_str
    inner = _wrap(f'<div style="max-height:480px;overflow-y:auto;">{body}</div>')
    return (
        f'<div class="nt nt-wrap">'
        f'{_section("⏱", "Decision Timeline", symbol)}'
        f'{inner}'
        f'</div>'
    )


@timed(_logger)
@safe_render("Decision Timeline")
def render_all_timelines() -> str:
    """Overview: all symbols with decision history, most recent first."""
    _sync_from_trades()
    if not os.path.exists(DB_PATH):
        return (f'<div class="nt nt-wrap">'
                f'{_section("⏱", "Decision Timeline", "All positions")}'
                f'{_card(_empty_state("⏱", "No trade history", "History appears after trades are logged."))}'
                f'</div>')
    try:
        with get_db_conn() as con:
            rows = con.execute(
                "SELECT DISTINCT symbol, MAX(decision_date) AS last_date,"
                "COUNT(*) AS n_decisions FROM decision_log GROUP BY symbol "
                "ORDER BY last_date DESC LIMIT 20"
            ).fetchall()
    except Exception:
        rows = []

    if not rows:
        return (f'<div class="nt nt-wrap">'
                f'{_section("⏱", "Decision Timeline", "All positions")}'
                f'{_card(_empty_state("⏱", "No decisions logged yet", ""))}'
                f'</div>')

    items_html = ""
    for sym, last_date, n in rows:
        items_html += (
            f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'padding:8px 14px;border-bottom:1px solid {BORDER};">'
            f'<span style="font-family:Courier New,monospace;font-weight:700;color:{TEXT1};">'
            f'{sym}</span>'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">'
            f'{n} decision{"s" if n != 1 else ""}</span>'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT3};">Last: {last_date}</span>'
            f'</div>'
        )
    note = f"{len(rows)} symbol{'s' if len(rows) != 1 else ''} with decision history"
    return (
        f'<div class="nt nt-wrap">'
        f'{_section("⏱", "Decision Timeline", note)}'
        f'{_wrap(items_html)}'
        f'</div>'
    )
