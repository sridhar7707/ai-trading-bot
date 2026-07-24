"""Weekly holdings review — what happened, why, and why we still hold."""
from __future__ import annotations

import datetime
import json
from loguru import logger

from dashboard.design_system import (
    BG, SURFACE, SURFACE2, BORDER, TEXT1, TEXT2, TEXT3,
    GAIN, LOSS, NEURAL, PRIMARY,
    FONT_SECTION, FONT_VALUE, FONT_LABEL,
    WEIGHT_BOLD, WEIGHT_MEDIUM, WEIGHT_NORMAL,
    _section, _empty_state, _card, _wrap,
    TH, TD, TD0,
)
from dashboard.data import get_data, get_db_conn, DB_PATH
from bot.core.error_logger import safe_render, timed
import os

_logger = logger

_FEATURE_LABELS: dict[str, str] = {
    "atr_pct":         "High volatility",
    "bb_width":        "Bollinger breakout",
    "ema_spread":      "Strong uptrend",
    "high_52w_pct":    "Near 52-week high",
    "mom_12_1":        "12-month momentum",
    "volume_ratio":    "Above-avg volume",
    "rsi":             "RSI signal",
    "ret_5d":          "5-day return",
    "ret_21d":         "1-month return",
    "ret_63d":         "3-month return",
    "ret_126d":        "6-month return",
    "hl_ratio":        "Wide candle range",
    "vwap_dev":        "VWAP deviation",
    "macd_diff_pct":   "MACD crossover",
    "bb_position":     "BB position",
    "obv_chg_pct":     "OBV momentum",
    "vol_ratio_trend": "Volume trend up",
    "gap_overnight":   "Overnight gap",
    "rsi_divergence":  "RSI divergence",
    "macd_cross_up":   "MACD cross up",
    "mfi":             "Money flow",
}

_ACTION_LABEL: dict[str, tuple[str, str]] = {
    "STRONG_BUY": ("Strengthening &uarr;&uarr;", GAIN),
    "BUY":        ("Bullish &uarr;", GAIN),
    "HOLD":       ("Neutral &mdash;", NEURAL),
    "SELL":       ("Weakening &darr;", LOSS),
    "STRONG_SELL":("Bearish &darr;&darr;", LOSS),
}

_SELL_REASONS: dict[str, str] = {
    "SELL_STOP":       "Stop-loss triggered",
    "SELL_TP":         "Take-profit hit",
    "SELL_TRAIL":      "Trailing stop hit",
    "SELL_TIME_EXIT":  "Time exit (7 days)",
    "SELL_ENSEMBLE":   "AI signal flipped bearish",
    "SELL_TRIM":       "Position oversized — trimmed",
    "SELL_RECONCILE":  "Reconciled on restart",
    "SELL":            "Manual / AI exit",
}

_REGIME_DESC: dict[str, str] = {
    "HIGH_VOLATILITY": "volatile market",
    "RANGING":         "sideways market",
    "BULL":            "bull market",
    "BEAR":            "bear market",
    "TRENDING_UP":     "uptrending market",
    "TRENDING_DOWN":   "downtrending market",
}


def _why_hold(action: str, score: float, regime: str) -> str:
    """One-sentence reason the bot still holds the position."""
    regime_desc = _REGIME_DESC.get(regime, regime.lower().replace("_", " ") if regime else "current market")
    if action in ("STRONG_BUY", "BUY"):
        return f"AI ensemble score {score:.2f} — signal remains bullish in a {regime_desc}."
    if action == "HOLD":
        return f"AI ensemble score {score:.2f} — signal neutral; no exit trigger met in a {regime_desc}."
    if action in ("SELL", "STRONG_SELL"):
        return f"AI signal turned bearish (score {score:.2f}) — bot may exit soon."
    return f"Ensemble score {score:.2f}; no stop-loss or take-profit triggered."


@timed(_logger)
@safe_render("Weekly Summary")
def render_weekly_summary() -> str:
    if not os.path.exists(DB_PATH):
        return (f'<div class="nt nt-wrap">'
                f'{_section("📅","Weekly Holdings Review","—")}'
                + _card(_empty_state("📅", "No data yet", "Check back after the first trading session."))
                + f'</div>')

    d      = get_data()
    prices = d.get("prices", {})
    today  = datetime.date.today()
    mon    = today - datetime.timedelta(days=today.weekday())
    week_label = f"Week of {mon.strftime('%b %d, %Y')}"

    try:
        with get_db_conn() as con:
            # Week-start portfolio value (Monday's earliest snapshot)
            week_snap = con.execute(
                "SELECT portfolio_value FROM portfolio_snapshots "
                "WHERE date(timestamp) >= ? ORDER BY timestamp ASC LIMIT 1",
                (str(mon),),
            ).fetchone()

            cur_snap = con.execute(
                "SELECT portfolio_value FROM portfolio_snapshots "
                "ORDER BY timestamp DESC LIMIT 1",
            ).fetchone()

            positions = con.execute(
                "SELECT symbol, entry_price, high_water_mark, opened_at "
                "FROM position_state",
            ).fetchall()

            # Latest signal per held symbol
            syms = [r[0] for r in positions]
            sig_map: dict[str, dict] = {}
            if syms:
                placeholders = ",".join("?" * len(syms))
                sig_rows = con.execute(
                    f"SELECT s.symbol, s.ensemble_score, s.ensemble_action, s.regime "
                    f"FROM signal_log s "
                    f"INNER JOIN (SELECT symbol, MAX(timestamp) AS mt FROM signal_log "
                    f"            WHERE symbol IN ({placeholders}) GROUP BY symbol) m "
                    f"ON s.symbol = m.symbol AND s.timestamp = m.mt",
                    syms,
                ).fetchall()
                sig_map = {r[0]: {"score": r[1], "action": r[2], "regime": r[3]}
                           for r in sig_rows}

            # Entry trade feature drivers for each position
            entry_map: dict[str, dict] = {}
            for sym, *_ in positions:
                row = con.execute(
                    "SELECT feature_drivers, ensemble_score, regime FROM trades "
                    "WHERE symbol = ? AND action = 'BUY' ORDER BY timestamp DESC LIMIT 1",
                    (sym,),
                ).fetchone()
                if row and row[0]:
                    try:
                        entry_map[sym] = {
                            "drivers": json.loads(row[0]),
                            "score":   row[1],
                            "regime":  row[2],
                        }
                    except Exception as exc:
                        _logger.debug(f"weekly_summary: entry_map parse for {sym}: {exc}")

            # Sells this week
            week_sells = con.execute(
                "SELECT symbol, action, pnl_pct, realized_pnl, holding_days, timestamp "
                "FROM trades WHERE action LIKE 'SELL%' AND date(timestamp) >= ? "
                "ORDER BY timestamp DESC",
                (str(mon),),
            ).fetchall()

    except Exception as exc:
        _logger.warning(f"render_weekly_summary DB: {exc}")
        positions = []
        week_snap = cur_snap = None
        sig_map = entry_map = {}
        week_sells = []

    # ── Week P&L banner ──────────────────────────────────────────────────────
    pv_start = float(week_snap[0]) if week_snap else None
    pv_now   = float(cur_snap[0])  if cur_snap  else d.get("portfolio_value") or 0

    if pv_start and pv_now:
        delta  = pv_now - pv_start
        delta_pct = delta / pv_start * 100
        pc = GAIN if delta >= 0 else LOSS
        banner = (
            f'<div style="display:flex;gap:28px;flex-wrap:wrap;'
            f'background:{SURFACE};border-radius:8px;padding:14px 20px;'
            f'border-left:4px solid {pc};margin-bottom:12px;">'
            + _stat("This Week", f'{"+" if delta>=0 else ""}{delta_pct:.2f}%', pc)
            + _stat("Value",     f'${pv_now:,.0f}', TEXT1)
            + _stat("Positions", str(len(positions)), TEXT1)
            + _stat("Closed",    str(len(week_sells)), TEXT1)
            + f'</div>'
        )
    else:
        banner = ""

    # ── Per-position breakdown ───────────────────────────────────────────────
    pos_rows = ""
    for i, (sym, entry_px, hwm, opened_at) in enumerate(positions):
        cur_px = prices.get(sym) or entry_px or 0
        pnl_pct = (cur_px - entry_px) / entry_px * 100 if entry_px else 0
        pnl_c   = GAIN if pnl_pct >= 0 else LOSS

        try:
            opened_dt = datetime.datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
            days_held = (datetime.datetime.now(datetime.timezone.utc) - opened_dt).days
        except Exception:
            days_held = 0

        entry_info = entry_map.get(sym, {})
        drivers    = entry_info.get("drivers", [])
        entry_reg  = entry_info.get("regime", "")
        why_bought = " + ".join(
            _FEATURE_LABELS.get(d[0], d[0]) for d in drivers[:2]
        ) or "Ensemble signal"

        sig      = sig_map.get(sym, {})
        action   = sig.get("action", "HOLD")
        cur_score = sig.get("score") or entry_info.get("score") or 0
        regime   = sig.get("regime") or entry_reg or ""
        act_label, act_c = _ACTION_LABEL.get(action, ("Neutral", NEURAL))
        hold_reason = _why_hold(action, cur_score, regime)

        td = TD if i < len(positions) - 1 else TD0
        pos_rows += (
            f'<tr>'
            f'<td {td}>'
            f'<span style="font-family:Courier New,monospace;font-weight:700;'
            f'font-size:14px;color:{PRIMARY};">{sym}</span>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT3};">{days_held}d held</div>'
            f'</td>'
            f'<td {td}>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">Entry ${entry_px:.2f}</div>'
            f'<div style="font-size:{FONT_VALUE};font-weight:{WEIGHT_BOLD};color:{TEXT1};">${cur_px:.2f}</div>'
            f'</td>'
            f'<td {td}>'
            f'<span style="font-weight:{WEIGHT_BOLD};color:{pnl_c};">'
            f'{"+" if pnl_pct>=0 else ""}{pnl_pct:.2f}%</span>'
            f'</td>'
            f'<td {td}>'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT1};">{why_bought}</span>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT3};">'
            f'{_REGIME_DESC.get(entry_reg,entry_reg)} · ens {entry_info.get("score",0):.2f}</div>'
            f'</td>'
            f'<td {td}>'
            f'<span style="font-weight:{WEIGHT_BOLD};color:{act_c};">{act_label}</span>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};margin-top:3px;">{hold_reason}</div>'
            f'</td>'
            f'</tr>'
        )

    if pos_rows:
        pos_section = _wrap(
            f'<table class="nt-tbl"><thead><tr>'
            f'<th {TH}>Symbol</th>'
            f'<th {TH}>Price</th>'
            f'<th {TH}>P&amp;L</th>'
            f'<th {TH}>Why we bought it</th>'
            f'<th {TH}>Why we still hold it</th>'
            f'</tr></thead><tbody>{pos_rows}</tbody></table>'
        )
    else:
        pos_section = _card(
            _empty_state("📭", "No open positions", "The bot will enter new positions during market hours.")
        )

    # ── Closed this week ─────────────────────────────────────────────────────
    closed_section = ""
    if week_sells:
        rows = ""
        for j, (sym, action, pnl_p, pnl_d, held, ts) in enumerate(week_sells):
            p  = (pnl_p or 0) * 100
            dl = pnl_d or 0
            c  = GAIN if p >= 0 else LOSS
            td = TD if j < len(week_sells) - 1 else TD0
            reason = _SELL_REASONS.get(action, action.replace("_", " ").title())
            rows += (
                f'<tr>'
                f'<td {td}><span style="font-family:Courier New,monospace;font-weight:700;'
                f'font-size:14px;color:{PRIMARY};">{sym}</span></td>'
                f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{reason}</span></td>'
                f'<td {td}><span style="font-weight:{WEIGHT_BOLD};color:{c};">'
                f'{"+" if p>=0 else ""}{p:.2f}%</span></td>'
                f'<td {td}><span style="font-weight:{WEIGHT_BOLD};color:{c};">'
                f'{"+" if dl>=0 else ""}${dl:,.0f}</span></td>'
                f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT3};">{held}d</span></td>'
                f'</tr>'
            )
        closed_section = (
            _section("📤", "Closed This Week", f"{len(week_sells)} exit{'s' if len(week_sells)!=1 else ''}")
            + _wrap(
                f'<table class="nt-tbl"><thead><tr>'
                f'<th {TH}>Symbol</th><th {TH}>Reason</th>'
                f'<th {TH}>P&amp;L %</th><th {TH}>P&amp;L $</th><th {TH}>Days held</th>'
                f'</tr></thead><tbody>{rows}</tbody></table>'
            )
        )

    return (
        f'<div class="nt nt-wrap">'
        + _section("📅", "Weekly Holdings Review", week_label)
        + _card(banner + pos_section, padding="0" if pos_rows else None)
        + closed_section
        + f'</div>'
    )


def _stat(label: str, value: str, color: str) -> str:
    return (
        f'<div>'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};margin-bottom:2px;">{label}</div>'
        f'<div style="font-size:{FONT_VALUE};font-weight:{WEIGHT_BOLD};color:{color};">{value}</div>'
        f'</div>'
    )


from dashboard.registry import ComponentSpec, RefreshGroup, register
register(ComponentSpec("weekly_summary_out", RefreshGroup.FAST, render_weekly_summary, priority=30))
