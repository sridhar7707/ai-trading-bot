"""Positions and trades tables."""
from __future__ import annotations

from loguru import logger
from dashboard.design_system import (
    BG, SURFACE, SURFACE2, BORDER, TEXT1, TEXT2, TEXT3,
    ACTION_BUY, ACTION_SELL, ACTION_TRIM,
    ACTION_BUY_BG, ACTION_SELL_BG, ACTION_TRIM_BG,
    GAIN, LOSS, NEURAL,
    FONT_HERO, FONT_SECTION, FONT_VALUE, FONT_LABEL,
    WEIGHT_BOLD, WEIGHT_MEDIUM, WEIGHT_NORMAL,
    CARD_PADDING, SECTION_GAP,
    _card, _label, _section_title, _action_badge, _symbol,
    _metric_row, _divider, _empty_state, _section, _wrap,
    _sym, _badge, _num, _pnl, TH, TD, TD0,
)
from dashboard.data import get_data
from dashboard.builders import build_positions_vm, build_trades_vm
from bot.core.error_logger import safe_render, timed
_logger = logger

_SELL_REASON: dict[str, str] = {
    "SELL":           "Target exit",
    "SELL_STOP":      "Stop-loss triggered",
    "SELL_TP":        "Take-profit hit",
    "SELL_TRAIL":     "Trailing stop hit",
    "SELL_TRIM":      "Oversize trim",
    "SELL_TIME":      "Time-based exit",
    "SELL_ENSEMBLE":  "Signal deteriorated",
    "SELL_RECONCILE": "Reconciled on startup",
}

_BADGE_MAP: dict[str, str] = {
    "SELL_STOP": "SELL", "SELL_TP": "SELL", "SELL_TRAIL": "SELL",
    "SELL_ENSEMBLE": "SELL", "SELL_TIME": "SELL", "SELL_TRIM": "TRIM",
}


# ── Render: positions table ───────────────────────────────────────────────────
@timed(_logger)
@safe_render("Positions")
def render_positions() -> str:
    """Columns: Symbol | Action | Weight | Target | Confidence | P&L"""
    vm_rows = build_positions_vm()

    if not vm_rows:
        return (
            f'<div class="nt nt-wrap">'
            f'{_section("📊","Open Positions","no positions held")}'
            + _card(_empty_state("📊", "No open positions",
                                 "The bot enters trades during market hours "
                                 "(9:30am-4pm ET, Mon-Fri) when signals align."))
            + f'</div>'
        )

    n = len(vm_rows)
    row_htmls = []
    for i, r in enumerate(vm_rows):
        _urgent = r.action in ("EXIT", "SELL")
        _medium = r.action in ("TRIM", "BUY", "ADD")
        if _urgent:
            row_bg = f'background:{ACTION_SELL_BG};border-left:3px solid {ACTION_SELL};'
        elif _medium:
            _c = ACTION_TRIM if r.action == "TRIM" else ACTION_BUY
            _b = ACTION_TRIM_BG if r.action == "TRIM" else ACTION_BUY_BG
            row_bg = f'background:{_b};border-left:3px solid {_c};'
        else:
            row_bg = ""

        conf_int = r.confidence
        conf_c   = r.score_color   # reuse sell_score color as confidence color
        conf_c   = (ACTION_BUY if r.confidence >= 75 else
                    ACTION_TRIM if r.confidence >= 60 else ACTION_SELL)
        conf_html = (
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="background:{BORDER};border-radius:3px;height:5px;width:50px;">'
            f'<div style="background:{conf_c};height:100%;width:{conf_int}%;border-radius:3px;">'
            f'</div></div>'
            f'<span style="font-size:{FONT_LABEL};font-weight:{WEIGHT_BOLD};color:{conf_c};">'
            f'{conf_int}%</span></div>'
        )

        td  = TD if i < n - 1 else TD0
        reason = r.reason
        pnl_d_str = f'{r.pnl_dollar:+,.2f}' if r.pnl_dollar != 0 else '0.00'
        days_str  = f'{r.days_held}d' if r.days_held > 0 else '—'
        days_c    = TEXT2 if r.days_held <= 7 else (NEURAL if r.days_held <= 14 else LOSS)
        stop_str  = f'${r.stop_price:.2f}' if r.stop_price else '—'
        stop_c    = LOSS
        row_htmls.append(
            f'<tr style="{row_bg}">'
            f'<td {td}>{_symbol(r.symbol)}</td>'
            f'<td {td}>{_action_badge(r.action, r.action_size)}</td>'
            f'<td {td}><span style="font-size:{FONT_VALUE};color:{TEXT1};">'
            f'{r.weight_pct:.1f}%</span></td>'
            f'<td {td}><span style="font-size:{FONT_VALUE};color:{TEXT2};">'
            f'{r.target_pct:.1f}%</span></td>'
            f'<td {td}>{conf_html}</td>'
            f'<td {td}><span style="font-weight:{WEIGHT_BOLD};color:{r.pnl_color};">'
            f'{r.pnl_pct:+.1f}%</span>'
            f'<span style="font-size:{FONT_LABEL};color:{r.pnl_color};margin-left:4px;">'
            f'(${pnl_d_str})</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{days_c};">'
            f'{days_str}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{stop_c};">'
            f'{stop_str}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT3};">'
            f'{reason[:50]}{"…" if len(reason) > 50 else ""}</span></td>'
            f'</tr>'
        )

    from dashboard.design_system import _table
    note  = f"{n} position{'s' if n != 1 else ''} · live price · 60s refresh"
    table = _table(
        ["Symbol", "Action", "Weight", "Target", "Confidence", "P&L", "Held", "Stop at", "Reason"],
        row_htmls,
    )
    return (f'<div class="nt nt-wrap">'
            f'{_section("📊","Open Positions",note)}'
            + _card(table, padding="0")
            + f'</div>')


# ── Render: trades table ──────────────────────────────────────────────────────
@safe_render("Trades")
def render_trades() -> str:
    d   = get_data()
    vm_rows = [r for r in build_trades_vm() if r.action != "SELL_RECONCILE"]
    total_trades = d.get("total_trades", 0)

    if not vm_rows:
        return (f'<div class="nt nt-wrap">'
                f'{_section("⚡","Recent Trades")}'
                f'{_card(_empty_state("⚡", "No trades yet", "The bot logs trades here as they execute during market hours."))}</div>')

    shown = len(vm_rows)
    note  = f"last {shown} of {total_trades}" if total_trades > shown else f"{shown} total"

    rows = ""
    for i, r in enumerate(vm_rows):
        pnl_str = f"{r.pnl_pct:+.2%}" if r.pnl_pct is not None else "&mdash;"
        val_str = f"${r.notional:.2f}" if r.notional else "&mdash;"
        qty_str = f"{r.shares:.4f}"    if r.shares   else "&mdash;"
        px_str  = f"${r.price:.2f}"   if r.price    else "&mdash;"
        td   = TD if i < shown - 1 else TD0
        anim = f'style="animation:slideInRow .35s ease both;animation-delay:{i*0.05:.2f}s;"'
        rows += (
            f'<tr {anim}>'
            f'<td {td}><span style="font-family:Courier New,monospace;font-size:{FONT_LABEL};'
            f'color:{TEXT2} !important;">{r.timestamp}</span></td>'
            f'<td {td}>{_sym(r.symbol)}</td>'
            f'<td {td}>{_badge(_BADGE_MAP.get(r.action, r.action))}</td>'
            f'<td {td}>{_num(qty_str)}</td>'
            f'<td {td}>{_num(px_str,bold=True)}</td>'
            f'<td {td}>{_num(val_str,bold=True)}</td>'
            f'<td {td}>{_pnl(pnl_str,big=True)}</td>'
            f'<td {td}><span style="font-size:12px;color:{TEXT2} !important;'
            f'font-weight:600;">{_SELL_REASON.get(r.action, r.regime) if r.action.startswith("SELL") else r.regime}</span></td>'
            f'</tr>'
        )
    legend_row = (
        f'<tr><td colspan="8" style="padding:6px 16px 4px;background:{BG};'
        f'font-size:{FONT_LABEL};color:{TEXT2};border-bottom:1px solid {BORDER};">'
        f'BUY = bot entered a position &nbsp;·&nbsp; '
        f'SELL = bot exited a position &nbsp;·&nbsp; '
        f'TRIM = position reduced (too large) &nbsp;·&nbsp; '
        f'P&amp;L shown on exits only'
        f'</td></tr>'
    )
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Time (CT)</th><th {TH}>Symbol</th>'
        f'<th {TH}>Action</th><th {TH}>Qty</th>'
        f'<th {TH}>Price</th><th {TH}>Value</th>'
        f'<th {TH}>P&amp;L</th><th {TH}>Context</th>'
        f'</tr>{legend_row}</thead><tbody>{rows}</tbody></table>'
    )
    return (f'<div class="nt nt-wrap">'
            f'{_section("⚡","Recent Trades", note)}{table}</div>')
