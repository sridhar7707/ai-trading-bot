"""Rebalance panel."""
from __future__ import annotations

import datetime
from loguru import logger
from dashboard.design_system import (
    BG, SURFACE, SURFACE2, BORDER, TEXT1, TEXT2, TEXT3,
    ACTION_BUY, ACTION_SELL, ACTION_TRIM, ACTION_HOLD, ACTION_WATCH,
    GAIN, LOSS, NEURAL,
    FONT_HERO, FONT_SECTION, FONT_VALUE, FONT_LABEL,
    WEIGHT_BOLD, WEIGHT_MEDIUM, WEIGHT_NORMAL,
    CARD_PADDING, SECTION_GAP,
    _card, _label, _section_title, _action_badge, _symbol,
    _metric_row, _divider, _empty_state, _section, _wrap,
    _stat_card, TH, TD, TD0,
)
from dashboard.data import get_data
from dashboard.builders import build_rebalance_vm
from bot.core.error_logger import safe_render, timed
from bot.core.recommendation_engine import get_portfolio_health
_logger = logger


# ── PANEL: Rebalance — current vs target allocation ───────────────────────────
@timed(_logger)
@safe_render("Rebalance")
def render_rebalance() -> str:
    vm_rows = build_rebalance_vm()

    if not vm_rows:
        return (f'<div class="nt nt-wrap">'
                f'{_section("⚖","Rebalance","Current vs target allocation")}'
                f'{_card(_empty_state("💰", "Fully in cash", "Rebalance panel activates once the bot holds positions."))}</div>')

    d = get_data()
    _pv = 0.0
    try:
        _pv = float(d["portfolio"].replace("$","").replace(",","")) if d["portfolio"] != "—" else 0.0
    except Exception as exc:
        logger.debug(f"parse_portfolio_value render_rebalance: {exc}")

    prices = d.get("prices", {})
    open_pos = d.get("open_pos", {})
    invested_total = sum(
        pos["shares"] * prices.get(sym, pos["invested"] / max(pos["shares"], 1))
        for sym, pos in open_pos.items()
    )
    cash_pct = max(0.0, (_pv - invested_total) / _pv * 100) if _pv > 0 else 0.0

    n    = len(vm_rows)
    rows = ""
    for i, r in enumerate(vm_rows):
        td = TD if i < n - 1 else TD0
        delta_str = f"{r.delta_weight:+.1f}%"
        rows += (
            f'<tr>'
            f'<td {td}>{_symbol(r.symbol)}</td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT1};'
            f'font-family:Courier New,monospace;">{r.cur_weight:.1f}%</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{r.delta_color};'
            f'font-weight:{WEIGHT_BOLD};">{r.tgt_weight:.1f}%</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{r.delta_color};'
            f'font-weight:{WEIGHT_BOLD};">{delta_str}</span></td>'
            f'<td {td}>{_action_badge(r.badge_action, "small")}</td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT1};'
            f'font-family:Courier New,monospace;">{r.dollar_display}</span></td>'
            f'</tr>'
        )

    tgt_sum     = sum(r.tgt_weight for r in vm_rows)
    target_cash = max(0.0, 100.0 - tgt_sum)
    cash_delta  = target_cash - cash_pct
    cash_c      = ACTION_BUY if cash_delta > 1 else (ACTION_SELL if cash_delta < -1 else TEXT2)
    cash_badge  = "ADD" if cash_delta > 1 else ("TRIM" if cash_delta < -1 else "HOLD")
    rows += (
        f'<tr>'
        f'<td {TD0}><span style="font-family:Courier New,monospace;font-weight:{WEIGHT_BOLD};'
        f'color:{TEXT3};font-size:{FONT_VALUE};">CASH</span></td>'
        f'<td {TD0}><span style="font-size:{FONT_LABEL};color:{TEXT1};'
        f'font-family:Courier New,monospace;">{cash_pct:.1f}%</span></td>'
        f'<td {TD0}><span style="font-size:{FONT_LABEL};color:{cash_c};'
        f'font-weight:{WEIGHT_BOLD};">{target_cash:.1f}%</span></td>'
        f'<td {TD0}><span style="font-size:{FONT_LABEL};color:{cash_c};'
        f'font-weight:{WEIGHT_BOLD};">{cash_delta:+.1f}%</span></td>'
        f'<td {TD0}>{_action_badge(cash_badge, "small")}</td>'
        f'<td {TD0}>—</td>'
        f'</tr>'
    )

    net_rebalance = sum(abs(r.delta_dollars) for r in vm_rows) / 2
    net_str = f"${net_rebalance:,.0f}" if net_rebalance > 0 else "—"

    health       = get_portfolio_health(d)
    health_score = health.get("total", 0)
    grade        = health.get("grade", "—")

    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Symbol</th><th {TH}>Current</th><th {TH}>Target</th>'
        f'<th {TH}>Delta</th><th {TH}>Action</th><th {TH}>Amount</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    summary = (
        f'<div style="display:flex;gap:0;flex-direction:column;">'
        + _metric_row("Net to rebalance", net_str, TEXT1)
        + _metric_row("Health score", f"{health_score}/100", ACTION_BUY if health_score >= 75 else (ACTION_TRIM if health_score >= 50 else ACTION_SELL), grade)
        + f'</div>'
    )
    note = f"{n} positions · ~{net_str} to rebalance"
    return (
        f'<div class="nt nt-wrap">'
        f'{_section("⚖","Rebalance",note)}'
        f'{table}'
        f'{_card(summary)}'
        f'</div>'
    )


# ── Gradio layout — 4-tab design ──────────────────────────────────────────────
# Gradio 5 removed every= from components. Use gr.Timer + .tick() instead.
