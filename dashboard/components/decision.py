"""Decision center panel."""
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
    _metric_row, _divider, _empty_state, _action_row, _section, _wrap,
    _stat_card, TH, TD, TD0,
)
from dashboard.builders import build_decision_vm
from bot.core.error_logger import safe_render, timed
_logger = logger


# ── PANEL: Decision Center — what to do with each position ────────────────────
# NOTE: render_portfolio_actions, render_sell_analysis, render_position_sizing_panel
#       are consolidated here. They remain functional but are not wired to layout.
@timed(_logger)
@safe_render("Decision Center")
def render_decision_center() -> str:
    vm_rows = build_decision_vm()

    if not vm_rows:
        return (f'<div class="nt nt-wrap">'
                f'{_section("🎯","Decision Center","What to do with each position")}'
                f'{_card(_empty_state("💰", "Fully in cash", "Bot enters trades during market hours when signals align."))}</div>')

    n    = len(vm_rows)
    rows = ""
    for i, r in enumerate(vm_rows):
        td = TD if i < n - 1 else TD0

        score_html = (
            f'<div style="display:inline-flex;align-items:center;gap:5px;">'
            f'<div style="background:{BORDER};border-radius:2px;height:4px;width:40px;overflow:hidden;">'
            f'<div style="background:{r.score_color};height:100%;width:{r.sell_score}%;border-radius:2px;"></div></div>'
            f'<span style="font-size:{FONT_LABEL};color:{r.score_color};font-weight:{WEIGHT_BOLD};">{r.sell_score}</span>'
            f'</div>'
        )

        weight_html = (
            f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">{r.cur_weight:.1f}%</span>'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT3};margin:0 3px;">→</span>'
            f'<span style="font-size:{FONT_LABEL};color:{r.delta_color};font-weight:{WEIGHT_BOLD};">{r.tgt_weight:.1f}%</span>'
        )

        reason_parts = []
        if r.action == "HOLD":
            # HOLD: show what's good first, then any soft concerns
            if r.reasons_hold:
                reason_parts.append(
                    f'<span style="color:{ACTION_BUY};">✓</span>'
                    f'<span style="font-size:{FONT_LABEL};color:{TEXT2};"> {r.reasons_hold[0]}</span>'
                )
            if r.reasons_sell:
                reason_parts.append(
                    f'<span style="color:{TEXT3};">○</span>'
                    f'<span style="font-size:{FONT_LABEL};color:{TEXT3};"> {r.reasons_sell[0]}</span>'
                )
        else:
            # WATCH / TRIM / SELL / EXIT: lead with concerns
            if r.reasons_sell:
                reason_parts.append(
                    f'<span style="color:{ACTION_SELL};">✗</span>'
                    f'<span style="font-size:{FONT_LABEL};color:{TEXT2};"> {r.reasons_sell[0]}</span>'
                )
            if r.reasons_hold:
                reason_parts.append(
                    f'<span style="color:{ACTION_BUY};">✓</span>'
                    f'<span style="font-size:{FONT_LABEL};color:{TEXT2};"> {r.reasons_hold[0]}</span>'
                )
        if not reason_parts and r.pa_reason:
            reason_parts.append(f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">{r.pa_reason}</span>')
        reasons_html = '<br>'.join(reason_parts) if reason_parts else (
            f'<span style="font-size:{FONT_LABEL};color:{TEXT3};">No signal</span>'
        )

        rows += (
            f'<tr>'
            f'<td {td}>{_symbol(r.symbol)}</td>'
            f'<td {td}>{_action_badge(r.action)}</td>'
            f'<td {td}>{score_html}</td>'
            f'<td {td}>{weight_html}</td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT1};'
            f'font-family:Courier New,monospace;">{r.dollar_display}</span></td>'
            f'<td {td}><div style="line-height:1.7;">{reasons_html}</div></td>'
            f'</tr>'
        )

    act_count = sum(1 for r in vm_rows if r.action not in ("HOLD", "WATCH"))
    note = f"{act_count} need action · {n} positions" if act_count else f"{n} positions · holding"
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Symbol</th><th {TH}>Action</th><th {TH}>Score</th>'
        f'<th {TH}>Weight</th><th {TH}>Amount</th><th {TH}>Reasons</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    return f'<div class="nt nt-wrap">{_section("🎯","Decision Center",note)}{table}</div>'
