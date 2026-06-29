"""Today's actions and portfolio actions panels."""
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
    _stat_card, _sym, TH, TD, TD0,
)
from dashboard.builders import build_actions_vm
from bot.core.error_logger import safe_render
_logger = logger


# ── PANEL 2: Today's Priority Actions &mdash; recommendation-based ─────────────────
@safe_render("Today's Actions")
def render_todays_actions() -> str:
    vm_rows = build_actions_vm()

    if not vm_rows:
        return (f'<div class="nt nt-wrap">'
                f'{_section("⚡","Priority Actions","What to do right now")}'
                f'{_card(_empty_state("💰", "Fully in cash", "Bot enters trades during market hours when signals align."))}</div>')

    n    = len(vm_rows)
    rows = ""
    for i, r in enumerate(vm_rows):
        td = TD if i < n - 1 else TD0
        conf   = r.confidence
        conf_c = GAIN if conf >= 75 else (NEURAL if conf >= 60 else TEXT2)
        urg_c  = LOSS if r.urgency == "high" else (NEURAL if r.urgency == "medium" else TEXT2)
        urg_dot = (
            f'<span style="width:6px;height:6px;border-radius:50%;'
            f'background:{urg_c};display:inline-block;margin-left:4px;"></span>'
        )
        rows += (
            f'<tr>'
            f'<td {td}>{_sym(r.symbol)}</td>'
            f'<td {td}>{_action_badge(r.action, r.badge_size)}{urg_dot}</td>'
            f'<td {td}><span style="font-weight:700;color:{conf_c};">{conf}%</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{r.reason}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT1};font-family:Courier New,monospace;">'
            f'{r.detail}</span></td>'
            f'</tr>'
        )

    urgent_count = sum(1 for r in vm_rows if r.urgency == "high")
    note = (f"{urgent_count} urgent · {n} positions" if urgent_count else
            f"{n} position{'s' if n != 1 else ''} · sorted by priority")
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Symbol</th><th {TH}>Action</th>'
        f'<th {TH}>Conf.</th><th {TH}>Reason</th><th {TH}>Adjust</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    return f'<div class="nt nt-wrap">{_section("⚡","Priority Actions",note)}{table}</div>'


# ── Render: portfolio actions &mdash; dead code (called internally by decision center)
@safe_render("Portfolio Actions")
def render_portfolio_actions() -> str:
    from dashboard.data import get_data
    from dashboard.design_system import _action_badge
    from bot.core.recommendation_engine import get_portfolio_action, get_position_sizing
    d        = get_data()
    open_pos = d["open_pos"]
    prices   = d["prices"]
    df       = d["trades_df"]

    if not open_pos:
        return (f'<div class="nt nt-wrap">'
                f'{_section("🎯","Portfolio Actions","AI recommendation per position")}'
                f'{_card(_empty_state("💰", "Fully in cash", "Bot enters trades during market hours when signals align."))}</div>')

    _pv = 0.0
    try:
        _pv = float(d["portfolio"].replace("$", "").replace(",", "")) if d["portfolio"] != "&mdash;" else 0.0
    except Exception as exc:
        logger.debug(f"parse_portfolio_value render_portfolio_actions: {exc}")

    _ens: dict[str, float] = {}
    if not df.empty:
        buys = df[df["action"] == "BUY"]
        for sym in open_pos:
            sym_buys = buys[buys["symbol"] == sym]
            if not sym_buys.empty:
                _ens[sym] = float(sym_buys.iloc[-1].get("ensemble_score", 0.0) or 0.0)

    _AMBER = "#f59e0b"
    rows  = ""
    items = list(open_pos.items())
    for i, (sym, pos) in enumerate(items):
        cur      = prices.get(sym, 0.0)
        invested = pos["invested"]
        cur_val  = pos["shares"] * cur
        pnl_pct  = ((cur_val - invested) / invested * 100) if invested > 0 else 0.0
        pos_pct  = (cur_val / _pv * 100) if _pv > 0 else 0.0
        ens      = _ens.get(sym, 1.0)

        sz_pts   = 30 if pos_pct > 25 else (20 if pos_pct > 15 else (10 if pos_pct > 10 else 0))
        pr_pts   = (30 if pnl_pct > 50 else (20 if pnl_pct > 25 else (10 if pnl_pct > 10 else 0))) if pnl_pct > 0 else 0
        cf_pts   = 25 if ens < 0.55 else (15 if ens < 0.65 else 0)
        dd_pts   = (15 if pnl_pct < -8 else (10 if pnl_pct < -5 else 0)) if pnl_pct < 0 else 0
        total    = sz_pts + pr_pts + cf_pts + dd_pts

        scored: list[tuple[int, str]] = []
        if sz_pts: scored.append((sz_pts, "Position oversized"))
        if pr_pts:
            scored.append((pr_pts, "Profit > 50%" if pnl_pct > 50 else ("Profit > 25%" if pnl_pct > 25 else "Profit > 10%")))
        if cf_pts: scored.append((cf_pts, "AI confidence weakening"))
        if dd_pts: scored.append((dd_pts, "Drawdown risk"))
        scored.sort(key=lambda x: -x[0])
        reason = scored[0][1] if scored else "All metrics healthy"

        if total <= 30:   label, bc, bbg = "HOLD",  GAIN,   "#0a2010"
        elif total <= 59: label, bc, bbg = "WATCH", NEURAL, "#1a1030"
        elif total <= 79: label, bc, bbg = "TRIM",  _AMBER, "#2a1f08"
        else:             label, bc, bbg = "EXIT",  LOSS,   "#2a0a0a"

        pnl_c   = GAIN if pnl_pct >= 0 else LOSS
        ens_c   = GAIN if ens >= 0.75 else (NEURAL if ens >= 0.60 else TEXT2)
        ens_str = f"{ens*100:.0f}%" if ens > 0 else "&mdash;"
        td = TD if i < len(items) - 1 else TD0
        rows += (
            f'<tr><td {td}>{_sym(sym)}</td>'
            f'<td {td}>{_action_badge(label)}</td>'
            f'<td {td}><span style="font-weight:700;color:{ens_c};">{ens_str}</span></td>'
            f'<td {td}><span style="font-weight:700;color:{pnl_c};">{pnl_pct:+.1f}%</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{reason}</span></td>'
            f'</tr>'
        )

    note  = f"{len(items)} position{'s' if len(items) != 1 else ''}"
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Symbol</th><th {TH}>Action</th><th {TH}>AI Score</th>'
        f'<th {TH}>P&amp;L</th><th {TH}>Top Reason</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    return (f'<div class="nt nt-wrap">'
            f'{_section("🎯","Portfolio Actions","AI recommendation per position")}{table}</div>')
