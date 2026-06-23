"""Sell analysis, position sizing panels."""
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
    _sym, _stat_card, TH, TD, TD0,
)
from dashboard.data import get_data
from bot.core.error_logger import safe_render
from bot.core.recommendation_engine import (
    get_portfolio_action, get_position_sizing, get_sell_analysis,
    get_recommendation_explanation,
)
_logger = logger

# ── Render: position sizing recommendations ────────────────────────────────────
@safe_render("Position Sizing")
def render_position_sizing() -> str:
    d        = get_data()
    open_pos = d["open_pos"]
    prices   = d["prices"]
    df       = d["trades_df"]

    if not open_pos:
        return (f'<div class="nt nt-wrap">'
                f'{_section("📐","Position Sizing","conviction-based target allocation")}'
                f'{_card(_empty_state("💰", "Fully in cash", "Bot enters trades during market hours when signals align."))}</div>')

    _pv = 0.0
    try:
        _pv = float(d["portfolio"].replace("$", "").replace(",", "")) if d["portfolio"] != "—" else 0.0
    except Exception as exc:
        logger.debug(f"parse_portfolio_value render_position_sizing: {exc}")

    _ens: dict[str, float] = {}
    if not df.empty:
        buys = df[df["action"] == "BUY"]
        for sym in open_pos:
            sym_buys = buys[buys["symbol"] == sym]
            if not sym_buys.empty:
                _ens[sym] = float(sym_buys.iloc[-1].get("ensemble_score", 0.65) or 0.65)

    rows  = ""
    items = list(open_pos.items())
    for i, (sym, pos) in enumerate(items):
        cur      = prices.get(sym, 0.0)
        cur_val  = pos["shares"] * cur if cur > 0 else pos["invested"]
        cur_pct  = (cur_val / _pv * 100) if _pv > 0 else 0.0
        ens      = _ens.get(sym, 0.65)

        if ens >= 0.75:   target_pct, rationale = 12.0, "High conviction"
        elif ens >= 0.65: target_pct, rationale = 8.0,  "Moderate conviction"
        elif ens >= 0.55: target_pct, rationale = 5.0,  "Low conviction"
        else:             target_pct, rationale = 3.0,  "Very low — consider exit"

        delta = target_pct - cur_pct
        if abs(delta) < 0.5:   adj_lbl, adj_c = "On target",           TEXT2
        elif delta > 0:        adj_lbl, adj_c = f"Add +{delta:.1f}%",  GAIN
        else:                  adj_lbl, adj_c = f"Reduce {delta:.1f}%", "#f59e0b"

        target_val = (_pv * target_pct / 100) if _pv > 0 else 0.0
        val_hint   = f"(~${target_val:,.0f})" if target_val > 0 else ""
        td = TD if i < len(items) - 1 else TD0
        rows += (
            f'<tr><td {td}>{_sym(sym)}</td>'
            f'<td {td}><span style="font-weight:700;color:{TEXT1};">{cur_pct:.1f}%</span></td>'
            f'<td {td}><span style="font-weight:700;color:{NEURAL};">{target_pct:.0f}%</span></td>'
            f'<td {td}><span style="font-weight:700;color:{adj_c};">{adj_lbl}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{rationale}</span>'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT2};margin-left:6px;">{val_hint}</span></td>'
            f'</tr>'
        )

    help_block = (
        f'<div style="background:{BG};border-top:1px solid {BORDER};'
        f'padding:8px 14px;font-size:{FONT_LABEL};color:{TEXT2};line-height:1.6;">'
        f'Target derived from AI ensemble score &nbsp;·&nbsp; '
        f'75%+ = 12% &nbsp;·&nbsp; 65%+ = 8% &nbsp;·&nbsp; 55%+ = 5% &nbsp;·&nbsp; &lt;55% = 3%'
        f'</div>'
    )
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Symbol</th><th {TH}>Current</th><th {TH}>Target</th>'
        f'<th {TH}>Adjustment</th><th {TH}>Rationale</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>' + help_block
    )
    return (f'<div class="nt nt-wrap">'
            f'{_section("📐","Position Sizing","conviction-based target allocation")}{table}</div>')


# ── Render: AI investment committee ─────────────────────────────────────────────
@safe_render("AI Committee")
def render_ai_committee() -> str:
    d        = get_data()
    open_pos = d["open_pos"]
    df       = d["trades_df"]

    if not open_pos:
        return (f'<div class="nt nt-wrap">'
                f'{_section("🏛","AI Committee","XGBoost · LSTM · Sentiment votes")}'
                f'{_card(_empty_state("🏛", "Fully in cash", "Committee convenes once the bot enters positions."))}</div>')

    # Extract latest BUY scores per symbol from trades_df
    _votes: dict[str, dict] = {}
    if not df.empty:
        buys = df[df["action"] == "BUY"]
        for sym in open_pos:
            sym_buys = buys[buys["symbol"] == sym]
            if not sym_buys.empty:
                lb = sym_buys.iloc[-1]
                _votes[sym] = {
                    "xgb":  float(lb.get("xgb_prob",        0.0) or 0.0),
                    "lstm": float(lb.get("lstm_prob",        0.0) or 0.0),
                    "sent": float(lb.get("sentiment_score",  0.0) or 0.0),
                }

    def _vote_chip(label: str, pct_val: float, threshold: float = 0.60) -> str:
        vote  = "BUY" if pct_val >= threshold else ("HOLD" if pct_val >= 0.45 else "SELL")
        c     = GAIN if vote == "BUY" else (TEXT2 if vote == "HOLD" else LOSS)
        v_str = f"{pct_val*100:.0f}%" if pct_val > 0 else "—"
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;gap:3px;'
            f'background:{BG};border-radius:6px;padding:8px 10px;min-width:64px;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;letter-spacing:.6px;">{label}</div>'
            f'<div style="font-size:{FONT_VALUE};font-weight:700;color:{c};">{v_str}</div>'
            f'<div style="font-size:{FONT_LABEL};font-weight:700;color:{c};">{vote}</div></div>'
        )

    rows_html = ""
    for i, (sym, _pos) in enumerate(list(open_pos.items())[:8]):
        v     = _votes.get(sym, {})
        xgb   = v.get("xgb",  0.0)
        lstm  = v.get("lstm", 0.0)
        sent  = v.get("sent", 0.0)
        sent_n = min(max((sent + 1) / 2, 0.0), 1.0)   # -1..1 → 0..1

        buy_votes = (1 if xgb >= 0.60 else 0) + (1 if lstm >= 0.60 else 0) + (1 if sent_n >= 0.55 else 0)
        verdict_c = GAIN if buy_votes >= 2 else (NEURAL if buy_votes == 1 else LOSS)
        verdict   = f"{buy_votes}/3 BUY" if buy_votes > 0 else "No BUY votes"
        no_data   = not v

        border_b = f'border-bottom:1px solid {BORDER};' if i < min(len(open_pos), 8) - 1 else ''
        if no_data:
            chip_html = (f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">'
                         f'No BUY trade on record yet</span>')
        else:
            chip_html = (
                _vote_chip("XGBoost", xgb)
                + _vote_chip("LSTM", lstm)
                + _vote_chip("Sentiment", sent_n, 0.55)
            )
        rows_html += (
            f'<div style="display:flex;align-items:center;gap:12px;padding:12px 14px;{border_b}">'
            f'{_sym(sym)}'
            f'<div style="display:flex;gap:6px;flex:1;">{chip_html}</div>'
            f'<div style="text-align:right;min-width:80px;">'
            f'<div style="font-size:{FONT_LABEL};font-weight:700;color:{verdict_c};">'
            f'{"—" if no_data else verdict}</div></div></div>'
        )

    if not rows_html:
        rows_html = _empty_state("🏛", "No positions", "Committee convenes once the bot enters positions.")

    return (f'<div class="nt nt-wrap">'
            f'{_section("🏛","AI Committee","3-model vote per open position")}'
            f'{_wrap(rows_html)}</div>')


# ── PANEL 3: Sell Analysis — called internally by render_decision_center ──────
@safe_render("Sell Analysis")
def render_sell_analysis() -> str:
    d        = get_data()
    open_pos = d.get("open_pos", {})

    if not open_pos:
        return f'<div class="nt nt-wrap">{_section("📉","Sell Analysis","When should I sell?")}{_card(_empty_state("💰", "Fully in cash", "Sell analysis runs once the bot holds positions."))}</div>'

    _REC_ORDER = {"EXIT": 0, "SELL": 1, "TRIM": 2, "WATCH": 3, "HOLD": 4}

    analyses = []
    for sym in open_pos:
        sa = get_sell_analysis(sym, d)
        sa["symbol"] = sym
        analyses.append(sa)
    analyses.sort(key=lambda a: (_REC_ORDER.get(a["recommendation"], 9), -a["sell_score"]))

    n    = len(analyses)
    rows = ""
    for i, sa in enumerate(analyses):
        sym    = sa["symbol"]
        score  = sa["sell_score"]
        rec    = sa["recommendation"]
        unreal = sa.get("unrealised_pct", 0.0)
        pw     = sa.get("position_weight", 0.0)
        ens    = sa.get("ensemble_score", 0.0)
        reasons_sell = sa.get("reasons_to_sell", [])
        reasons_hold = sa.get("reasons_to_hold", [])
        trim_pct = sa.get("trim_amount_pct", 0)

        td = TD if i < n - 1 else TD0
        bar_c = LOSS if score > 65 else (NEURAL if score > 35 else GAIN)
        bar_html = (
            f'<div style="display:inline-flex;align-items:center;gap:6px;">'
            f'<div style="background:{BORDER};border-radius:2px;height:4px;width:60px;">'
            f'<div style="background:{bar_c};height:100%;width:{score}%;border-radius:2px;"></div>'
            f'</div>'
            f'<span style="font-size:{FONT_LABEL};color:{bar_c};font-weight:600;">{score}</span>'
            f'</div>'
        )
        unreal_c  = GAIN if unreal >= 0 else LOSS
        unreal_str = f"{unreal:+.1f}%"
        trim_note  = f"Trim {trim_pct}%" if trim_pct > 0 else ""

        # Primary sell reason or hold reason
        primary_reason = reasons_sell[0] if reasons_sell else (reasons_hold[0] if reasons_hold else "No signal")

        rows += (
            f'<tr>'
            f'<td {td}>{_sym(sym)}</td>'
            f'<td {td}>{_action_badge(rec)}</td>'
            f'<td {td}>{bar_html}</td>'
            f'<td {td}><span style="color:{unreal_c};font-weight:700;">{unreal_str}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{pw:.0f}%</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{primary_reason}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{NEURAL};">{trim_note}</span></td>'
            f'</tr>'
        )

    act_count = sum(1 for a in analyses if a["recommendation"] != "HOLD")
    note = f"{act_count} need attention · stop-loss 8%" if act_count else f"{n} positions — all holding"
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Symbol</th><th {TH}>Signal</th><th {TH}>Score</th>'
        f'<th {TH}>P&amp;L</th><th {TH}>Weight</th><th {TH}>Top Reason</th><th {TH}>Action</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    return f'<div class="nt nt-wrap">{_section("📉","Sell Analysis",note)}{table}</div>'


# ── PANEL 5: Position Sizing — called internally by render_decision_center ────
@safe_render("Position Sizing")
def render_position_sizing_panel() -> str:
    d        = get_data()
    open_pos = d.get("open_pos", {})

    if not open_pos:
        return f'<div class="nt nt-wrap">{_section("📐","Position Sizing","Conviction-based target allocations")}{_card(_empty_state("💰", "Fully in cash", "Sizing guidance runs once the bot holds positions."))}</div>'

    sizings = []
    for sym in open_pos:
        sz = get_position_sizing(sym, d)
        sz["symbol"] = sym
        sizings.append(sz)
    sizings.sort(key=lambda s: abs(s["delta_weight"]), reverse=True)

    n    = len(sizings)
    rows = ""
    for i, sz in enumerate(sizings):
        sym     = sz["symbol"]
        cur_w   = sz["current_weight"]
        tgt_w   = sz["target_weight"]
        delta_w = sz["delta_weight"]
        dol_disp = sz["dollar_display"]
        reason  = sz["reason"]
        action  = sz["action"]
        td = TD if i < n - 1 else TD0

        act_c = GAIN if action == "add" else (LOSS if action == "reduce" else TEXT2)
        delta_str = f"{delta_w:+.1f}%"
        delta_c   = GAIN if delta_w > 0 else (LOSS if delta_w < 0 else TEXT2)

        # Weight bar showing current vs target
        bar_max = max(tgt_w, cur_w, 5.0)
        cur_bar_w = int(cur_w / bar_max * 100)
        tgt_bar_w = int(tgt_w / bar_max * 100)
        bar_html = (
            f'<div style="position:relative;width:80px;height:6px;background:{BORDER};border-radius:3px;">'
            f'<div style="position:absolute;left:0;top:0;height:100%;width:{cur_bar_w}%;'
            f'background:{TEXT2};border-radius:3px;"></div>'
            f'<div style="position:absolute;left:0;top:0;height:100%;width:{tgt_bar_w}%;'
            f'background:{act_c};opacity:.4;border-radius:3px;"></div>'
            f'</div>'
        )

        rows += (
            f'<tr>'
            f'<td {td}>{_sym(sym)}</td>'
            f'<td {td}><span style="font-family:Courier New,monospace;color:{TEXT1};">{cur_w:.1f}%</span></td>'
            f'<td {td}><span style="font-family:Courier New,monospace;color:{act_c};">{tgt_w:.1f}%</span></td>'
            f'<td {td}>{bar_html}</td>'
            f'<td {td}><span style="font-weight:700;color:{delta_c};">{delta_str}</span></td>'
            f'<td {td}><span style="font-family:Courier New,monospace;color:{act_c};">{dol_disp}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{reason}</span></td>'
            f'</tr>'
        )

    note = f"{n} positions · conviction-weighted · max 25% single stock"
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Symbol</th><th {TH}>Current</th><th {TH}>Target</th>'
        f'<th {TH}>Bar</th><th {TH}>Delta</th><th {TH}>Amount</th><th {TH}>Reason</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    return f'<div class="nt nt-wrap">{_section("📐","Position Sizing",note)}{table}</div>'

