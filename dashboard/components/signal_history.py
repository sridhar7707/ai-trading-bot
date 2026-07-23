"""High-confidence signal tracker &mdash; entry / stop / target / outcome."""
from __future__ import annotations

from loguru import logger

from dashboard.data import safe_query, _to_ct
from dashboard.design_system import (
    BG, SURFACE, SURFACE2, BORDER,
    TEXT1, TEXT2, TEXT3,
    GAIN, LOSS, NEURAL,
    GAIN_BG, LOSS_BG, NEURAL_BG,
    GAIN_BD, LOSS_BD, NEURAL_BD,
    FONT_LABEL, FONT_VALUE, FONT_SECTION,
    WEIGHT_BOLD,
    _card, _empty_state, _section, _wrap, _sym, _stat_card,
    TH, TD, TD0,
)
from bot.core.error_logger import safe_render

_logger = logger

# ── Outcome colours and labels ─────────────────────────────────────────────────
_OUTCOME_STYLE: dict[str, tuple[str, str, str]] = {
    # outcome → (label, text-color, background)
    "target_hit": ("TARGET HIT ✓", GAIN,   GAIN_BG),
    "stop_hit":   ("STOP HIT ✗",   LOSS,   LOSS_BG),
    "expired":    ("EXPIRED",       TEXT2,  SURFACE2),
    "pending":    ("PENDING",       NEURAL, NEURAL_BG),
}


def _outcome_badge(outcome: str) -> str:
    label, color, bg = _OUTCOME_STYLE.get(outcome, ("?", TEXT3, SURFACE2))
    bd = GAIN_BD if outcome == "target_hit" else (LOSS_BD if outcome == "stop_hit" else NEURAL_BD)
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
        f'background:{bg};border:1px solid {bd};color:{color};'
        f'font-size:{FONT_LABEL};font-weight:{WEIGHT_BOLD};letter-spacing:.4px;">'
        f'{label}</span>'
    )


def _setup_badge(setup: str) -> str:
    color = NEURAL if setup == "breakout" else GAIN
    label = setup.upper() if setup not in ("none", "") else "&mdash;"
    return (
        f'<span style="font-size:{FONT_LABEL};color:{color};'
        f'font-weight:{WEIGHT_BOLD};">{label}</span>'
    )


@safe_render("Signal History")
def render_signal_history() -> str:
    rows = safe_query("""
        SELECT id, timestamp, symbol, entry_price, stop_price, target_price,
               rr_ratio, setup_type, xgb_prob, lstm_prob, ensemble_score,
               macro_score, outcome, outcome_price, outcome_pct
        FROM signal_history
        ORDER BY timestamp DESC
        LIMIT 50
    """, default=[])

    if not rows:
        es = _empty_state(
            "🎯",
            "No high-confidence signals yet",
            "Signals appear when all 7 gates pass: XGB >65%, LSTM >55%, "
            "volume 1.5×, technical setup, SPY positive, R:R ≥2:1, macro neutral+.",
        )
        return (
            f'<div class="nt nt-wrap">'
            f'{_section("🎯", "High-Confidence Signals", "all 7 gates must pass")}'
            f'{_card(es)}</div>'
        )

    # ── Stats ──────────────────────────────────────────────────────────────────
    total   = len(rows)
    wins    = sum(1 for r in rows if r[12] == "target_hit")
    losses  = sum(1 for r in rows if r[12] == "stop_hit")
    pending = sum(1 for r in rows if r[12] == "pending")
    resolved = wins + losses
    win_rate = (wins / resolved * 100) if resolved > 0 else 0.0

    avg_rr = sum(r[6] or 0 for r in rows) / total if total > 0 else 0.0

    wr_c  = GAIN if win_rate >= 65 else (NEURAL if win_rate >= 50 else LOSS)
    wr_bk = GAIN_BG if win_rate >= 65 else (NEURAL_BG if win_rate >= 50 else LOSS_BG)

    stats = (
        f'<div class="nt-cards">'
        + _stat_card("Signals Sent",  str(total),
                     TEXT2, TEXT1, "high-confidence only", 0.00)
        + _stat_card("Win Rate",      f"{win_rate:.0f}%",
                     TEXT2, wr_c, f"{wins}W / {losses}L / {pending} pending", 0.06)
        + _stat_card("Avg R:R",       f"{avg_rr:.1f} : 1",
                     TEXT2, GAIN, "risk vs reward ratio", 0.12)
        + _stat_card("Pending",       str(pending),
                     TEXT2, NEURAL, "awaiting resolution", 0.18)
        + "</div>"
    )

    # ── Table ──────────────────────────────────────────────────────────────────
    table_rows = ""
    for i, row in enumerate(rows):
        (sid, ts, symbol, entry, stop, target, rr,
         setup, xgb, lstm, ens, macro, outcome,
         out_price, out_pct) = row

        entry  = float(entry  or 0)
        stop   = float(stop   or 0)
        target = float(target or 0)
        rr     = float(rr     or 0)
        xgb    = float(xgb    or 0)
        lstm   = float(lstm   or 0)
        out_pct = float(out_pct or 0)

        stop_pct   = (stop   - entry) / entry * 100 if entry > 0 else 0
        target_pct = (target - entry) / entry * 100 if entry > 0 else 0

        td = TD if i < len(rows) - 1 else TD0
        anim = f'style="animation:slideInRow .3s ease both;animation-delay:{i*0.03:.2f}s;"'

        pnl_html = ""
        if outcome != "pending" and out_pct != 0:
            pnl_c = GAIN if out_pct >= 0 else LOSS
            pnl_html = (
                f'<span style="font-weight:700;color:{pnl_c};">'
                f'{out_pct:+.1%}</span>'
            )

        table_rows += (
            f'<tr {anim}>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{_to_ct(ts)}</span></td>'
            f'<td {td}>{_sym(symbol)}</td>'
            f'<td {td}>{_setup_badge(setup or "")}</td>'
            f'<td {td}>'
            f'<span style="font-family:Courier New,monospace;color:{TEXT1};">${entry:.2f}</span>'
            f'</td>'
            f'<td {td}>'
            f'<span style="font-family:Courier New,monospace;color:{LOSS};">${stop:.2f} '
            f'<span style="font-size:{FONT_LABEL};">({stop_pct:+.1f}%)</span></span>'
            f'</td>'
            f'<td {td}>'
            f'<span style="font-family:Courier New,monospace;color:{GAIN};">${target:.2f} '
            f'<span style="font-size:{FONT_LABEL};">({target_pct:+.1f}%)</span></span>'
            f'</td>'
            f'<td {td}><span style="color:{TEXT1};font-weight:700;">{rr:.1f}:1</span></td>'
            f'<td {td}>'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">'
            f'XGB {xgb*100:.0f}% · LSTM {lstm*100:.0f}%</span>'
            f'</td>'
            f'<td {td}>{_outcome_badge(outcome or "pending")}</td>'
            f'<td {td}>{pnl_html}</td>'
            f'</tr>'
        )

    help_block = (
        f'<div style="background:{BG};border-top:1px solid {BORDER};'
        f'padding:8px 14px;font-size:{FONT_LABEL};color:{TEXT2};line-height:1.8;">'
        f'<b>Entry</b> = price when signal fired &nbsp;·&nbsp; '
        f'<b>Stop</b> = exit if thesis wrong (−4%) &nbsp;·&nbsp; '
        f'<b>Target</b> = profit objective (+8%) &nbsp;·&nbsp; '
        f'<b>R:R</b> = reward ÷ risk &nbsp;·&nbsp; '
        f'<b>Outcome</b> auto-resolved when price hits stop or target'
        f'</div>'
    )

    table_inner = (
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Time (CT)</th>'
        f'<th {TH}>Symbol</th>'
        f'<th {TH}>Setup</th>'
        f'<th {TH}>Entry</th>'
        f'<th {TH}>Stop</th>'
        f'<th {TH}>Target</th>'
        f'<th {TH}>R:R</th>'
        f'<th {TH}>Models</th>'
        f'<th {TH}>Outcome</th>'
        f'<th {TH}>P&L</th>'
        f'</tr></thead><tbody>{table_rows}</tbody></table>'
        + help_block
    )

    note = f"{total} signals · {win_rate:.0f}% win rate · auto-resolved on price"
    return (
        f'<div class="nt nt-wrap">'
        f'{_section("🎯", "High-Confidence Signals", note)}'
        f'{stats}'
        f'{_wrap(table_inner)}'
        f'</div>'
    )


from dashboard.registry import ComponentSpec, RefreshGroup, register
register(ComponentSpec("signal_history_out", RefreshGroup.SLOW, render_signal_history, priority=34))
