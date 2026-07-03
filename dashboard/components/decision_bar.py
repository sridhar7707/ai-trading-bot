"""Decision Bar — today's AI action chips, always expanded (req 4)."""
from __future__ import annotations

import datetime
from loguru import logger

from dashboard.design_system import (
    SURFACE2, BORDER, TEXT1, TEXT2, TEXT3,
    GAIN, LOSS, NEURAL,
    FONT_SECTION, FONT_VALUE, FONT_LABEL, WEIGHT_BOLD,
)
from dashboard.data import get_data, safe_query, _market_status, _next_market_open
from bot.core.error_logger import safe_render, timed

_logger = logger

_TYPE_COLOR = {
    "sell": LOSS, "trim": LOSS, "exit": LOSS,
    "buy": GAIN, "add": GAIN,
    "review": NEURAL, "deploy_cash": NEURAL,
}


def _get_daily_actions() -> list[dict]:
    """Fetch pending actions from daily_actions, fall back to sell-analysis."""
    today = str(datetime.date.today())
    try:
        rows = safe_query(
            "SELECT action_type, symbol, reasoning, confidence, estimated_minutes "
            "FROM daily_actions WHERE session_date = ? AND status = 'pending' "
            "ORDER BY confidence DESC LIMIT 5",
            (today,), default=None,
        )
        if rows:
            return [
                {
                    "action_type": r[0], "symbol": r[1] or "",
                    "reasoning": r[2] or "", "confidence": int(r[3] or 0),
                    "minutes": int(r[4] or 2),
                }
                for r in rows
            ]
    except Exception:
        pass
    # Fallback: derive from sell analysis
    from dashboard.components.brief import _action_items
    items = _action_items()
    result = []
    for item in items[:5]:
        parts = item.split()
        atype = parts[0].lower() if parts else "review"
        sym = parts[1] if len(parts) > 1 else ""
        result.append({
            "action_type": atype, "symbol": sym,
            "reasoning": item, "confidence": 0, "minutes": 2,
        })
    return result


@timed(_logger)
@safe_render("Decision Bar")
def render_decision_bar() -> str:
    mkt_label, _ = _market_status()
    actions = _get_daily_actions()
    market_open = "open" in mkt_label.lower()

    if not market_open:
        next_open = _next_market_open()
        pending = len(actions)
        return (
            f'<div class="nt nt-wrap">'
            f'<div class="nt-card" style="padding:14px 18px;display:flex;'
            f'align-items:center;gap:14px;">'
            f'<span style="font-size:18px;">🌙</span>'
            f'<div>'
            f'<div style="font-size:{FONT_VALUE};color:{TEXT1};font-weight:{WEIGHT_BOLD};">'
            f'Market closed. Next session: {next_open}</div>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};margin-top:2px;">'
            f'Pending for tomorrow: {pending} action{"s" if pending != 1 else ""}'
            f'</div></div></div></div>'
        )

    if not actions:
        return (
            f'<div class="nt nt-wrap">'
            f'<div class="nt-card" style="padding:14px 18px;display:flex;'
            f'align-items:center;gap:10px;">'
            f'<span style="color:{GAIN};font-size:18px;">✓</span>'
            f'<span style="font-size:{FONT_VALUE};color:{TEXT1};">'
            f'No actions needed today. Portfolio is on track.</span>'
            f'</div></div>'
        )

    visible = actions[:3]
    overflow = len(actions) - 3

    chips_html = ""
    for a in visible:
        color = _TYPE_COLOR.get(a["action_type"].lower(), NEURAL)
        label = f"{a['action_type'].capitalize()} {a['symbol']}".strip()
        chips_html += (
            f'<span style="display:inline-flex;align-items:center;'
            f'padding:6px 16px;background:{color}22;border:1px solid {color};'
            f'border-radius:20px;font-size:{FONT_LABEL};font-weight:700;color:{color};'
            f'cursor:pointer;white-space:nowrap;">{label}</span>'
        )
    if overflow > 0:
        chips_html += (
            f'<span style="display:inline-flex;align-items:center;'
            f'padding:6px 14px;background:{SURFACE2};border:1px solid {BORDER};'
            f'border-radius:20px;font-size:{FONT_LABEL};color:{TEXT3};'
            f'white-space:nowrap;">+ {overflow} more</span>'
        )

    conf_actions = [a for a in actions if a["confidence"]]
    avg_conf = (
        sum(a["confidence"] for a in conf_actions) // len(conf_actions)
        if conf_actions else 0
    )
    total_mins = sum(a["minutes"] for a in actions)

    header = (
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:center;margin-bottom:10px;">'
        f'<span style="font-size:{FONT_SECTION};font-weight:800;color:{TEXT1};">'
        f"Today's Decisions</span>"
        f'<span style="font-size:{FONT_LABEL};color:{TEXT3};">'
        f'{len(actions)} action{"s" if len(actions) != 1 else ""} ›</span>'
        f'</div>'
    )
    chips_row = (
        f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;">'
        f'{chips_html}</div>'
    )
    footer = (
        f'<div style="display:flex;gap:20px;flex-wrap:wrap;'
        f'border-top:1px solid {BORDER};padding-top:8px;">'
        + (f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">Confidence: '
           f'<strong style="color:{TEXT1};">{avg_conf}%</strong></span>'
           if avg_conf else "")
        + f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">Est. time: '
        f'<strong style="color:{TEXT1};">{total_mins} min</strong></span>'
        f'<span style="font-size:{FONT_LABEL};color:{TEXT3};">All AI-generated ⓘ</span>'
        f'</div>'
    )

    return (
        f'<div class="nt nt-wrap">'
        f'<div class="nt-card" style="padding:16px 18px;">'
        f'{header}{chips_row}{footer}'
        f'</div></div>'
    )
