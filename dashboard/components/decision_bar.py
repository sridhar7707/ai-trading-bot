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
    """Fetch today's actions (pending first, then executed), fall back to sell-analysis."""
    today = str(datetime.date.today())
    try:
        rows = safe_query(
            "SELECT action_type, symbol, reasoning, confidence, estimated_minutes, "
            "expected_impact, recommended_time, status "
            "FROM daily_actions WHERE session_date = ? "
            "ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, confidence DESC, "
            "created_at DESC LIMIT 5",
            (today,), default=None,
        )
        if rows:
            return [
                {
                    "action_type": r[0], "symbol": r[1] or "",
                    "reasoning": r[2] or "", "confidence": int(r[3] or 0),
                    "minutes": int(r[4] or 2),
                    "expected_impact": r[5] or "",
                    "recommended_time": r[6] or "Today",
                    "status": r[7] or "pending",
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
            "expected_impact": "", "recommended_time": "Today",
            "status": "pending",
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
        pending = sum(1 for a in actions if a.get("status") == "pending")
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

    primary = actions[0]
    rest    = actions[1:]
    p_color = _TYPE_COLOR.get(primary["action_type"].lower(), NEURAL)
    p_label = f"{primary['action_type'].upper()} {primary['symbol']}".strip()
    executed = primary.get("status") == "executed"

    # Spotlight card for the #1 action
    impact_html = (
        f'<div style="font-size:{FONT_LABEL};color:{TEXT3};margin-top:4px;">'
        f'Expected impact: <strong style="color:{TEXT1};">{primary["expected_impact"]}</strong></div>'
        if primary.get("expected_impact") else ""
    )
    time_html = (
        f'<span style="font-size:{FONT_LABEL};color:{TEXT3};">Recommended: '
        f'<strong style="color:{TEXT1};">{primary["recommended_time"]}</strong></span>'
    )
    conf_html = (
        f'<span style="font-size:{FONT_LABEL};color:{p_color};font-weight:{WEIGHT_BOLD};">'
        f'Confidence: {primary["confidence"]}%</span>'
        if primary["confidence"] else ""
    )
    exec_badge = (
        f'<span style="font-size:{FONT_LABEL};color:{NEURAL};background:{NEURAL}22;'
        f'border:1px solid {NEURAL};border-radius:10px;padding:1px 8px;">✓ Executed</span>'
        if executed else ""
    )
    spotlight = (
        f'<div style="border-left:3px solid {p_color};padding:10px 14px;'
        f'background:{p_color}11;border-radius:0 8px 8px 0;margin-bottom:10px;">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
        f'<span style="font-size:{FONT_SECTION};font-weight:800;color:{p_color};">{p_label}</span>'
        f'{conf_html}{exec_badge}</div>'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">{primary["reasoning"]}</div>'
        f'{impact_html}'
        f'<div style="display:flex;gap:16px;margin-top:6px;">{time_html}</div>'
        f'</div>'
    )

    # Remaining actions as chips
    chips_html = ""
    for a in rest[:4]:
        color = _TYPE_COLOR.get(a["action_type"].lower(), NEURAL)
        label = f"{a['action_type'].capitalize()} {a['symbol']}".strip()
        chips_html += (
            f'<span style="display:inline-flex;align-items:center;'
            f'padding:5px 14px;background:{color}22;border:1px solid {color};'
            f'border-radius:20px;font-size:{FONT_LABEL};font-weight:700;color:{color};'
            f'white-space:nowrap;">{label}</span>'
        )

    chips_row = (
        f'<div style="display:flex;flex-wrap:wrap;gap:8px;">{chips_html}</div>'
        if chips_html else ""
    )
    header = (
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:center;margin-bottom:10px;">'
        f'<span style="font-size:{FONT_SECTION};font-weight:800;color:{TEXT1};">'
        f"Today's Decision</span>"
        f'<span style="font-size:{FONT_LABEL};color:{TEXT3};">'
        f'{len(actions)} action{"s" if len(actions) != 1 else ""} today</span>'
        f'</div>'
    )

    return (
        f'<div class="nt nt-wrap">'
        f'<div class="nt-card" style="padding:16px 18px;">'
        f'{header}{spotlight}{chips_row}'
        f'</div></div>'
    )


from dashboard.registry import ComponentSpec, RefreshGroup, register
register(ComponentSpec("decision_bar_out", RefreshGroup.FAST, render_decision_bar, priority=11))
