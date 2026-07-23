"""Sticky Executive Summary Card — visible above all tabs (req 3)."""
from __future__ import annotations

import datetime
from loguru import logger

from dashboard.design_system import (
    BG, BORDER, TEXT1, TEXT2, TEXT3,
    GAIN, LOSS, NEURAL,
    FONT_SECTION, FONT_VALUE, FONT_LABEL, WEIGHT_BOLD,
)
from dashboard.data import get_data, safe_query, _market_status
from bot.core.error_logger import safe_render, timed

_logger = logger


def _today_pnl() -> tuple[float, float]:
    """Return (dollar_delta, pct_delta) vs yesterday's portfolio snapshot."""
    today = str(datetime.date.today())
    yest = str(datetime.date.today() - datetime.timedelta(days=1))
    try:
        t_rows = safe_query(
            "SELECT portfolio_value FROM portfolio_snapshots "
            "WHERE date(timestamp) = ? ORDER BY timestamp DESC LIMIT 1",
            (today,), default=[],
        )
        y_rows = safe_query(
            "SELECT portfolio_value FROM portfolio_snapshots "
            "WHERE date(timestamp) = ? ORDER BY timestamp DESC LIMIT 1",
            (yest,), default=[],
        )
        tv = float(t_rows[0][0]) if t_rows else None
        yv = float(y_rows[0][0]) if y_rows else None
        if tv and yv and yv > 0:
            return tv - yv, (tv - yv) / yv * 100
    except Exception:
        pass
    return 0.0, 0.0


@timed(_logger)
@safe_render("Executive Summary")
def render_executive_summary() -> str:
    """One-line capital + health status card — rendered once above gr.Tabs()."""
    d = get_data()
    mkt_label, mkt_color = _market_status()

    try:
        pv = float(d.get("portfolio", "0").replace("$", "").replace(",", ""))
    except Exception:
        pv = 0.0

    delta, delta_pct = _today_pnl()
    delta_color = GAIN if delta >= 0 else LOSS
    sign = "+" if delta >= 0 else ""
    delta_str = (
        f'{sign}${abs(delta):,.2f} ({delta_pct:+.2f}%)'
        if (delta or delta_pct) else ""
    )

    # Health score
    health_score = 0
    try:
        from dashboard.builders import build_health_vm
        vm = build_health_vm()
        health_score = int(vm.total)
    except Exception:
        pass
    health_color = GAIN if health_score >= 80 else (NEURAL if health_score >= 60 else LOSS)

    # Session state
    session_state = "UNKNOWN"
    try:
        from scheduler.session_manager import get_today_session
        session = get_today_session()
        session_state = session.state if session else "UNKNOWN"
    except Exception:
        pass

    # Last cron cycle
    last_cycle = "&mdash;"
    try:
        from scheduler.health_monitor import get_health_summary
        from zoneinfo import ZoneInfo
        health = get_health_summary()
        if health and health.last_execution_time:
            et = health.last_execution_time.astimezone(ZoneInfo("America/New_York"))
            last_cycle = et.strftime("%I:%M %p ET")
    except Exception:
        pass

    line1 = (
        f'<div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;">'
        f'<span style="font-size:{FONT_VALUE};font-weight:{WEIGHT_BOLD};color:{TEXT2};">'
        f'TradeGenius Capital</span>'
        f'<span style="font-size:{FONT_SECTION};font-weight:800;color:{TEXT1};">'
        f'${pv:,.2f}</span>'
    )
    if delta_str:
        line1 += (
            f'<span style="font-size:{FONT_VALUE};font-weight:700;color:{delta_color};">'
            f'{delta_str} today</span>'
        )
    line1 += f'</div>'

    line2 = (
        f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:3px;">'
        f'<span style="font-size:{FONT_LABEL};color:{health_color};">'
        f'Health: {health_score}/100</span>'
        f'<span style="font-size:{FONT_LABEL};color:{TEXT3};">·</span>'
        f'<span style="font-size:{FONT_LABEL};color:{mkt_color};">Session: {session_state}</span>'
        f'<span style="font-size:{FONT_LABEL};color:{TEXT3};">·</span>'
        f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">Last cycle: {last_cycle}</span>'
        f'</div>'
    )

    return (
        f'<div class="nt-exec-summary" style="background:{BG};'
        f'border-bottom:1px solid {BORDER};padding:10px 18px;'
        f'margin-bottom:4px;">'
        f'{line1}{line2}'
        f'</div>'
    )


from dashboard.registry import ComponentSpec, RefreshGroup, register
register(ComponentSpec("exec_summary_out", RefreshGroup.FAST, render_executive_summary, priority=10))
