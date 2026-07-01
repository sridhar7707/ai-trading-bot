"""Settings component — HTML summary of currently active risk settings."""
from __future__ import annotations

from dashboard.design_system import (
    SURFACE, BORDER, TEXT1, TEXT2, PRIMARY, CARD_PADDING, CARD_RADIUS,
)
from database.user_settings import get_all_settings


def render_settings_summary() -> str:
    """Return HTML card summarising the currently saved settings."""
    s = get_all_settings()
    try:
        max_pos = float(s.get("max_position_pct", "0.20")) * 100
        max_dd  = float(s.get("max_drawdown_pct", "0.12")) * 100
        stop    = float(s.get("stop_loss_pct",    "0.04")) * 100
    except (ValueError, TypeError):
        max_pos, max_dd, stop = 20.0, 12.0, 4.0

    notif = s.get("notifications_enabled", "false").lower() == "true"
    rows = [
        ("Risk Tolerance",    s.get("risk_tolerance", "Moderate")),
        ("Benchmark",         s.get("benchmark", "SPY")),
        ("Max Position Size", f"{max_pos:.0f}%"),
        ("Max Drawdown",      f"{max_dd:.0f}%"),
        ("Stop-Loss Default", f"{stop:.1f}%"),
        ("Notifications",     "On" if notif else "Off"),
    ]
    inner = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:7px 0;'
        f'border-bottom:1px solid {BORDER};">'
        f'<span style="color:{TEXT2};font-size:13px">{label}</span>'
        f'<span style="color:{TEXT1};font-weight:600;font-size:13px">{val}</span>'
        f'</div>'
        for label, val in rows
    )
    return (
        f'<div style="background:{SURFACE};border:1px solid {BORDER};'
        f'border-radius:{CARD_RADIUS};padding:{CARD_PADDING}">'
        f'<p style="color:{PRIMARY};font-weight:700;margin:0 0 10px;font-size:14px">'
        f'Active Settings</p>'
        f'{inner}'
        f'<p style="color:{TEXT2};font-size:11px;margin:10px 0 0">'
        f'Changes take effect on the next bot cycle (&sim;60 s).</p>'
        f'</div>'
    )
