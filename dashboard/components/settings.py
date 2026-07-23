"""Settings component — active risk settings summary + AI Memory / Investor Profile (req 11.10)."""
from __future__ import annotations
import datetime
import json
import os
from loguru import logger
from dashboard.design_system import (
    SURFACE, SURFACE2, BORDER, TEXT1, TEXT2, TEXT3,
    PRIMARY, GAIN, LOSS, NEURAL,
    FONT_SECTION, FONT_VALUE, FONT_LABEL, WEIGHT_BOLD,
    CARD_PADDING, CARD_RADIUS,
    _section, _wrap,
)
from database.user_settings import get_all_settings
from dashboard.data import get_data, get_db_conn, DB_PATH, safe_query
from bot.core.error_logger import safe_render, timed, log_exception
import sqlite3

_logger = logger
_MIN_TRADES_FOR_INSIGHTS = 20


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


def _compute_investor_profile(d: dict) -> dict | None:
    """Compute behavioral metrics from trades history. Returns None if < 20 closed trades."""
    if not os.path.exists(DB_PATH):
        return None
    try:
        with get_db_conn() as con:
            sells = con.execute(
                "SELECT symbol, price, timestamp, pnl_pct, holding_days "
                "FROM trades WHERE action LIKE 'SELL%' AND action != 'SELL_RECONCILE'"
            ).fetchall()
    except Exception as exc:
        log_exception(_logger, "_compute_investor_profile", exc)
        return None

    if len(sells) < _MIN_TRADES_FOR_INSIGHTS:
        return {"not_enough": True, "count": len(sells)}

    pnl_vals     = [float(r[3] or 0) for r in sells]
    hold_vals    = [float(r[4] or 0) for r in sells if r[4] is not None]
    wins         = [p for p in pnl_vals if p > 0]
    losses       = [p for p in pnl_vals if p < 0]
    avg_hold     = sum(hold_vals) / len(hold_vals) if hold_vals else 0.0
    win_rate     = len(wins) / len(pnl_vals) * 100 if pnl_vals else 0.0
    avg_win      = sum(wins) / len(wins) if wins else 0.0
    avg_loss     = abs(sum(losses) / len(losses)) if losses else 0.0

    # Early-exit rate: trades closed at < 50% of average win
    early_exits  = sum(1 for p in wins if p < avg_win * 0.5) if wins else 0
    early_exit_r = early_exits / len(wins) * 100 if wins else 0.0

    insights: list[str] = []
    adaptations: list[str] = []
    if early_exit_r > 30:
        insights.append(f"You tend to sell winners too early (avg exit at +{avg_win:.0f}%)")
        adaptations.append("Raising take-profit targets on winners by 20%")
    if win_rate > 60:
        insights.append(f"Strong win rate ({win_rate:.0f}%) — strategy is performing well")
    if avg_loss > avg_win * 1.5:
        insights.append(f"Losses ({avg_loss:.0f}%) run larger than wins ({avg_win:.0f}%) — review stop-loss")
        adaptations.append("Tightening stop-loss on lower-confidence entries")
    if avg_hold < 2:
        insights.append(f"Average hold time is {avg_hold:.1f} days — consider longer holds for larger moves")
    elif avg_hold > 30:
        insights.append(f"Average hold time {avg_hold:.0f} days — positions held longer than target (1–7 days)")

    return {
        "not_enough":      False,
        "trade_count":     len(pnl_vals),
        "win_rate":        round(win_rate, 1),
        "avg_hold_days":   round(avg_hold, 1),
        "avg_win_pct":     round(avg_win, 2),
        "avg_loss_pct":    round(avg_loss, 2),
        "early_exit_rate": round(early_exit_r, 1),
        "insights":        insights[:5],
        "adaptations":     adaptations[:3],
    }


@timed(_logger)
@safe_render("AI Memory")
def render_investor_profile() -> str:
    """AI Memory / Investor Profile panel for Settings tab."""
    d = get_data()
    profile = _compute_investor_profile(d)

    section_html = _section("🧠", "AI Memory", "Investor Profile &amp; Behavioral Insights")

    if profile is None:
        return (
            f'<div class="nt nt-wrap">{section_html}'
            f'<div style="background:{SURFACE};border:1px solid {BORDER};'
            f'border-radius:8px;padding:18px;color:{TEXT2};font-size:{FONT_LABEL};">'
            f'No trade data available yet.</div></div>'
        )

    if profile.get("not_enough"):
        n = profile["count"]
        remaining = _MIN_TRADES_FOR_INSIGHTS - n
        return (
            f'<div class="nt nt-wrap">{section_html}'
            f'<div style="background:{SURFACE};border:1px solid {BORDER};'
            f'border-radius:8px;padding:18px;">'
            f'<div style="color:{TEXT2};font-size:{FONT_LABEL};margin-bottom:8px;">'
            f'{n} completed trade{"s" if n != 1 else ""} recorded. '
            f'Need {remaining} more before behavioral insights are available.</div>'
            f'<div style="background:{BORDER};border-radius:3px;height:6px;">'
            f'<div style="background:{PRIMARY};height:100%;width:{int(n/_MIN_TRADES_FOR_INSIGHTS*100)}%;'
            f'border-radius:3px;"></div></div></div></div>'
        )

    # Stats row
    def _stat(label: str, val: str, color: str = TEXT1) -> str:
        return (
            f'<div style="text-align:center;padding:12px 10px;background:{SURFACE};'
            f'border-radius:8px;border:1px solid {BORDER};">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
            f'letter-spacing:.6px;margin-bottom:4px;">{label}</div>'
            f'<div style="font-size:{FONT_VALUE};font-weight:700;color:{color};">{val}</div>'
            f'</div>'
        )

    wr_c = GAIN if profile["win_rate"] >= 60 else (NEURAL if profile["win_rate"] >= 45 else LOSS)
    stats = (
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));'
        f'gap:8px;margin-bottom:14px;">'
        + _stat("Trades", str(profile["trade_count"]))
        + _stat("Win Rate", f'{profile["win_rate"]:.0f}%', wr_c)
        + _stat("Avg Hold", f'{profile["avg_hold_days"]:.1f}d')
        + _stat("Avg Win", f'+{profile["avg_win_pct"]:.1f}%', GAIN)
        + _stat("Avg Loss", f'-{profile["avg_loss_pct"]:.1f}%', LOSS)
        + f'</div>'
    )

    # Behavioral insights
    def _insight(text: str, icon: str = "⚠", color: str = "#f59e0b") -> str:
        return (
            f'<div style="display:flex;gap:8px;padding:6px 0;border-bottom:1px solid {BORDER};">'
            f'<span style="color:{color};font-weight:700;flex-shrink:0;">{icon}</span>'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">{text}</span>'
            f'</div>'
        )

    insights_html = ""
    for item in profile["insights"]:
        icon  = "✓" if any(w in item.lower() for w in ("strong", "well", "outperform")) else "⚠"
        color = GAIN if icon == "✓" else "#f59e0b"
        insights_html += _insight(item, icon, color)

    adaptations_html = ""
    for item in profile["adaptations"]:
        adaptations_html += _insight(f"→ {item}", "→", PRIMARY)

    body = stats
    if insights_html:
        body += (
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};font-weight:700;'
            f'text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px;">'
            f'Behavioral Insights</div>'
            + insights_html
        )
    if adaptations_html:
        body += (
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};font-weight:700;'
            f'text-transform:uppercase;letter-spacing:.6px;margin:10px 0 6px;">'
            f'AI Adaptations</div>'
            + adaptations_html
        )

    body += (
        f'<div style="font-size:{FONT_LABEL};color:{TEXT3};margin-top:10px;">'
        f'Profile recalculates after every 5 completed trades. '
        f'Insights are factual &mdash; they show your data, not judgments.</div>'
    )

    card = (
        f'<div style="background:{SURFACE};border:1px solid {BORDER};'
        f'border-radius:8px;padding:18px;">{body}</div>'
    )
    return f'<div class="nt nt-wrap">{section_html}{card}</div>'


from dashboard.registry import ComponentSpec, RefreshGroup, register
register(ComponentSpec("settings_summary_out", RefreshGroup.FAST, render_settings_summary, priority=80))
register(ComponentSpec("investor_profile_out", RefreshGroup.FAST, render_investor_profile, priority=81))
