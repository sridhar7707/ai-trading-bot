"""Capital tab — initial deposit vs AI profit, growth chart, reinvestment toggle (req 6)."""
from __future__ import annotations

import datetime
from loguru import logger
import plotly.graph_objects as go

from dashboard.design_system import (
    BG, SURFACE, SURFACE2, BORDER, TEXT1, TEXT2, TEXT3,
    GAIN, LOSS, NEURAL, PRIMARY,
    FONT_HERO, FONT_SECTION, FONT_VALUE, FONT_LABEL, WEIGHT_BOLD,
    PLOTLY_LAYOUT, _section,
)
from dashboard.data import get_data, safe_query
from database.user_settings import get_setting, save_setting
from bot.core.error_logger import safe_render, timed

_logger = logger
_DEFAULT_DEPOSIT = 1000.0


def _initial_deposit() -> float:
    custom = get_setting("initial_deposit", None)
    if custom:
        try:
            return float(custom)
        except Exception:
            pass
    # Derive from the earliest recorded portfolio value so the figure is
    # accurate even when the user hasn't configured the setting explicitly.
    try:
        row = safe_query(
            "SELECT portfolio_value FROM portfolio_snapshots "
            "WHERE portfolio_value > 0 ORDER BY timestamp ASC LIMIT 1",
            default=[],
        )
        if row:
            return float(row[0][0])
        row = safe_query(
            "SELECT portfolio_value FROM trades "
            "WHERE portfolio_value > 0 ORDER BY id ASC LIMIT 1",
            default=[],
        )
        if row:
            return float(row[0][0])
    except Exception:
        pass
    return _DEFAULT_DEPOSIT


def _capital_stats() -> dict:
    d = get_data()
    try:
        pv = float(d.get("portfolio", "0").replace("$", "").replace(",", ""))
    except Exception:
        pv = 0.0

    initial = _initial_deposit()
    ai_profit = pv - initial

    # Realized P&L from closed trades
    realized = 0.0
    try:
        rows = safe_query(
            "SELECT SUM(pnl_pct * notional / 100.0) FROM trades "
            "WHERE action LIKE 'SELL%' AND pnl_pct IS NOT NULL AND notional IS NOT NULL",
            default=[],
        )
        if rows and rows[0][0] is not None:
            realized = float(rows[0][0])
    except Exception:
        pass

    unrealized = ai_profit - realized

    # Best / worst closed trade by absolute dollar P&L
    best_sym, best_pnl = "&mdash;", 0.0
    worst_sym, worst_pnl = "&mdash;", 0.0
    try:
        rows = safe_query(
            "SELECT symbol, pnl_pct * notional / 100.0 AS abs_pnl FROM trades "
            "WHERE action LIKE 'SELL%' AND pnl_pct IS NOT NULL "
            "ORDER BY abs_pnl DESC LIMIT 1",
            default=[],
        )
        if rows:
            best_sym, best_pnl = rows[0][0], float(rows[0][1] or 0)
        rows = safe_query(
            "SELECT symbol, pnl_pct * notional / 100.0 AS abs_pnl FROM trades "
            "WHERE action LIKE 'SELL%' AND pnl_pct IS NOT NULL "
            "ORDER BY abs_pnl ASC LIMIT 1",
            default=[],
        )
        if rows:
            worst_sym, worst_pnl = rows[0][0], float(rows[0][1] or 0)
    except Exception:
        pass

    return {
        "pv": pv, "initial": initial, "ai_profit": ai_profit,
        "realized": realized, "unrealized": unrealized,
        "best_sym": best_sym, "best_pnl": best_pnl,
        "worst_sym": worst_sym, "worst_pnl": worst_pnl,
    }


def _pnl_str(val: float) -> tuple[str, str]:
    color = GAIN if val >= 0 else LOSS
    sign = "+" if val >= 0 else ""
    return f"{sign}${abs(val):,.2f}", color


@timed(_logger)
@safe_render("Capital Overview")
def render_capital_overview() -> str:
    s = _capital_stats()
    profit_str, profit_color = _pnl_str(s["ai_profit"])

    def _big_card(label: str, val: str, sub: str, val_color: str = TEXT1) -> str:
        return (
            f'<div style="flex:1;text-align:center;padding:24px 16px;'
            f'background:{SURFACE};border-radius:10px;border:1px solid {BORDER};'
            f'min-width:160px;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT3};text-transform:uppercase;'
            f'letter-spacing:.8px;margin-bottom:8px;">{label}</div>'
            f'<div style="font-size:{FONT_HERO};font-weight:800;color:{val_color};">{val}</div>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT3};margin-top:6px;">{sub}</div>'
            f'</div>'
        )

    two_cards = (
        f'<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;">'
        + _big_card("Initial Deposit", f"${s['initial']:,.2f}", "your money", TEXT2)
        + _big_card("AI-Generated Profit", profit_str, "AI earned this", profit_color)
        + f'</div>'
    )

    total_card = (
        f'<div style="text-align:center;padding:18px;background:{SURFACE2};'
        f'border-radius:10px;border:1px solid {BORDER};">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT3};text-transform:uppercase;'
        f'letter-spacing:.8px;margin-bottom:6px;">Total Under Management</div>'
        f'<div style="font-size:{FONT_HERO};font-weight:800;color:{TEXT1};">'
        f'${s["pv"]:,.2f}</div>'
        f'</div>'
    )

    return (
        f'<div class="nt nt-wrap">'
        f'{_section("💰", "TradeGenius Capital Fund", "Initial deposit + AI-generated profits")}'
        f'<div class="nt-card" style="padding:18px;">{two_cards}{total_card}</div>'
        f'</div>'
    )


@timed(_logger)
def render_capital_chart() -> go.Figure:
    """Portfolio growth line chart from portfolio_snapshots."""
    fig = go.Figure()
    layout = dict(PLOTLY_LAYOUT)

    try:
        rows = safe_query(
            "SELECT timestamp, portfolio_value FROM portfolio_snapshots "
            "WHERE portfolio_value > 0 ORDER BY timestamp ASC",
            default=[],
        )
        if rows:
            dates = [str(r[0])[:10] for r in rows]
            vals = [float(r[1]) for r in rows]
            fig.add_trace(go.Scatter(
                x=dates, y=vals,
                name="TradeGenius Capital",
                line=dict(color=GAIN, width=2),
                fill="tozeroy",
                fillcolor=f"{GAIN}22",
            ))
    except Exception:
        pass

    layout.update({
        "title": "",
        "xaxis_title": "",
        "yaxis_title": "Portfolio Value ($)",
        "showlegend": False,
    })
    fig.update_layout(**layout)
    return fig


@timed(_logger)
@safe_render("Profit Breakdown")
def render_profit_breakdown() -> str:
    s = _capital_stats()

    def _row(label: str, val: float, border: bool = True) -> str:
        val_str, color = _pnl_str(val)
        sep = f"border-bottom:1px solid {BORDER};" if border else ""
        return (
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:8px 0;{sep}">'
            f'<span style="font-size:{FONT_VALUE};color:{TEXT2};">{label}</span>'
            f'<span style="font-size:{FONT_VALUE};font-weight:700;color:{color};">'
            f'{val_str}</span></div>'
        )

    total_str, total_color = _pnl_str(s["ai_profit"])
    best_str, best_color = _pnl_str(s["best_pnl"])
    worst_str, worst_color = _pnl_str(s["worst_pnl"])

    breakdown = (
        _row("Realized Profit (closed trades)", s["realized"])
        + _row("Unrealized Profit (open positions)", s["unrealized"], border=False)
        + f'<div style="display:flex;justify-content:space-between;'
        f'padding:10px 0;border-top:2px solid {BORDER};margin-top:4px;">'
        f'<span style="font-size:{FONT_VALUE};font-weight:{WEIGHT_BOLD};color:{TEXT1};">'
        f'Total</span>'
        f'<span style="font-size:{FONT_VALUE};font-weight:800;color:{total_color};">'
        f'{total_str}</span></div>'
    )

    extras = (
        f'<div style="margin-top:4px;padding-top:10px;border-top:1px solid {BORDER};">'
        f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
        f'<span style="font-size:{FONT_LABEL};color:{TEXT3};">Best trade</span>'
        f'<span style="font-size:{FONT_LABEL};color:{best_color};">'
        f'{s["best_sym"]}  {best_str}</span></div>'
        f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
        f'<span style="font-size:{FONT_LABEL};color:{TEXT3};">Worst trade</span>'
        f'<span style="font-size:{FONT_LABEL};color:{worst_color};">'
        f'{s["worst_sym"]}  {worst_str}</span></div>'
        f'</div>'
    )

    return (
        f'<div class="nt nt-wrap">'
        f'{_section("📊", "Profit Breakdown", "Realized vs. unrealized P&L")}'
        f'<div class="nt-card" style="padding:16px 18px;">{breakdown}{extras}</div>'
        f'</div>'
    )


@timed(_logger)
@safe_render("Managed Capital Pool")
def render_managed_capital() -> str:
    """Show the active CapitalPool breakdown — what the AI can actually trade."""
    try:
        from dashboard.data import get_db_conn
        from bot.capital.pool import load_active_pool
        with get_db_conn() as con:
            pool = load_active_pool(con)
    except Exception:
        return ""  # pool table not yet created or bot not initialized

    def _row(label: str, val: float, color: str = TEXT2, border: bool = True) -> str:
        sign = "+" if val > 0 else ""
        sep = f"border-bottom:1px solid {BORDER};" if border else ""
        return (
            f'<div style="display:flex;justify-content:space-between;padding:7px 0;{sep}">'
            f'<span style="font-size:{FONT_VALUE};color:{TEXT2};">{label}</span>'
            f'<span style="font-size:{FONT_VALUE};font-weight:700;color:{color};">'
            f'{sign}${abs(val):,.2f}</span></div>'
        )

    invested_color = TEXT1 if pool.invested_amount > 0 else TEXT3
    pnl_color = GAIN if pool.realized_profit >= 0 else LOSS
    avail_color = GAIN if pool.tradeable_cash > 0 else TEXT3
    rows = (
        _row("Allocated (total managed)", pool.allocated_amount, TEXT2)
        + _row("Available Cash", pool.available_cash, TEXT1)
        + _row("Reserve (held back)", pool.reserve, TEXT3)
        + _row("Tradeable Cash", pool.tradeable_cash, avail_color)
        + _row("Invested (open positions)", pool.invested_amount, invested_color)
        + _row("Realized Profit", pool.realized_profit, pnl_color, border=False)
    )
    total_color = GAIN if pool.total_value >= pool.allocated_amount else LOSS
    footer = (
        f'<div style="display:flex;justify-content:space-between;'
        f'padding:9px 0;border-top:2px solid {BORDER};margin-top:4px;">'
        f'<span style="font-size:{FONT_VALUE};font-weight:{WEIGHT_BOLD};color:{TEXT1};">'
        f'Total Pool Value</span>'
        f'<span style="font-size:{FONT_VALUE};font-weight:800;color:{total_color};">'
        f'${pool.total_value:,.2f}</span></div>'
    )
    return (
        f'<div class="nt nt-wrap">'
        f'{_section("🏦", "Managed Capital Pool", f"Pool: {pool.name}")}'
        f'<div class="nt-card" style="padding:14px 18px;">{rows}{footer}</div>'
        f'</div>'
    )


def save_reinvestment_mode(mode: str) -> str:
    """Persist reinvestment toggle selection; returns status HTML snippet."""
    profits_only = "profits only" in mode.lower()
    value = "true" if profits_only else "false"
    ok = save_setting("reinvest_profits_only", value)
    if ok:
        desc = (
            "Reinvest profits only &mdash; your initial deposit is always protected"
            if profits_only
            else "Reinvest everything &mdash; profits and initial deposit both grow the position"
        )
        return (
            f'<span style="color:{GAIN};font-size:12px;">'
            f'&#10003; Active: {desc}</span>'
        )
    return f'<span style="color:{LOSS};font-size:12px;">⚠ Save failed</span>'


from dashboard.registry import ComponentSpec, RefreshGroup, register
register(ComponentSpec("capital_overview_out",  RefreshGroup.FAST, render_capital_overview,  priority=70))
register(ComponentSpec("profit_breakdown_out",  RefreshGroup.FAST, render_profit_breakdown,  priority=71))
register(ComponentSpec("managed_capital_out",   RefreshGroup.FAST, render_managed_capital,   priority=72))
register(ComponentSpec("capital_chart_out",     RefreshGroup.SLOW, render_capital_chart,     priority=50))
