"""Capital Management tab — managed pool, health score, ledger, profit handling (Phase 3)."""
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
    # Try portfolio snapshots, then trades (earliest recorded value)
    for sql in (
        "SELECT portfolio_value FROM portfolio_snapshots WHERE portfolio_value > 0 ORDER BY timestamp ASC LIMIT 1",
        "SELECT portfolio_value FROM trades WHERE portfolio_value > 0 ORDER BY id ASC LIMIT 1",
    ):
        try:
            row = safe_query(sql, default=[])
            if row:
                return float(row[0][0])
        except Exception:
            pass
    # Last resort: use the live Alpaca account value from the dashboard cache
    try:
        pv_str = get_data().get("portfolio", "")
        if pv_str:
            return float(pv_str.replace("$", "").replace(",", ""))
    except Exception:
        pass
    return _DEFAULT_DEPOSIT


def _load_pool():
    """Return the active CapitalPool with auto-detected initial amount. Returns None on failure."""
    try:
        from dashboard.data import get_db_conn
        from bot.capital.pool import load_active_pool
        initial = _initial_deposit()
        with get_db_conn() as con:
            return load_active_pool(con, initial_amount=initial)
    except Exception as exc:
        _logger.debug(f"_load_pool: {exc}")
        return None


def _reinvest_on() -> bool:
    return get_setting("reinvest_profits_only", "false") != "true"


def _capital_stats() -> dict:
    d = get_data()
    try:
        pv = float(d.get("portfolio", "0").replace("$", "").replace(",", ""))
    except Exception:
        pv = 0.0
    initial = _initial_deposit()
    ai_profit = pv - initial
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
    best_sym, best_pnl = "&mdash;", 0.0
    worst_sym, worst_pnl = "&mdash;", 0.0
    try:
        _base = "SELECT symbol, pnl_pct * notional / 100.0 AS abs_pnl FROM trades WHERE action LIKE 'SELL%' AND pnl_pct IS NOT NULL ORDER BY abs_pnl "
        rows = safe_query(_base + "DESC LIMIT 1", default=[])
        if rows:
            best_sym, best_pnl = rows[0][0], float(rows[0][1] or 0)
        rows = safe_query(_base + "ASC LIMIT 1", default=[])
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


def _pnl_colored(val: float) -> tuple[str, str]:
    color = GAIN if val >= 0 else LOSS
    sign = "+" if val >= 0 else ""
    return f"{sign}${abs(val):,.2f}", color


@timed(_logger)
@safe_render("Capital Overview")
def render_capital_overview() -> str:
    s = _capital_stats()
    profit_str, profit_color = _pnl_colored(s["ai_profit"])

    def _big_card(label: str, val: str, sub: str, val_color: str = TEXT1) -> str:
        return (
            f'<div style="flex:1;text-align:center;padding:24px 16px;'
            f'background:{SURFACE};border-radius:10px;border:1px solid {BORDER};min-width:160px;">'
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
        f'${s["pv"]:,.2f}</div></div>'
    )
    return (
        f'<div class="nt nt-wrap">'
        f'{_section("💰", "TradeGenius Capital Fund", "Initial deposit + AI-generated profits")}'
        f'<div class="nt-card" style="padding:18px;">{two_cards}{total_card}</div>'
        f'</div>'
    )


@timed(_logger)
def render_capital_chart() -> go.Figure:
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
                x=dates, y=vals, name="TradeGenius Capital",
                line=dict(color=GAIN, width=2),
                fill="tozeroy", fillcolor=f"{GAIN}22",
            ))
    except Exception:
        pass
    layout.update({"title": "", "xaxis_title": "", "yaxis_title": "Portfolio Value ($)", "showlegend": False})
    fig.update_layout(**layout)
    return fig


@timed(_logger)
@safe_render("Profit Breakdown")
def render_profit_breakdown() -> str:
    s = _capital_stats()

    def _row(label: str, val: float, border: bool = True) -> str:
        val_str, color = _pnl_colored(val)
        sep = f"border-bottom:1px solid {BORDER};" if border else ""
        return (
            f'<div style="display:flex;justify-content:space-between;padding:8px 0;{sep}">'
            f'<span style="font-size:{FONT_VALUE};color:{TEXT2};">{label}</span>'
            f'<span style="font-size:{FONT_VALUE};font-weight:700;color:{color};">{val_str}</span>'
            f'</div>'
        )

    total_str, total_color = _pnl_colored(s["ai_profit"])
    best_str, best_color = _pnl_colored(s["best_pnl"])
    worst_str, worst_color = _pnl_colored(s["worst_pnl"])
    breakdown = (
        _row("Realized Profit (closed trades)", s["realized"])
        + _row("Unrealized Profit (open positions)", s["unrealized"], border=False)
        + f'<div style="display:flex;justify-content:space-between;padding:10px 0;'
        f'border-top:2px solid {BORDER};margin-top:4px;">'
        f'<span style="font-size:{FONT_VALUE};font-weight:{WEIGHT_BOLD};color:{TEXT1};">Total</span>'
        f'<span style="font-size:{FONT_VALUE};font-weight:800;color:{total_color};">{total_str}</span></div>'
    )
    extras = (
        f'<div style="margin-top:4px;padding-top:10px;border-top:1px solid {BORDER};">'
        f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
        f'<span style="font-size:{FONT_LABEL};color:{TEXT3};">Best trade</span>'
        f'<span style="font-size:{FONT_LABEL};color:{best_color};">{s["best_sym"]}  {best_str}</span></div>'
        f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
        f'<span style="font-size:{FONT_LABEL};color:{TEXT3};">Worst trade</span>'
        f'<span style="font-size:{FONT_LABEL};color:{worst_color};">{s["worst_sym"]}  {worst_str}</span>'
        f'</div></div>'
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
    pool = _load_pool()
    if pool is None:
        return ""

    reinvest_on = _reinvest_on()
    withdrawable = 0.0 if reinvest_on else max(0.0, pool.realized_profit)
    buying_power = pool.tradeable_cash if reinvest_on else max(0.0, pool.tradeable_cash - withdrawable)

    def _row(label: str, val: float, color: str = TEXT2, indent: bool = False,
             border: bool = True, bold: bool = False) -> str:
        pad = "padding-left:16px;" if indent else ""
        sep = f"border-bottom:1px solid {BORDER};" if border else ""
        fw = f"font-weight:800;" if bold else f"font-weight:700;"
        return (
            f'<div style="display:flex;justify-content:space-between;padding:7px 0;{sep}{pad}">'
            f'<span style="font-size:{FONT_VALUE};color:{TEXT2};">{label}</span>'
            f'<span style="font-size:{FONT_VALUE};{fw}color:{color};">'
            f'${val:,.2f}</span></div>'
        )

    managed_color = GAIN if pool.total_value >= pool.allocated_amount else LOSS
    buying_color  = GAIN if buying_power > 0 else TEXT3

    rows = (
        _row("Managed Capital", pool.total_value, managed_color, bold=True)
        + _row("Cash", pool.available_cash, TEXT1, indent=True)
        + _row("Invested", pool.invested_amount,
               TEXT1 if pool.invested_amount > 0 else TEXT3, indent=True)
        + _row("Reserve", pool.reserve, TEXT3, indent=True, border=not reinvest_on)
    )
    if not reinvest_on:
        profit_color = GAIN if withdrawable > 0 else TEXT3
        rows += (
            _row("Profit (withdrawable)", withdrawable, profit_color, indent=True)
        )

    rows += (
        f'<div style="display:flex;justify-content:space-between;'
        f'padding:9px 0;border-top:2px solid {BORDER};margin-top:4px;">'
        f'<span style="font-size:{FONT_VALUE};font-weight:{WEIGHT_BOLD};color:{TEXT1};">'
        f'Buying Power</span>'
        f'<span style="font-size:{FONT_VALUE};font-weight:800;color:{buying_color};">'
        f'${buying_power:,.2f}</span></div>'
    )

    return (
        f'<div class="nt nt-wrap">'
        f'{_section("🏦", "Capital Pool", f"Pool: {pool.name}")}'
        f'<div class="nt-card" style="padding:14px 18px;">{rows}</div>'
        f'</div>'
    )


@timed(_logger)
@safe_render("Capital Health")
def render_capital_health() -> str:
    pool = _load_pool()
    d    = get_data()
    open_pos = d.get("open_pos", {})
    prices   = d.get("prices", {})

    score = 0
    checks: list[tuple[bool, str]] = []
    managed = pool.total_value if pool else 0.0
    avail   = pool.available_cash if pool else 0.0

    cash_pct = avail / managed if managed > 0 else 0.0
    if cash_pct >= 0.30:
        score += 25; checks.append((True,  "Cash Reserve Healthy"))
    elif cash_pct >= 0.15:
        score += 15; checks.append((True,  "Cash Reserve Adequate"))
    else:
        score += 5;  checks.append((False, "Cash Reserve Low"))

    n = len(open_pos)
    if 2 <= n <= 5:
        score += 25; checks.append((True,  "Diversification Good"))
    elif n == 1:
        score += 15; checks.append((False, "Concentrated — only 1 position"))
    elif n == 0:
        score += 15; checks.append((True,  "No open positions"))
    else:
        score += 15; checks.append((True,  f"Diversified ({n} positions)"))

    if managed > 0 and open_pos:
        pos_pcts = {
            sym: (pos["shares"] * prices.get(sym, 0.0) if prices.get(sym) else pos["invested"]) / managed
            for sym, pos in open_pos.items()
        }
        max_sym = max(pos_pcts, key=pos_pcts.get)
        max_pct = pos_pcts[max_sym]
        if max_pct <= 0.20:
            score += 25; checks.append((True,  "Position Sizes Safe"))
        elif max_pct <= 0.30:
            score += 15; checks.append((False, f"Slightly Overweight {max_sym} ({max_pct:.0%})"))
        else:
            score += 5;  checks.append((False, f"Overweight {max_sym} ({max_pct:.0%}) — trim"))
    else:
        score += 25; checks.append((True, "Position Sizes Safe"))

    realized = pool.realized_profit if pool else 0.0
    alloc    = pool.allocated_amount if pool else 1.0
    dd_pct   = realized / alloc if alloc > 0 else 0.0
    if dd_pct >= 0.0:
        score += 25; checks.append((True,  "Pool Profitable"))
    elif dd_pct >= -0.03:
        score += 15; checks.append((True,  "Pool Near Break-even"))
    else:
        score += 5;  checks.append((False, f"Pool Down {abs(dd_pct):.1%}"))

    icon_color = GAIN if score >= 75 else (NEURAL if score >= 50 else LOSS)
    header = (
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">'
        f'<span style="font-size:{FONT_SECTION};font-weight:800;color:{TEXT1};">Capital Health</span>'
        f'<span style="font-size:{FONT_HERO};font-weight:800;color:{icon_color};">'
        f'{score} <span style="font-size:{FONT_SECTION};color:{TEXT3};">/ 100</span></span></div>'
    )
    items = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;">'
        f'<span style="color:{GAIN if ok else NEURAL};">{"✓" if ok else "⚠"}</span>'
        f'<span style="font-size:{FONT_VALUE};color:{TEXT2};">{label}</span></div>'
        for ok, label in checks
    )
    return (
        f'<div class="nt nt-wrap">'
        f'{_section("❤️", "Capital Health", "Pool-level risk check")}'
        f'<div class="nt-card" style="padding:14px 18px;">{header}{items}</div></div>'
    )


@timed(_logger)
@safe_render("Capital Ledger")
def render_capital_ledger() -> str:
    try:
        rows = safe_query(
            "SELECT event_type, amount, balance_after, symbol, notes, created_at "
            "FROM capital_ledger ORDER BY id DESC LIMIT 20",
            default=[],
        )
    except Exception as exc:
        _logger.debug(f"render_capital_ledger: {exc}")
        return ""
    if not rows:
        return (
            f'<div class="nt nt-wrap">'
            f'{_section("📒", "Capital Ledger", "No events yet")}'
            f'</div>'
        )

    _LABELS = {
        "deposit": "Deposit", "withdrawal": "Withdrawal",
        "buy": "Buy", "sell": "Sell",
    }

    def _ev_row(ev_type: str, amount: float, balance: float,
                symbol: str | None, notes: str | None, ts: str) -> str:
        label = _LABELS.get(ev_type, ev_type.title())
        if symbol and ev_type in ("buy", "sell"):
            label = f"{label} {symbol}"
        elif notes and ev_type == "deposit" and "initial" in (notes or "").lower():
            label = "Initial Allocation"
        color = GAIN if amount >= 0 else LOSS
        sign  = "+" if amount >= 0 else ""
        date  = ts[:10] if ts else ""
        return (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:6px 0;border-bottom:1px solid {BORDER};">'
            f'<div>'
            f'<span style="font-size:{FONT_VALUE};color:{TEXT1};">{label}</span>'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT3};margin-left:8px;">{date}</span>'
            f'</div>'
            f'<div style="text-align:right;">'
            f'<span style="font-size:{FONT_VALUE};font-weight:700;color:{color};">'
            f'{sign}${abs(amount):,.2f}</span>'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT3};margin-left:8px;">'
            f'bal ${balance:,.2f}</span>'
            f'</div></div>'
        )

    events = "".join(_ev_row(*r) for r in rows)
    return (
        f'<div class="nt nt-wrap">'
        f'{_section("📒", "Capital Ledger", "Append-only transaction log")}'
        f'<div class="nt-card" style="padding:14px 18px;">{events}</div>'
        f'</div>'
    )


def save_reinvestment_mode(mode: str) -> str:
    profits_only = "off" in mode.lower()   # "Auto Reinvest OFF" → protect profits
    value = "true" if profits_only else "false"
    ok = save_setting("reinvest_profits_only", value)
    if ok:
        desc = (
            "Profit stays protected — not reinvested automatically"
            if profits_only
            else "Profit is reinvested — grows your managed capital"
        )
        return f'<span style="color:{GAIN};font-size:12px;">&#10003; {desc}</span>'
    return f'<span style="color:{LOSS};font-size:12px;">⚠ Save failed</span>'


def do_pool_deposit(amount_str: str) -> str:
    """Append a deposit event to the active pool. Returns status HTML."""
    try:
        amount = float(amount_str)
        if amount <= 0:
            raise ValueError("amount must be positive")
    except Exception:
        return f'<span style="color:{LOSS};font-size:12px;">⚠ Invalid amount</span>'
    try:
        from dashboard.data import get_db_conn
        from bot.capital.pool import load_active_pool, deposit as _deposit
        initial = _initial_deposit()
        with get_db_conn() as con:
            pool = load_active_pool(con, initial_amount=initial)
            _deposit(con, pool.id, amount, notes="Dashboard deposit")
        return f'<span style="color:{GAIN};font-size:12px;">&#10003; Deposited ${amount:,.2f}</span>'
    except Exception as exc:
        _logger.warning(f"do_pool_deposit: {exc}")
        return f'<span style="color:{LOSS};font-size:12px;">⚠ {exc}</span>'


def do_pool_withdraw(amount_str: str) -> str:
    """Append a withdrawal event to the active pool. Returns status HTML."""
    try:
        amount = float(amount_str)
        if amount <= 0:
            raise ValueError("amount must be positive")
    except Exception:
        return f'<span style="color:{LOSS};font-size:12px;">⚠ Invalid amount</span>'
    try:
        from dashboard.data import get_db_conn
        from bot.capital.pool import load_active_pool, withdraw as _withdraw
        initial = _initial_deposit()
        with get_db_conn() as con:
            pool = load_active_pool(con, initial_amount=initial)
            if amount > pool.available_cash:
                return (
                    f'<span style="color:{LOSS};font-size:12px;">'
                    f'⚠ Exceeds available cash (${pool.available_cash:,.2f})</span>'
                )
            _withdraw(con, pool.id, amount, notes="Dashboard withdrawal")
        return f'<span style="color:{GAIN};font-size:12px;">&#10003; Withdrew ${amount:,.2f}</span>'
    except Exception as exc:
        _logger.warning(f"do_pool_withdraw: {exc}")
        return f'<span style="color:{LOSS};font-size:12px;">⚠ {exc}</span>'


def do_set_reserve(amount_str: str) -> str:
    """Set the cash reserve floor. Returns status HTML."""
    try:
        amount = float(amount_str)
        if amount < 0:
            raise ValueError("reserve cannot be negative")
    except Exception:
        return f'<span style="color:{LOSS};font-size:12px;">⚠ Invalid amount</span>'
    try:
        from dashboard.data import get_db_conn
        from bot.capital.pool import load_active_pool, set_reserve as _set_reserve
        initial = _initial_deposit()
        with get_db_conn() as con:
            pool = load_active_pool(con, initial_amount=initial)
            _set_reserve(con, pool.id, amount)
        return f'<span style="color:{GAIN};font-size:12px;">&#10003; Reserve set to ${amount:,.2f}</span>'
    except Exception as exc:
        _logger.warning(f"do_set_reserve: {exc}")
        return f'<span style="color:{LOSS};font-size:12px;">⚠ {exc}</span>'


from dashboard.registry import ComponentSpec, RefreshGroup, register
register(ComponentSpec("capital_overview_out",  RefreshGroup.FAST, render_capital_overview,  priority=70))
register(ComponentSpec("profit_breakdown_out",  RefreshGroup.FAST, render_profit_breakdown,  priority=71))
register(ComponentSpec("managed_capital_out",   RefreshGroup.FAST, render_managed_capital,   priority=72))
register(ComponentSpec("capital_health_out",    RefreshGroup.FAST, render_capital_health,    priority=73))
register(ComponentSpec("capital_ledger_out",    RefreshGroup.FAST, render_capital_ledger,    priority=74))
register(ComponentSpec("capital_chart_out",     RefreshGroup.SLOW, render_capital_chart,     priority=50))
