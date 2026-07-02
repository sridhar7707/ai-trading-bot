"""Portfolio Simulator — read-only 'what if I buy more' preview (req 11.8)."""
from __future__ import annotations
import datetime
from loguru import logger
from dashboard.design_system import (
    SURFACE, SURFACE2, BORDER, TEXT1, TEXT2, TEXT3,
    GAIN, LOSS, NEURAL, PRIMARY,
    FONT_SECTION, FONT_VALUE, FONT_LABEL, WEIGHT_BOLD,
    _section, _wrap, _card, _metric_row, _empty_state,
    TH, TD, TD0,
)
from dashboard.data import get_data
from bot.core.error_logger import safe_render, timed
from database.user_settings import get_setting
from config import SECTOR_MAP as _SECTOR_MAP

_logger = logger


def _sector(sym: str) -> str:
    return _SECTOR_MAP.get(sym.upper(), "Other")


def _portfolio_stats(open_pos: dict, prices: dict, pv: float) -> dict[str, float]:
    """Return sector allocations + volatility proxy from current positions."""
    sectors: dict[str, float] = {}
    for sym, pos in open_pos.items():
        cur = prices.get(sym, 0.0)
        val = pos["shares"] * cur if cur > 0 else pos["invested"]
        pct = val / pv * 100 if pv > 0 else 0.0
        sec = _sector(sym)
        sectors[sec] = sectors.get(sec, 0.0) + pct
    return sectors


def simulate_buy(symbol: str, dollar_amount: float, d: dict) -> dict:
    """
    Pure calculation — NO DB writes.
    Returns 'before' vs 'after' metrics for the proposed buy.
    """
    open_pos = d.get("open_pos", {})
    prices   = d.get("prices", {})
    pv = 0.0
    try:
        raw = d.get("portfolio", "0") or "0"
        if raw.startswith("$"):
            pv = float(raw.replace("$", "").replace(",", ""))
    except (ValueError, AttributeError):
        pass
    cash = d.get("cash", 0.0)
    cur_price = prices.get(symbol, 0.0)

    # Before
    before_cash = cash
    before_weight: dict[str, float] = {}
    for sym, pos in open_pos.items():
        cur = prices.get(sym, 0.0)
        val = pos["shares"] * cur if cur > 0 else pos["invested"]
        before_weight[sym] = val / pv * 100 if pv > 0 else 0.0
    before_sectors = _portfolio_stats(open_pos, prices, pv)

    # After
    after_cash  = max(0.0, cash - dollar_amount)
    after_pv    = pv - dollar_amount + dollar_amount    # pv unchanged (cash → equity)
    new_shares  = dollar_amount / cur_price if cur_price > 0 else 0.0
    after_weight: dict[str, float] = dict(before_weight)
    existing_val = (open_pos[symbol]["shares"] * cur_price
                    if symbol in open_pos and cur_price > 0 else 0.0)
    after_weight[symbol] = (existing_val + dollar_amount) / pv * 100 if pv > 0 else 0.0
    after_sectors = dict(before_sectors)
    sec = _sector(symbol)
    after_sectors[sec] = after_sectors.get(sec, 0.0) + (dollar_amount / pv * 100 if pv > 0 else 0.0)

    # Health score: use real engine for 'before'; estimate delta for 'after'
    max_pos_limit = float(get_setting("max_position_pct", "0.20")) * 100
    max_sec_limit = 25.0
    try:
        from bot.core.recommendation_engine import get_portfolio_health
        before_health = int(get_portfolio_health(d).get("total", 0))
    except Exception:
        before_health = 0
    delta = 0
    for w in after_weight.values():
        if w > max_pos_limit:
            delta -= 5
    for w in before_weight.values():
        if w > max_pos_limit:
            delta += 5
    for _sec, w in after_sectors.items():
        if w > max_sec_limit:
            delta -= 3
    for _sec, w in before_sectors.items():
        if w > max_sec_limit:
            delta += 3
    after_health = max(0, min(100, before_health + delta))

    warnings: list[str] = []
    sym_after_w = after_weight.get(symbol, 0.0)
    if sym_after_w > max_pos_limit:
        warnings.append(f"{symbol} weight {sym_after_w:.0f}% exceeds {max_pos_limit:.0f}% limit")
    sec_after_w = after_sectors.get(sec, 0.0)
    if sec_after_w > max_sec_limit:
        warnings.append(f"{sec} sector {sec_after_w:.0f}% exceeds {max_sec_limit:.0f}% limit")
    if after_cash < pv * 0.05:
        warnings.append("Cash would drop below 5% — reduced buying power")

    return {
        "symbol":          symbol,
        "dollar_amount":   dollar_amount,
        "cur_price":       cur_price,
        "new_shares":      new_shares,
        "before_cash":     before_cash,
        "after_cash":      after_cash,
        "before_sym_wt":   before_weight.get(symbol, 0.0),
        "after_sym_wt":    sym_after_w,
        "before_sec_wt":   before_sectors.get(sec, 0.0),
        "after_sec_wt":    sec_after_w,
        "sec":             sec,
        "before_health":   before_health,
        "after_health":    after_health,
        "warnings":        warnings,
    }


def _row(label: str, before: str, after: str, warn: bool = False) -> str:
    after_c = LOSS if warn else TEXT1
    return (
        f'<tr>'
        f'<td style="padding:8px 14px;border-bottom:1px solid {BORDER};'
        f'font-size:{FONT_LABEL};color:{TEXT2};">{label}</td>'
        f'<td style="padding:8px 14px;border-bottom:1px solid {BORDER};'
        f'font-size:{FONT_VALUE};color:{TEXT1};font-weight:{WEIGHT_BOLD};">{before}</td>'
        f'<td style="padding:8px 14px;border-bottom:1px solid {BORDER};'
        f'font-size:{FONT_VALUE};color:{after_c};font-weight:{WEIGHT_BOLD};">{after}</td>'
        f'</tr>'
    )


@timed(_logger)
@safe_render("Portfolio Simulator")
def render_portfolio_simulator(symbol: str | None = None,
                                dollar_amount: float = 500.0) -> str:
    d = get_data()
    open_pos = d.get("open_pos", {})
    prices   = d.get("prices", {})

    section_html = _section("🔬", "Portfolio Simulator", "Read-only — no trades placed")

    if not symbol:
        syms = sorted(prices.keys()) or ["(no symbols available)"]
        sym_list = ", ".join(syms[:12])
        return (
            f'<div class="nt nt-wrap">{section_html}'
            f'{_card(_empty_state("🔬", "Select a symbol to simulate", f"Available: {sym_list}"))}'
            f'</div>'
        )

    result = simulate_buy(symbol, dollar_amount, d)
    warnings = result["warnings"]

    warn_html = ""
    if warnings:
        w_items = "".join(
            f'<div style="color:{LOSS};font-size:{FONT_LABEL};padding:3px 0;">'
            f'⚠ {w}</div>' for w in warnings
        )
        warn_html = (
            f'<div style="background:{LOSS}11;border:1px solid {LOSS}44;border-radius:6px;'
            f'padding:10px 14px;margin-top:8px;">{w_items}</div>'
        )

    bh = result["before_health"]
    ah = result["after_health"]
    health_delta = ah - bh
    health_delta_str = (f'+{health_delta}' if health_delta >= 0 else str(health_delta))
    health_after_c = GAIN if ah >= 80 else (NEURAL if ah >= 60 else LOSS)

    table = (
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr>'
        f'<th style="padding:8px 14px;background:{SURFACE2};color:{TEXT2};font-size:{FONT_LABEL};'
        f'text-transform:uppercase;letter-spacing:.6px;text-align:left;border-bottom:2px solid {BORDER};">'
        f'Metric</th>'
        f'<th style="padding:8px 14px;background:{SURFACE2};color:{TEXT2};font-size:{FONT_LABEL};'
        f'text-transform:uppercase;letter-spacing:.6px;border-bottom:2px solid {BORDER};">Before</th>'
        f'<th style="padding:8px 14px;background:{SURFACE2};color:{TEXT2};font-size:{FONT_LABEL};'
        f'text-transform:uppercase;letter-spacing:.6px;border-bottom:2px solid {BORDER};">After</th>'
        f'</tr></thead><tbody>'
        + _row("Cash Remaining",
               f'${result["before_cash"]:,.0f}',
               f'${result["after_cash"]:,.0f}',
               warn=result["after_cash"] < result["before_cash"] * 0.1)
        + _row(f'{symbol} Weight',
               f'{result["before_sym_wt"]:.1f}%',
               f'{result["after_sym_wt"]:.1f}%',
               warn=bool(result["after_sym_wt"] > 20))
        + _row(f'{result["sec"]} Sector',
               f'{result["before_sec_wt"]:.1f}%',
               f'{result["after_sec_wt"]:.1f}%',
               warn=bool(result["after_sec_wt"] > 25))
        + _row("Health Score",
               f'{bh}/100',
               f'<span style="color:{health_after_c};">{ah}/100 ({health_delta_str})</span>',
               warn=health_delta < 0)
        + f'</tbody></table>'
    )

    buy_note = (
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};padding:8px 14px;'
        f'border-top:1px solid {BORDER};background:{SURFACE2};">'
        f'Buying ${dollar_amount:,.0f} of {symbol} @ ${result["cur_price"]:.2f} '
        f'= {result["new_shares"]:.2f} shares &nbsp;·&nbsp; '
        f'<span style="color:{TEXT3};">Simulation only — no trade placed</span>'
        f'</div>'
    )
    card_body = table + buy_note + warn_html
    card = _wrap(card_body)
    return f'<div class="nt nt-wrap">{section_html}{card}</div>'
