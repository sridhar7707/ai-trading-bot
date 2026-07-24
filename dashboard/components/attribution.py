"""Portfolio Attribution — answers which holdings, trades, sectors, and models drove P&L."""
from __future__ import annotations

from collections import defaultdict

from loguru import logger

from bot.core.error_logger import safe_render, timed
from config import SECTOR_MAP
from dashboard.data import safe_query
from dashboard.design_system import (
    BORDER, GAIN, LOSS, NEURAL, SURFACE, TEXT1, TEXT2, TEXT3,
    FONT_HERO, FONT_LABEL, FONT_VALUE, WEIGHT_BOLD,
    TH, TD, TD0, _section, _empty_state, _card, _wrap,
)

_logger = logger

_CELL = f"padding:8px 14px;font-size:{FONT_VALUE};color:{TEXT2};"
_NUM  = f"padding:8px 14px;font-size:{FONT_VALUE};color:{TEXT1};text-align:right;font-family:Courier New,monospace;font-weight:{WEIGHT_BOLD};"


def _pnl_color(val: float) -> str:
    return GAIN if val > 0 else (LOSS if val < 0 else TEXT3)


def _pnl_str(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}${val:,.2f}"


def _query_closed_trades() -> list[tuple]:
    """Return (symbol, realized_pnl, pnl_pct, xgb_prob, lstm_prob, sentiment_score, macro_score, ensemble_score, timestamp) for all SELL rows."""
    return safe_query(
        "SELECT symbol, realized_pnl, pnl_pct, xgb_prob, lstm_prob, "
        "sentiment_score, macro_score, ensemble_score, timestamp "
        "FROM trades WHERE action LIKE 'SELL%' "
        "AND realized_pnl IS NOT NULL AND pnl_pct IS NOT NULL "
        "ORDER BY timestamp DESC",
        default=[],
    ) or []


# ── By Symbol ─────────────────────────────────────────────────────────────────

@timed(_logger)
@safe_render("Attribution by Symbol")
def render_attribution_by_symbol() -> str:
    rows = _query_closed_trades()
    if not rows:
        return _card(_empty_state("📊", "No closed trades yet", "Attribution builds as trades close."))

    agg: dict[str, dict] = {}
    for sym, pnl, pct, *_ in rows:
        if sym not in agg:
            agg[sym] = {"pnl": 0.0, "trades": 0, "wins": 0}
        agg[sym]["pnl"] += float(pnl or 0)
        agg[sym]["trades"] += 1
        if (pct or 0) > 0:
            agg[sym]["wins"] += 1

    ranked = sorted(agg.items(), key=lambda kv: kv[1]["pnl"], reverse=True)
    total_pnl = sum(v["pnl"] for v in agg.values())

    tbody = ""
    for i, (sym, v) in enumerate(ranked):
        td = TD if i < len(ranked) - 1 else TD0
        win_rate = v["wins"] / v["trades"] * 100 if v["trades"] else 0
        color = _pnl_color(v["pnl"])
        bar_pct = abs(v["pnl"]) / (max(abs(v2["pnl"]) for v2 in agg.values()) or 1) * 100
        bar_color = GAIN if v["pnl"] >= 0 else LOSS
        bar = (
            f'<div style="background:{BORDER};border-radius:2px;height:4px;width:60px;overflow:hidden;">'
            f'<div style="background:{bar_color};height:100%;width:{bar_pct:.0f}%;border-radius:2px;"></div></div>'
        )
        tbody += (
            f"<tr>"
            f"<td {td} style='{_CELL}'>{i+1}</td>"
            f"<td {td} style='{_CELL}font-weight:{WEIGHT_BOLD};color:{TEXT1};'>{sym}</td>"
            f"<td {td} style='{_NUM}color:{color};'>{_pnl_str(v['pnl'])}</td>"
            f"<td {td} style='{_CELL}'>{bar}</td>"
            f"<td {td} style='{_CELL}'>{win_rate:.0f}%</td>"
            f"<td {td} style='{_CELL}'>{v['trades']}</td>"
            f"</tr>"
        )

    total_color = _pnl_color(total_pnl)
    footer = (
        f'<tr style="background:{SURFACE};">'
        f'<td colspan="2" {TD0} style="{_CELL}font-weight:{WEIGHT_BOLD};color:{TEXT1};">Total</td>'
        f'<td {TD0} style="{_NUM}color:{total_color};">{_pnl_str(total_pnl)}</td>'
        f'<td colspan="3" {TD0}></td></tr>'
    )
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>#</th><th {TH}>Symbol</th><th {TH}>P&amp;L</th>'
        f'<th {TH}>Contribution</th><th {TH}>Win Rate</th><th {TH}>Trades</th>'
        f'</tr></thead><tbody>{tbody}{footer}</tbody></table>'
    )
    return f'<div class="nt nt-wrap">{_section("🏆", "By Holding", f"{len(ranked)} symbols · {_pnl_str(total_pnl)} net")}{table}</div>'


# ── By Sector ─────────────────────────────────────────────────────────────────

@timed(_logger)
@safe_render("Attribution by Sector")
def render_attribution_by_sector() -> str:
    rows = _query_closed_trades()
    if not rows:
        return _card(_empty_state("🏭", "No closed trades yet", "Sector attribution builds as trades close."))

    sectors: dict[str, dict] = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0})
    for sym, pnl, pct, *_ in rows:
        sector = SECTOR_MAP.get(sym, "Other")
        sectors[sector]["pnl"] += float(pnl or 0)
        sectors[sector]["trades"] += 1
        if (pct or 0) > 0:
            sectors[sector]["wins"] += 1

    ranked = sorted(sectors.items(), key=lambda kv: kv[1]["pnl"], reverse=True)
    total_pnl = sum(v["pnl"] for v in sectors.values())
    max_abs = max(abs(v["pnl"]) for v in sectors.values()) or 1

    tbody = ""
    for i, (sector, v) in enumerate(ranked):
        td = TD if i < len(ranked) - 1 else TD0
        color = _pnl_color(v["pnl"])
        bar_pct = abs(v["pnl"]) / max_abs * 100
        bar_color = GAIN if v["pnl"] >= 0 else LOSS
        bar = (
            f'<div style="background:{BORDER};border-radius:2px;height:4px;width:80px;overflow:hidden;">'
            f'<div style="background:{bar_color};height:100%;width:{bar_pct:.0f}%;border-radius:2px;"></div></div>'
        )
        share = v["pnl"] / total_pnl * 100 if total_pnl else 0
        tbody += (
            f"<tr>"
            f"<td {td} style='{_CELL}font-weight:{WEIGHT_BOLD};color:{TEXT1};'>{sector.replace('_', ' ')}</td>"
            f"<td {td} style='{_NUM}color:{color};'>{_pnl_str(v['pnl'])}</td>"
            f"<td {td} style='{_CELL}'>{bar}</td>"
            f"<td {td} style='{_CELL}color:{TEXT3};'>{share:+.0f}%</td>"
            f"<td {td} style='{_CELL}'>{v['trades']}</td>"
            f"</tr>"
        )

    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Sector</th><th {TH}>P&amp;L</th>'
        f'<th {TH}>Contribution</th><th {TH}>Share</th><th {TH}>Trades</th>'
        f'</tr></thead><tbody>{tbody}</tbody></table>'
    )
    return f'<div class="nt nt-wrap">{_section("🏭", "By Sector", f"{len(ranked)} sectors")}{table}</div>'


# ── By Model ──────────────────────────────────────────────────────────────────

@timed(_logger)
@safe_render("Attribution by Model")
def render_attribution_by_model() -> str:
    rows = [r for r in _query_closed_trades() if (r[3] or 0) > 0]  # xgb_prob > 0
    if not rows:
        return _card(_empty_state(
            "🤖", "Model attribution needs more data",
            "Scores are recorded from V4 onwards. Attribution grows as new trades close."
        ))

    wins  = [r for r in rows if (r[2] or 0) > 0]
    loses = [r for r in rows if (r[2] or 0) <= 0]

    def _avg(lst: list, idx: int) -> float:
        vals = [float(r[idx] or 0) for r in lst if r[idx]]
        return sum(vals) / len(vals) if vals else 0.0

    models = [
        ("XGBoost",   3, "🤖"),
        ("LSTM",      4, "🧠"),
        ("Sentiment", 5, "📰"),
        ("Macro",     6, "🌐"),
        ("Ensemble",  7, "⚡"),
    ]

    tbody = ""
    n = len(models)
    for i, (name, idx, icon) in enumerate(models):
        td = TD if i < n - 1 else TD0
        w_avg = _avg(wins, idx)
        l_avg = _avg(loses, idx)
        delta = w_avg - l_avg
        color = _pnl_color(delta)
        verdict = "Adds edge" if delta > 0.02 else ("Hurts" if delta < -0.02 else "Neutral")
        tbody += (
            f"<tr>"
            f"<td {td} style='{_CELL}'>{icon} {name}</td>"
            f"<td {td} style='{_CELL}'>{w_avg:.3f}</td>"
            f"<td {td} style='{_CELL}'>{l_avg:.3f}</td>"
            f"<td {td} style='{_NUM}color:{color};'>{delta:+.3f}</td>"
            f"<td {td} style='{_CELL}'>{verdict}</td>"
            f"</tr>"
        )

    note = f"{len(wins)}W / {len(loses)}L from {len(rows)} scored trades"
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Model</th><th {TH}>Avg score (wins)</th>'
        f'<th {TH}>Avg score (losses)</th><th {TH}>Delta</th><th {TH}>Verdict</th>'
        f'</tr></thead><tbody>{tbody}</tbody></table>'
    )
    return f'<div class="nt nt-wrap">{_section("🤖", "By Model", note)}{table}</div>'


# ── Best / Worst Trades ───────────────────────────────────────────────────────

@timed(_logger)
@safe_render("Attribution by Trade")
def render_attribution_by_trade() -> str:
    rows = _query_closed_trades()
    if not rows:
        return _card(_empty_state("📋", "No closed trades yet", "Individual trade breakdown appears here."))

    sorted_rows = sorted(rows, key=lambda r: float(r[1] or 0), reverse=True)
    top5    = sorted_rows[:5]
    bottom5 = list(reversed(sorted_rows[-5:]))

    def _trade_rows(trades: list, last_border: bool = True) -> str:
        html = ""
        n = len(trades)
        for i, r in enumerate(trades):
            sym, pnl, pct, *_, ts = r
            td = TD if (i < n - 1 or last_border) else TD0
            color = _pnl_color(float(pnl or 0))
            date = str(ts)[:10] if ts else "—"
            html += (
                f"<tr>"
                f"<td {td} style='{_CELL}font-weight:{WEIGHT_BOLD};color:{TEXT1};'>{sym}</td>"
                f"<td {td} style='{_NUM}color:{color};'>{_pnl_str(float(pnl or 0))}</td>"
                f"<td {td} style='{_CELL}color:{color};'>{float(pct or 0)*100:+.2f}%</td>"
                f"<td {td} style='{_CELL}color:{TEXT3};'>{date}</td>"
                f"</tr>"
            )
        return html

    header = f'<th {TH}>Symbol</th><th {TH}>P&amp;L</th><th {TH}>Return</th><th {TH}>Date</th>'
    best_table  = _wrap(f'<table class="nt-tbl"><thead><tr>{header}</tr></thead><tbody>{_trade_rows(top5, last_border=False)}</tbody></table>')
    worst_table = _wrap(f'<table class="nt-tbl"><thead><tr>{header}</tr></thead><tbody>{_trade_rows(bottom5, last_border=False)}</tbody></table>')

    return (
        f'<div class="nt nt-wrap">'
        f'{_section("📋", "Best Trades", "Top 5 by realized P&L")}{best_table}'
        f'{_section("📋", "Worst Trades", "Bottom 5 by realized P&L")}{worst_table}'
        f'</div>'
    )


from dashboard.registry import ComponentSpec, RefreshGroup, register
register(ComponentSpec("attribution_symbol_out", RefreshGroup.SLOW, render_attribution_by_symbol, priority=80))
register(ComponentSpec("attribution_sector_out", RefreshGroup.SLOW, render_attribution_by_sector, priority=81))
register(ComponentSpec("attribution_model_out",  RefreshGroup.SLOW, render_attribution_by_model,  priority=82))
register(ComponentSpec("attribution_trade_out",  RefreshGroup.SLOW, render_attribution_by_trade,  priority=83))
