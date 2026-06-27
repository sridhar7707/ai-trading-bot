"""AI Decision Log — recommendations vs executed trades (14-day window)."""
from __future__ import annotations
from loguru import logger
from dashboard.design_system import (
    GAIN, TEXT2, TEXT3,
    ACTION_SELL,
    FONT_LABEL,
    _card, _action_badge, _symbol, _confidence_bar,
    _empty_state, _section, _wrap,
    TH, TD, TD0,
)
from dashboard.data import safe_query
from bot.core.error_logger import safe_render

_logger = logger


@safe_render("Recommendation History")
def render_recommendation_history() -> str:
    recs = safe_query("""
        SELECT r.symbol, r.prediction_date, r.recommendation, r.confidence,
               r.prev_recommendation, r.price_at_recommendation,
               sl.regime, sl.ensemble_score
        FROM recommendations r
        LEFT JOIN (
            SELECT symbol, date(timestamp) AS d,
                   regime, ensemble_score,
                   ROW_NUMBER() OVER (PARTITION BY symbol, date(timestamp) ORDER BY id DESC) AS rn
            FROM signal_log
        ) sl ON sl.symbol = r.symbol AND sl.d = r.prediction_date AND sl.rn = 1
        WHERE r.prediction_date >= date('now', '-14 days')
        ORDER BY r.prediction_date DESC, r.symbol ASC
        LIMIT 200
    """, default=[])

    executed = set()
    exec_rows = safe_query("""
        SELECT symbol, date(timestamp), price FROM trades
        WHERE action = 'BUY' AND date(timestamp) >= date('now', '-14 days')
    """, default=[])
    executed_price: dict = {}
    for sym, dt, price in (exec_rows or []):
        executed.add((sym, str(dt)[:10]))
        executed_price[(sym, str(dt)[:10])] = price

    if not recs:
        return (
            f'<div class="nt nt-wrap">'
            f'{_section("📋", "Signal History", "")}'
            f'{_card(_empty_state("📋", "No history yet", "Signals are recorded every cycle. Check back after the next bot run."))}'
            f'</div>'
        )

    n         = len(recs)
    n_changes = sum(
        1 for r in recs
        if r[4] and r[4] != r[2]
    )
    note = f"{n} signals · {n_changes} changes" if n_changes else f"{n} signals"

    _regime_map = {
        "bullish": "Rising market", "bull": "Rising market",
        "bearish": "Falling market", "bear": "Falling market",
        "ranging": "Sideways market", "range": "Sideways market",
        "volatile": "Volatile market",
    }

    rows = ""
    for i, row in enumerate(recs):
        symbol, pred_date, rec, conf, prev, price_at_rec, regime, sl_score = row
        is_last = (i == n - 1)
        td      = TD0 if is_last else TD
        rec     = str(rec or "—")
        conf    = float(conf or sl_score or 0)
        changed = bool(prev and prev != rec)
        date_str = str(pred_date)[:10]
        key = (symbol, date_str)

        change_html = ""
        if changed:
            change_html = (
                f' <span style="font-size:{FONT_LABEL};color:{ACTION_SELL};">'
                f'was {prev}</span>'
            )

        regime_clean = str(regime or "").lower().replace("_", " ").strip()
        market_ctx = _regime_map.get(regime_clean, regime_clean.title() if regime_clean else "")

        if rec == "BUY" and key in executed:
            exec_p = executed_price.get(key, price_at_rec)
            why = (f'<span style="color:{GAIN};font-weight:600;">Bought</span>'
                   + (f'<span style="color:{TEXT2};"> at ${exec_p:,.2f}</span>' if exec_p else ""))
        elif rec == "BUY":
            why = (f'<span style="color:{TEXT3};">Held back</span>'
                   + (f'<span style="color:{TEXT2};"> &middot; {market_ctx}</span>' if market_ctx else ""))
        else:
            why = f'<span style="color:{TEXT2};">{market_ctx}</span>' if market_ctx else "—"

        rows += (
            f'<tr>'
            f'<td {td}>{_symbol(str(symbol))}</td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{date_str}</span></td>'
            f'<td {td}>{_action_badge(rec)}{change_html}</td>'
            f'<td {td}>{_confidence_bar(conf, show_label=False)}</td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};">{why}</span></td>'
            f'</tr>'
        )

    disclaimer = (
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};padding:8px 4px 12px;">'
        f'These are AI recommendations, not executed trades. A BUY signal means the AI wanted to buy — '
        f'but the bot may hold back if too many positions are open, cash is low, or daily loss limits are hit. '
        f'Check the <strong>Trades</strong> tab to see what was actually bought or sold.'
        f'</div>'
    )
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Symbol</th><th {TH}>Date</th>'
        f'<th {TH}>AI Decision</th><th {TH}>Conviction</th><th {TH}>Reason</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    return (
        f'<div class="nt nt-wrap">'
        f'{_section("📋", "AI Decision Log", note)}'
        f'{disclaimer}'
        f'{table}'
        f'</div>'
    )
