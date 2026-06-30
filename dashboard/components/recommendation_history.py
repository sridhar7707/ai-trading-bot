"""AI Decision Log — recommendations vs executed trades (14-day window)."""
from __future__ import annotations
import json
from loguru import logger
from dashboard.design_system import (
    BG, SURFACE, BORDER,
    GAIN, LOSS, NEURAL, TEXT1, TEXT2, TEXT3,
    ACTION_SELL,
    FONT_LABEL, FONT_VALUE, FONT_SECTION,
    _card, _action_badge, _symbol, _confidence_bar,
    _empty_state, _section, _wrap,
    _sym, _badge,
    TH, TD, TD0,
)
from dashboard.data import safe_query
from bot.core.error_logger import safe_render

_logger = logger

# Plain-English labels for end users — no jargon, no indicator names
_PLAIN_WHY: dict[str, str] = {
    "rsi":             "Momentum rising",
    "mfi":             "Buyers stepping in",
    "volume_ratio":    "Unusual trading volume",
    "obv_chg_pct":     "Net buying pressure",
    "vol_ratio_trend": "Volume accelerating",
    "bb_width":        "Volatility expanding",
    "atr_pct":         "Big moves expected",
    "bb_position":     "Price near top of range",
    "returns":         "Recent price strength",
    "hl_ratio":        "Wide intraday range",
    "vwap_dev":        "Trading above average price",
    "macd_diff_pct":   "Trend flipping upward",
    "ema_spread":      "Short-term trend above long-term",
    "ret_5d":          "Strong past 5 days",
    "ret_21d":         "Strong past month",
    "ret_63d":         "Strong past quarter",
    "ret_126d":        "Strong 6-month run",
    "mom_12_1":        "12-month momentum",
    "high_52w_pct":    "Near 52-week high",
    "gap_overnight":   "Gapped up at open",
    "rsi_divergence":  "Momentum accelerating",
    "macd_cross_up":   "MACD just flipped positive",
}


@safe_render("Recommendation History")
def render_recommendation_history() -> str:
    recs = safe_query("""
        SELECT r.symbol, r.prediction_date, r.recommendation, r.confidence,
               r.prev_recommendation, r.price_at_recommendation,
               sl.regime, sl.ensemble_score, sl.xgb_prob, sl.lstm_prob, sl.feature_drivers
        FROM recommendations r
        LEFT JOIN (
            SELECT symbol, date(timestamp) AS d,
                   regime, ensemble_score, xgb_prob, lstm_prob, feature_drivers,
                   ROW_NUMBER() OVER (PARTITION BY symbol, date(timestamp) ORDER BY id DESC) AS rn
            FROM signal_log
        ) sl ON sl.symbol = r.symbol AND sl.d = r.prediction_date AND sl.rn = 1
        WHERE r.prediction_date >= date('now', '-14 days')
        ORDER BY r.prediction_date DESC, r.symbol ASC
        LIMIT 200
    """, default=[])

    exec_rows = safe_query("""
        SELECT symbol, date(timestamp), price FROM trades
        WHERE action = 'BUY' AND date(timestamp) >= date('now', '-14 days')
    """, default=[])
    executed: set = set()
    executed_price: dict = {}
    for sym, dt, price in (exec_rows or []):
        executed.add((sym, str(dt)[:10]))
        executed_price[(sym, str(dt)[:10])] = price

    if not recs:
        return (
            f'<div class="nt nt-wrap">'
            f'{_section("📋", "AI Decision Log", "")}'
            f'{_card(_empty_state("📋", "No history yet", "Signals are recorded every cycle. Check back after the next bot run."))}'
            f'</div>'
        )

    n         = len(recs)
    n_changes = sum(1 for r in recs if r[4] and r[4] != r[2])
    note      = f"{n} signals · {n_changes} changes" if n_changes else f"{n} signals"

    _regime_map = {
        "bullish": "Rising market", "bull": "Rising market",
        "bearish": "Falling market", "bear": "Falling market",
        "ranging": "Sideways market", "range": "Sideways market",
        "volatile": "Volatile market",
    }

    def _sort_key(row):
        symbol, pred_date, rec = row[0], row[1], str(row[2] or "")
        date_str = str(pred_date)[:10]
        conf = float(row[3] or row[7] or 0)
        if rec == "BUY" and (symbol, date_str) in executed:
            return (0, -conf)   # executed buys first, highest confidence first
        if rec == "BUY":
            return (1, -conf)   # pending buys second
        return (2, -conf)       # holds / other last

    sorted_recs = sorted(recs, key=_sort_key)

    rows = ""
    for i, row in enumerate(sorted_recs):
        (symbol, pred_date, rec, conf, prev,
         price_at_rec, regime, sl_score,
         xgb_prob, lstm_prob, feature_drivers) = row

        is_last  = (i == n - 1)
        td       = TD0 if is_last else TD
        rec      = str(rec or "—")
        conf     = float(conf or sl_score or 0)
        xgb      = float(xgb_prob or 0)
        lstm     = float(lstm_prob or 0)
        changed  = bool(prev and prev != rec)
        date_str = str(pred_date)[:10]
        key      = (symbol, date_str)

        change_html = (
            f' <span style="font-size:{FONT_LABEL};color:{ACTION_SELL};">was {prev}</span>'
            if changed else ""
        )

        regime_clean = str(regime or "").lower().replace("_", " ").strip()
        market_ctx   = _regime_map.get(regime_clean, regime_clean.title() if regime_clean else "")

        # ── Priority dot ──────────────────────────────────────────────────────
        if rec == "BUY" and key in executed:
            dot = f'<span style="color:{GAIN};font-size:10px;margin-right:5px;" title="Executed">●</span>'
        elif rec == "BUY":
            dot = f'<span style="color:{NEURAL};font-size:10px;margin-right:5px;" title="Held back">●</span>'
        else:
            dot = f'<span style="color:{TEXT3};font-size:10px;margin-right:5px;" title="Hold/Watch">●</span>'

        # ── Models column ─────────────────────────────────────────────────────
        if xgb > 0 or lstm > 0:
            xgb_c  = GAIN  if xgb  >= 0.70 else (NEURAL if xgb  >= 0.55 else TEXT2)
            lstm_c = GAIN  if lstm >= 0.65 else (NEURAL if lstm >= 0.50 else TEXT2)
            models_html = (
                f'<span style="font-size:{FONT_LABEL};font-family:monospace;">'
                f'<span style="color:{xgb_c};">XGB {xgb*100:.0f}%</span>'
                f'<span style="color:{TEXT3};"> · </span>'
                f'<span style="color:{lstm_c};">LSTM {lstm*100:.0f}%</span>'
                f'</span>'
            )
        else:
            models_html = f'<span style="color:{TEXT3};font-size:{FONT_LABEL};">—</span>'

        # ── Feature drivers (plain English) ──────────────────────────────────
        driver_parts: list[str] = []
        try:
            ds = json.loads(feature_drivers) if isinstance(feature_drivers, str) else (feature_drivers or [])
            pos = sorted([(f, float(v)) for f, v in (ds or []) if float(v) > 0], key=lambda x: -x[1])
            for feat, _ in pos[:2]:
                name = _PLAIN_WHY.get(feat, feat.replace("_", " ").title())
                driver_parts.append(name)
        except Exception as exc:
            _logger.debug(f"parse_drivers rec_history: {exc}")

        drivers_str = (
            f'<span style="color:{TEXT2};font-size:{FONT_LABEL};"> · {" · ".join(driver_parts)}</span>'
            if driver_parts else ""
        )

        # ── Why column (plain English for end users) ──────────────────────────
        if rec == "BUY" and key in executed:
            exec_p    = executed_price.get(key, price_at_rec)
            price_str = f" at ${exec_p:,.2f}" if exec_p else ""
            why = (
                f'<span style="color:{GAIN};font-weight:600;">✓ Bought{price_str}</span>'
                + drivers_str
            )
        elif rec == "BUY":
            # Confidence-based reason why bot held back
            if conf < 0.60:
                held_reason = "Signal too weak to act on"
            elif market_ctx:
                held_reason = f"Signal fired during {market_ctx.lower()} — bot at capacity"
            else:
                held_reason = "Bot at position or cash limit"
            why = (
                f'<span style="color:{NEURAL};">Wanted to buy</span>'
                f'<span style="color:{TEXT2};font-size:{FONT_LABEL};"> · {held_reason}</span>'
                + drivers_str
            )
        else:
            if market_ctx:
                hold_text = f"No strong signal · {market_ctx}"
            else:
                hold_text = "No strong signal yet"
            why = f'<span style="color:{TEXT2};font-size:{FONT_LABEL};">{hold_text}</span>'

        rows += (
            f'<tr>'
            f'<td {td}>{dot}{_symbol(str(symbol))}</td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{date_str}</span></td>'
            f'<td {td}>{_action_badge(rec)}{change_html}</td>'
            f'<td {td}>{_confidence_bar(conf, show_label=False)}</td>'
            f'<td {td}>{models_html}</td>'
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
    legend = (
        f'<div style="background:{BG};border-top:1px solid {BORDER};'
        f'padding:8px 14px;font-size:{FONT_LABEL};color:{TEXT2};line-height:1.8;">'
        f'<span style="color:{GAIN};font-weight:700;">●</span> Executed &nbsp;·&nbsp; '
        f'<span style="color:{NEURAL};font-weight:700;">●</span> Held back &nbsp;·&nbsp; '
        f'<span style="color:{TEXT3};font-weight:700;">●</span> Hold / Watch &nbsp;&nbsp;|&nbsp;&nbsp; '
        f'<b>Models</b>: XGB = breakout/momentum · LSTM = price-sequence &nbsp;·&nbsp; '
        f'<b>Why</b>: top signals that drove the AI decision'
        f'</div>'
    )
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Symbol</th>'
        f'<th {TH}>Date</th>'
        f'<th {TH}>Signal</th>'
        f'<th {TH}>Conviction</th>'
        f'<th {TH}>Models</th>'
        f'<th {TH}>Why</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
        + legend
    )
    return (
        f'<div class="nt nt-wrap">'
        f'{_section("📋", "AI Decision Log", note)}'
        f'{disclaimer}'
        f'{table}'
        f'</div>'
    )


@safe_render("Buy Candidates")
def render_buy_candidates() -> str:
    """Ranked list of stocks the AI wants to buy but hasn't yet — not in open positions."""
    rows = safe_query("""
        SELECT s.symbol, s.ensemble_score, s.xgb_prob, s.lstm_prob,
               s.feature_drivers, s.regime, s.ensemble_action, s.timestamp
        FROM signal_log s
        INNER JOIN (
            SELECT symbol, MAX(id) AS max_id
            FROM signal_log
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY symbol
        ) latest ON s.id = latest.max_id
        LEFT JOIN position_state ps ON ps.symbol = s.symbol
        WHERE ps.symbol IS NULL
          AND s.ensemble_score >= 0.50
        ORDER BY s.ensemble_score DESC
        LIMIT 15
    """, default=[])

    if not rows:
        msg = (
            f'<div style="color:{TEXT2};text-align:center;padding:28px;font-size:{FONT_VALUE};">'
            f'No candidates right now &mdash; either all high-scoring symbols are already held, '
            f'or the AI hasn\'t fired signals above 50% in the last 7 days.</div>'
        )
        return (
            f'<div class="nt nt-wrap">'
            f'{_section("🎯", "Buy Candidates", "ranked by AI confidence")}'
            f'{_card(msg)}'
            f'</div>'
        )

    _regime_map = {
        "bullish": "Rising market", "bull": "Rising market",
        "bearish": "Falling market", "bear": "Falling market",
        "ranging": "Sideways market", "range": "Sideways market",
        "volatile": "Volatile market",
    }

    table_rows = ""
    for rank, row in enumerate(rows, start=1):
        symbol, score, xgb_prob, lstm_prob, feature_drivers, regime, action, ts = row
        score    = float(score    or 0)
        xgb      = float(xgb_prob  or 0)
        lstm     = float(lstm_prob  or 0)
        is_last  = rank == len(rows)
        td       = TD0 if is_last else TD

        # Priority badge
        if score >= 0.75:
            badge_bg, badge_fg, badge_lbl = "#0d3320", GAIN,   "STRONG BUY"
        elif score >= 0.60:
            badge_bg, badge_fg, badge_lbl = "#1a1a0d", NEURAL, "BUY"
        else:
            badge_bg, badge_fg, badge_lbl = "#1a1a1a", TEXT2,  "BUILDING"

        priority_badge = (
            f'<span style="background:{badge_bg};border:1px solid {badge_fg};color:{badge_fg};'
            f'font-size:{FONT_LABEL};font-weight:700;padding:2px 7px;border-radius:4px;'
            f'white-space:nowrap;">{badge_lbl}</span>'
        )

        # Rank number
        rank_c = GAIN if rank == 1 else (NEURAL if rank <= 3 else TEXT2)
        rank_html = (
            f'<span style="font-size:{FONT_SECTION};font-weight:700;color:{rank_c};">#{rank}</span>'
        )

        # Confidence
        score_c = GAIN if score >= 0.75 else (NEURAL if score >= 0.60 else TEXT2)
        score_html = f'<span style="font-weight:700;color:{score_c};">{score*100:.0f}%</span>'

        # Models breakdown
        if xgb > 0 or lstm > 0:
            xgb_c  = GAIN  if xgb  >= 0.70 else (NEURAL if xgb  >= 0.55 else TEXT2)
            lstm_c = GAIN  if lstm >= 0.65 else (NEURAL if lstm >= 0.50 else TEXT2)
            models_html = (
                f'<span style="font-size:{FONT_LABEL};font-family:monospace;">'
                f'<span style="color:{xgb_c};">XGB {xgb*100:.0f}%</span>'
                f'<span style="color:{TEXT3};"> · </span>'
                f'<span style="color:{lstm_c};">LSTM {lstm*100:.0f}%</span>'
                f'</span>'
            )
        else:
            models_html = f'<span style="color:{TEXT3};font-size:{FONT_LABEL};">&mdash;</span>'

        # Plain-English "why" from top 2 positive feature drivers
        driver_parts: list[str] = []
        try:
            ds = json.loads(feature_drivers) if isinstance(feature_drivers, str) else (feature_drivers or [])
            pos = sorted([(f, float(v)) for f, v in (ds or []) if float(v) > 0], key=lambda x: -x[1])
            for feat, _ in pos[:2]:
                driver_parts.append(_PLAIN_WHY.get(feat, feat.replace("_", " ").title()))
        except Exception as exc:
            _logger.debug(f"buy_candidates driver parse: {exc}")
        why_html = (
            f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">'
            + (" · ".join(driver_parts) if driver_parts else "&mdash;")
            + "</span>"
        )

        # Market trend
        regime_clean = str(regime or "").lower().replace("_", " ").strip()
        market = _regime_map.get(regime_clean, regime_clean.title() if regime_clean else "&mdash;")
        market_html = f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">{market}</span>'

        # Last signal time
        ts_short = str(ts or "")[:16].replace("T", " ")
        ts_html  = f'<span style="font-size:{FONT_LABEL};color:{TEXT3};font-family:monospace;">{ts_short}</span>'

        table_rows += (
            f'<tr>'
            f'<td {td}>{rank_html}</td>'
            f'<td {td}>{_sym(str(symbol))}</td>'
            f'<td {td}>{priority_badge}</td>'
            f'<td {td}>{score_html}</td>'
            f'<td {td}>{models_html}</td>'
            f'<td {td}>{why_html}</td>'
            f'<td {td}>{market_html}</td>'
            f'<td {td}>{ts_html}</td>'
            f'</tr>'
        )

    note = f"{len(rows)} candidate{'s' if len(rows) != 1 else ''} · not currently held · last 7 days"
    disclaimer = (
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};padding:8px 4px 10px;">'
        f'These are stocks the AI scored highly but the bot hasn\'t bought yet &mdash; '
        f'either the bot is at its position limit, cash is low, or the signal fired outside market hours. '
        f'Higher rank = stronger AI conviction. Not a recommendation to trade manually.'
        f'</div>'
    )
    help_block = (
        f'<div style="background:{BG};border-top:1px solid {BORDER};'
        f'padding:8px 14px;font-size:{FONT_LABEL};color:{TEXT2};line-height:1.8;">'
        f'<b>STRONG BUY</b> ≥75% &nbsp;·&nbsp; <b>BUY</b> 60–75% &nbsp;·&nbsp; '
        f'<b>BUILDING</b> 50–60% (signal not strong enough to act yet) &nbsp;·&nbsp; '
        f'<b>Why</b> = top signals driving the AI score'
        f'</div>'
    )
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Rank</th>'
        f'<th {TH}>Symbol</th>'
        f'<th {TH}>Priority</th>'
        f'<th {TH}>AI Confidence</th>'
        f'<th {TH}>Models</th>'
        f'<th {TH}>Why the AI likes it</th>'
        f'<th {TH}>Market Trend</th>'
        f'<th {TH}>Last Signal</th>'
        f'</tr></thead><tbody>{table_rows}</tbody></table>'
        + help_block
    )
    return (
        f'<div class="nt nt-wrap">'
        f'{_section("🎯", "Buy Candidates", note)}'
        f'{disclaimer}'
        f'{table}'
        f'</div>'
    )
