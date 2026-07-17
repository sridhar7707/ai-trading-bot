"""Portfolio performance and since-yesterday comparison."""
from __future__ import annotations
import datetime
from loguru import logger
from dashboard.design_system import (
    BG, SURFACE, BORDER, TEXT1, TEXT2,
    PRIMARY, GAIN, LOSS,
    FONT_HERO, FONT_VALUE, FONT_LABEL,
    _section, _wrap,
)
from dashboard.data import get_data, get_db_conn, DB_PATH
from bot.core.error_logger import safe_render, timed
import os
_logger = logger

# â”€â”€ Render: since-yesterday comparison panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@timed(_logger)
@safe_render("Since Yesterday")
def render_whats_changed() -> str:
    today     = datetime.date.today().isoformat()
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    d         = datetime.date.today()
    date_label = f"{d.strftime('%B')} {d.day}"

    def _empty(msg: str) -> str:
        inner = (f'<div style="color:{TEXT2};text-align:center;padding:24px;font-size:{FONT_LABEL};">{msg}</div>')
        return (f'<div class="nt nt-wrap">'
                f'{_section("&#x1F4C5;", "Since Yesterday", date_label)}'
                f'{_wrap(inner)}</div>')

    if not os.path.exists(DB_PATH):
        return _empty("No trade data yet.")

    try:
        with get_db_conn() as con:
            def _latest_per_symbol(date_str: str) -> list:
                return con.execute(
                    "SELECT t.symbol, t.ensemble_score, t.regime, t.sentiment_score, t.portfolio_value "
                    "FROM trades t "
                    "INNER JOIN (SELECT symbol, MAX(id) AS mid FROM trades "
                    "            WHERE date(timestamp) = ? GROUP BY symbol) m "
                    "ON t.id = m.mid",
                    (date_str,),
                ).fetchall()

            today_rows = _latest_per_symbol(today)
            yest_rows  = _latest_per_symbol(yesterday)

            # Portfolio bookends: yesterday's last overall value vs today's last
            yest_pv_row  = con.execute(
                "SELECT portfolio_value FROM trades WHERE date(timestamp) = ? "
                "AND portfolio_value > 0 ORDER BY id DESC LIMIT 1", (yesterday,)
            ).fetchone()
            today_pv_row = con.execute(
                "SELECT portfolio_value FROM trades WHERE portfolio_value > 0 "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
    except Exception as exc:
        logger.warning(f"render_whats_changed DB error: {exc}")
        return _empty("Could not load comparison data.")

    if not yest_rows:
        # No prior-day trade data &mdash; show today's intraday progress instead
        try:
            with get_db_conn() as con:
                first_snap = con.execute(
                    "SELECT portfolio_value FROM portfolio_snapshots "
                    "WHERE date(timestamp) = ? ORDER BY timestamp ASC LIMIT 1",
                    (today,),
                ).fetchone()
                last_snap = con.execute(
                    "SELECT portfolio_value FROM portfolio_snapshots "
                    "ORDER BY timestamp DESC LIMIT 1",
                ).fetchone()
        except Exception:
            return _empty("First session &mdash; no comparison available yet.")

        if not first_snap or not last_snap:
            return _empty("First session &mdash; no comparison available yet.")

        start_v = float(first_snap[0] or 0)
        cur_v   = float(last_snap[0] or 0)
        if start_v <= 0:
            return _empty("First session &mdash; no comparison available yet.")

        delta = cur_v - start_v
        pct   = delta / start_v * 100
        d_c   = GAIN if delta >= 0 else LOSS
        icon  = "&#x1F4C8;" if delta >= 0 else "&#x1F4C9;"
        word  = "up" if delta >= 0 else "down"

        d_data   = get_data()
        open_pos = d_data["open_pos"]
        prices   = d_data["prices"]

        pos_rows = ""
        for sym, pos in open_pos.items():
            cur      = prices.get(sym, 0.0)
            invested = pos["invested"]
            cur_val  = pos["shares"] * cur if cur > 0 else invested
            p_pct    = (cur_val - invested) / invested * 100 if invested > 0 else 0.0
            p_c      = GAIN if p_pct >= 0 else LOSS
            pos_rows += (
                f'<div style="display:grid;grid-template-columns:90px 1fr 70px;'
                f'align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid {BORDER};">'
                f'<span style="font-family:Courier New,monospace;font-weight:700;'
                f'color:{PRIMARY};font-size:{FONT_VALUE};">{sym}</span>'
                f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">Open position</span>'
                f'<span style="font-size:{FONT_LABEL};color:{p_c};font-weight:700;'
                f'text-align:right;">{p_pct:+.1f}%</span>'
                f'</div>'
            )
        if not pos_rows:
            pos_rows = (f'<div style="color:{TEXT2};font-size:{FONT_LABEL};padding:8px 0;">'
                        f'No open positions yet.</div>')

        pv_html = (
            f'<div style="background:{BG};border:1px solid {d_c}33;border-radius:6px;'
            f'padding:10px 14px;margin-bottom:12px;display:flex;align-items:center;gap:10px;">'
            f'<span style="font-size:{FONT_VALUE};">{icon}</span>'
            f'<span style="font-size:{FONT_VALUE};color:{TEXT1};">Portfolio '
            f'<strong style="color:{d_c};">{word} ${abs(delta):,.2f} ({pct:+.2f}%)</strong>'
            f' today</span></div>'
        )
        _today_title = "Today's Progress"
        return (
            f'<div class="nt nt-wrap">'
            f'{_section("&#x1F4C5;", _today_title, date_label)}'
            f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;padding:14px 16px;">'
            f'{pv_html}{pos_rows}</div></div>'
        )

    yest_map  = {r[0]: {"score": float(r[1] or 0), "regime": r[2] or "",
                         "sent": float(r[3] or 0)} for r in yest_rows}
    today_map = {r[0]: {"score": float(r[1] or 0), "regime": r[2] or "",
                         "sent": float(r[3] or 0)} for r in today_rows}

    # â”€â”€ Portfolio delta summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    pv_html = ""
    if yest_pv_row and today_pv_row:
        yv = float(yest_pv_row[0] or 0)
        tv = float(today_pv_row[0] or 0)
        if yv > 0:
            delta = tv - yv
            pct   = delta / yv * 100
            d_c   = GAIN if delta >= 0 else LOSS
            icon  = "&#x1F4C8;" if delta >= 0 else "&#x1F4C9;"
            word  = "up" if delta >= 0 else "down"
            pv_html = (
                f'<div style="background:{BG};border:1px solid {d_c}33;border-radius:6px;'
                f'padding:10px 14px;margin-bottom:12px;display:flex;align-items:center;gap:10px;">'
                f'<span style="font-size:{FONT_VALUE};">{icon}</span>'
                f'<span style="font-size:{FONT_VALUE};color:{TEXT1};">Portfolio '
                f'<strong style="color:{d_c};">{word} ${abs(delta):,.2f} ({pct:+.2f}%)</strong>'
                f' since yesterday</span></div>'
            )

    # â”€â”€ Per-symbol change rows â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _sent_label(v: float) -> str:
        return "Positive" if v > 0.05 else ("Negative" if v < -0.05 else "Neutral")

    rows_html    = ""
    changes_seen = False

    for sym in sorted(set(yest_map) & set(today_map)):
        y = yest_map[sym]
        t = today_map[sym]

        score_delta = t["score"] - y["score"]
        regime_y    = y["regime"].replace("_", " ").title()
        regime_t    = t["regime"].replace("_", " ").title()
        sent_y      = _sent_label(y["sent"])
        sent_t      = _sent_label(t["sent"])

        changes: list[tuple[str, str, str]] = []

        if abs(score_delta) > 0.05:
            arrow = (f'<span style="color:{GAIN};font-weight:700;">&uarr;</span>' if score_delta > 0
                     else f'<span style="color:{LOSS};font-weight:700;">&darr;</span>')
            changes.append(("Confidence", arrow,
                            f'{y["score"] * 100:.0f}% &rarr; {t["score"] * 100:.0f}%'))

        if regime_y and regime_t and regime_y != regime_t:
            changes.append(("Regime",
                            f'<span style="color:{TEXT2};font-weight:700;">&rarr;</span>',
                            f'{regime_y} &rarr; {regime_t}'))

        if sent_y != sent_t:
            if sent_t == "Positive":
                s_arrow = f'<span style="color:{GAIN};font-weight:700;">&uarr;</span>'
            elif sent_t == "Negative":
                s_arrow = f'<span style="color:{LOSS};font-weight:700;">&darr;</span>'
            else:
                s_arrow = f'<span style="color:{TEXT2};font-weight:700;">&rarr;</span>'
            changes.append(("Sentiment", s_arrow, f'{sent_y} &rarr; {sent_t}'))

        if not changes:
            continue

        changes_seen = True
        for i, (metric, arrow_html, mag) in enumerate(changes):
            sym_cell = (f'<span style="font-family:Courier New,monospace;font-weight:700;'
                        f'color:{PRIMARY};font-size:{FONT_VALUE};">{sym}</span>') if i == 0 else ""
            rows_html += (
                f'<div style="display:grid;grid-template-columns:80px 100px 32px 1fr;'
                f'align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid {BORDER};">'
                f'<div>{sym_cell}</div>'
                f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">{metric}</div>'
                f'<div style="text-align:center;font-size:{FONT_VALUE};">{arrow_html}</div>'
                f'<div style="font-size:{FONT_LABEL};color:{TEXT1};">{mag}</div>'
                f'</div>'
            )

    if not changes_seen:
        rows_html = (
            f'<div style="color:{TEXT2};font-size:{FONT_LABEL};padding:12px 0;text-align:center;">'
            f'No significant changes since yesterday &mdash; AI signals are stable.</div>'
        )

    return (
        f'<div class="nt nt-wrap">'
        f'{_section("&#x1F4C5;", "Since Yesterday", date_label)}'
        f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;padding:14px 16px;">'
        f'{pv_html}{rows_html}'
        f'</div></div>'
    )


# â”€â”€ Portfolio performance period helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_PERF_PERIODS = [
    ("1D",       1),
    ("1W",       7),
    ("1M",      30),
    ("3M",      90),
    ("YTD",     None),   # special: Jan 1 of current year
    ("1Y",     365),
    ("All Time", None),  # special: first DB record
]
_PERF_LABELS = {
    "1D":       "today",
    "1W":       "this week",
    "1M":       "this month",
    "3M":       "last 3 months",
    "YTD":      "year to date",
    "1Y":       "last year",
    "All Time": "since inception",
}


def _query_perf_stats() -> dict[str, tuple[float, float, str] | None]:
    """
    Returns {period_key: (start_val, end_val, start_date_label) | None}.
    None means insufficient data for that period.
    """
    if not os.path.exists(DB_PATH):
        return {k: None for k, _ in _PERF_PERIODS}
    try:
        with get_db_conn() as con:
            today  = datetime.date.today()

            # Current portfolio value: use the most recent bot snapshot (written after every
            # trading cycle) as the authoritative source. The snapshot records actual cash +
            # position values at that moment and avoids errors from replaying the trades table.
            snap_row = con.execute(
                "SELECT portfolio_value FROM portfolio_snapshots WHERE portfolio_value > 0 "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if snap_row:
                cur_val = float(snap_row[0])
            else:
                cur_row = con.execute(
                    "SELECT portfolio_value FROM trades WHERE portfolio_value > 0 ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if not cur_row:
                    return {k: None for k, _ in _PERF_PERIODS}
                cur_val = float(cur_row[0])

            # First ever portfolio value
            first_row = con.execute(
                "SELECT portfolio_value, timestamp FROM trades WHERE portfolio_value > 0 ORDER BY id ASC LIMIT 1"
            ).fetchone()
            first_val = float(first_row[0]) if first_row else None
            first_ts  = first_row[1][:10]  if first_row else None

            result: dict[str, tuple[float, float, str] | None] = {}

            for key, days in _PERF_PERIODS:
                if key == "All Time":
                    if first_val is not None and first_val != cur_val:
                        result[key] = (first_val, cur_val, first_ts or "")
                    else:
                        result[key] = None
                    continue

                if key == "1D":
                    # "Today's change" = most recent prior trading day's close → now.
                    # Using < today (not = yesterday) so weekends and holidays are
                    # skipped automatically: Monday correctly shows Friday's close.
                    today_str = today.isoformat()
                    row = con.execute(
                        "SELECT portfolio_value, timestamp FROM portfolio_snapshots "
                        "WHERE portfolio_value > 0 AND date(timestamp) < ? "
                        "ORDER BY timestamp DESC LIMIT 1",
                        (today_str,),
                    ).fetchone()
                    if not row:
                        row = con.execute(
                            "SELECT portfolio_value, timestamp FROM trades "
                            "WHERE portfolio_value > 0 AND date(timestamp) < ? "
                            "ORDER BY id DESC LIMIT 1",
                            (today_str,),
                        ).fetchone()
                    if row:
                        result[key] = (float(row[0]), cur_val, row[1][:10])
                    else:
                        result[key] = None
                    continue

                if key == "YTD":
                    cutoff = datetime.date(today.year, 1, 1).isoformat()
                else:
                    cutoff = (today - datetime.timedelta(days=days)).isoformat()

                # portfolio_snapshots has denser coverage than trades (whose portfolio_value
                # is often 0 for BUY entries). Try snapshots first; fall back to trades.
                row = con.execute(
                    "SELECT portfolio_value, timestamp FROM portfolio_snapshots "
                    "WHERE portfolio_value > 0 AND date(timestamp) >= ? "
                    "ORDER BY timestamp ASC LIMIT 1",
                    (cutoff,),
                ).fetchone()
                if not row:
                    row = con.execute(
                        "SELECT portfolio_value, timestamp FROM trades "
                        "WHERE portfolio_value > 0 AND date(timestamp) >= ? "
                        "ORDER BY id ASC LIMIT 1",
                        (cutoff,),
                    ).fetchone()
                if row:
                    result[key] = (float(row[0]), cur_val, row[1][:10])
                else:
                    result[key] = None

            return result
    except Exception as exc:
        logger.warning(f"_query_perf_stats: {exc}")
        return {k: None for k, _ in _PERF_PERIODS}


def _perf_choices() -> list[str]:
    """Build Radio choices with inline % for display, e.g. '1M  +8.9%'."""
    stats = _query_perf_stats()
    choices = []
    for key, _ in _PERF_PERIODS:
        s = stats.get(key)
        if s and s[0] > 0:
            pct = (s[1] - s[0]) / s[0] * 100
            choices.append(f"{key}  {pct:+.1f}%")
        else:
            choices.append(f"{key}  —")
    return choices


@timed(_logger)
@safe_render("Portfolio Performance")
def render_portfolio_performance(period: str = "1M  —") -> str:
    # Choices are "{key}  {stat}" (double space); split()[0] breaks "All Time" → use split("  ")
    key = period.split("  ")[0].strip() if period else "1M"

    stats = _query_perf_stats()
    cur_row_val = None
    first_any   = any(v is not None for v in stats.values())

    if not first_any:
        empty = (f'<div style="color:{TEXT2};text-align:center;padding:20px;font-size:{FONT_LABEL};">'
                 f'No portfolio history yet &mdash; data appears after the first trade.</div>')
        return f'<div class="nt nt-wrap">{empty}</div>'

    s = stats.get(key)

    # â”€â”€ Detail card for selected period â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not s:
        detail = (
            f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;'
            f'padding:20px;text-align:center;color:{TEXT2};font-size:{FONT_VALUE};">'
            f'Not enough data yet for <strong>{key}</strong> &mdash; the bot needs more trading history.</div>'
        )
    else:
        start_val, end_val, start_date = s
        delta     = end_val - start_val
        pct       = (delta / start_val * 100) if start_val > 0 else 0.0
        c         = GAIN if delta >= 0 else LOSS
        sign      = "+" if delta >= 0 else ""
        label_str = _PERF_LABELS.get(key, key.lower())

        detail = (
            f'<div style="background:{SURFACE};border:1px solid {BORDER};'
            f'border-top:3px solid {c};border-radius:8px;padding:20px 24px;">'
            # Big headline
            f'<div style="font-size:{FONT_HERO};font-weight:700;color:{c};letter-spacing:-1px;'
            f'line-height:1;margin-bottom:8px;">'
            f'{sign}${abs(delta):,.2f}</div>'
            # Subline
            f'<div style="font-size:{FONT_VALUE};color:{TEXT2};margin-bottom:14px;">'
            f'{sign}{pct:.2f}% {label_str}</div>'
            # From â†’ To
            f'<div style="display:flex;align-items:center;gap:12px;">'
            f'<span style="font-size:{FONT_VALUE};color:{TEXT2};">From</span>'
            f'<span style="font-size:{FONT_VALUE};font-weight:700;color:{TEXT1};">${start_val:,.2f}</span>'
            f'<span style="font-size:{FONT_VALUE};color:{TEXT2};">&rarr;</span>'
            f'<span style="font-size:{FONT_VALUE};font-weight:700;color:{TEXT1};">${end_val:,.2f}</span>'
            f'</div>'
            # Start date
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};margin-top:10px;">'
            f'Period start: {start_date}</div>'
            f'</div>'
        )

    return f'<div class="nt nt-wrap">{detail}</div>'
