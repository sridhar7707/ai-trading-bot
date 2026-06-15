"""Portfolio performance and since-yesterday comparison."""
from __future__ import annotations
import datetime
from loguru import logger
from dashboard.design_system import (
    BG, SURFACE, SURFACE2, BORDER, TEXT1, TEXT2, TEXT3,
    ACTION_BUY, ACTION_SELL, ACTION_TRIM,
    GAIN, LOSS, NEURAL, GAIN_BD, LOSS_BD,
    FONT_HERO, FONT_SECTION, FONT_VALUE, FONT_LABEL,
    WEIGHT_BOLD, WEIGHT_MEDIUM, WEIGHT_NORMAL,
    CARD_PADDING, SECTION_GAP,
    _card, _label, _section_title, _action_badge, _symbol,
    _metric_row, _divider, _empty_state, _section, _wrap,
    _stat_card, TH, TD, TD0,
)
from dashboard.data import get_data, get_db_conn, DB_PATH
from bot.core.error_logger import safe_render, timed
import os
_logger = logger

# ── Render: since-yesterday comparison panel ─────────────────────────────────
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
                f'{_section("📅", "Since Yesterday", date_label)}'
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
        return _empty("First session — no comparison available yet.")

    yest_map  = {r[0]: {"score": float(r[1] or 0), "regime": r[2] or "",
                         "sent": float(r[3] or 0)} for r in yest_rows}
    today_map = {r[0]: {"score": float(r[1] or 0), "regime": r[2] or "",
                         "sent": float(r[3] or 0)} for r in today_rows}

    # ── Portfolio delta summary ────────────────────────────────────────────────
    pv_html = ""
    if yest_pv_row and today_pv_row:
        yv = float(yest_pv_row[0] or 0)
        tv = float(today_pv_row[0] or 0)
        if yv > 0:
            delta = tv - yv
            pct   = delta / yv * 100
            d_c   = GAIN if delta >= 0 else LOSS
            icon  = "📈" if delta >= 0 else "📉"
            word  = "up" if delta >= 0 else "down"
            pv_html = (
                f'<div style="background:{BG};border:1px solid {d_c}33;border-radius:6px;'
                f'padding:10px 14px;margin-bottom:12px;display:flex;align-items:center;gap:10px;">'
                f'<span style="font-size:{FONT_VALUE};">{icon}</span>'
                f'<span style="font-size:{FONT_VALUE};color:{TEXT1};">Portfolio '
                f'<strong style="color:{d_c};">{word} ${abs(delta):,.2f} ({pct:+.2f}%)</strong>'
                f' since yesterday</span></div>'
            )

    # ── Per-symbol change rows ─────────────────────────────────────────────────
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
            arrow = (f'<span style="color:{GAIN};font-weight:700;">↑</span>' if score_delta > 0
                     else f'<span style="color:{LOSS};font-weight:700;">↓</span>')
            changes.append(("Confidence", arrow,
                            f'{y["score"] * 100:.0f}% → {t["score"] * 100:.0f}%'))

        if regime_y and regime_t and regime_y != regime_t:
            changes.append(("Regime",
                            f'<span style="color:{TEXT2};font-weight:700;">→</span>',
                            f'{regime_y} → {regime_t}'))

        if sent_y != sent_t:
            if sent_t == "Positive":
                s_arrow = f'<span style="color:{GAIN};font-weight:700;">↑</span>'
            elif sent_t == "Negative":
                s_arrow = f'<span style="color:{LOSS};font-weight:700;">↓</span>'
            else:
                s_arrow = f'<span style="color:{TEXT2};font-weight:700;">→</span>'
            changes.append(("Sentiment", s_arrow, f'{sent_y} → {sent_t}'))

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
            f'No significant changes since yesterday — AI signals are stable.</div>'
        )

    return (
        f'<div class="nt nt-wrap">'
        f'{_section("📅", "Since Yesterday", date_label)}'
        f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;padding:14px 16px;">'
        f'{pv_html}{rows_html}'
        f'</div></div>'
    )


# ── Portfolio performance period helpers ──────────────────────────────────────
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

            # Current (latest) portfolio value
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

                if key == "YTD":
                    cutoff = datetime.date(today.year, 1, 1).isoformat()
                else:
                    cutoff = (today - datetime.timedelta(days=days)).isoformat()

                # First DB record on or after the cutoff date (start-of-period proxy)
                row = con.execute(
                    "SELECT portfolio_value, timestamp FROM trades "
                    "WHERE portfolio_value > 0 AND date(timestamp) >= ? ORDER BY id ASC LIMIT 1",
                    (cutoff,),
                ).fetchone()
                if row:
                    start_val  = float(row[0])
                    start_date = row[1][:10]
                    result[key] = (start_val, cur_val, start_date)
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
    # Strip the inline stat suffix so we always have a clean key
    key = period.split()[0] if period else "1M"

    stats = _query_perf_stats()
    cur_row_val = None
    first_any   = any(v is not None for v in stats.values())

    if not first_any:
        empty = (f'<div style="color:{TEXT2};text-align:center;padding:20px;font-size:{FONT_LABEL};">'
                 f'No portfolio history yet — data appears after the first trade.</div>')
        return f'<div class="nt nt-wrap">{empty}</div>'

    s = stats.get(key)

    # ── Strip: all period mini-badges (decorative, Radio handles selection) ────
    strip_items = ""
    for pk, _ in _PERF_PERIODS:
        ps = stats.get(pk)
        if ps and ps[0] > 0:
            pct = (ps[1] - ps[0]) / ps[0] * 100
            c   = GAIN if pct >= 0 else LOSS
            strip_items += (
                f'<div style="text-align:center;padding:8px 12px;background:{SURFACE};'
                f'border:1px solid {"" + PRIMARY if pk == key else BORDER};'
                f'border-radius:6px;min-width:60px;">'
                f'<div style="font-size:{FONT_LABEL};color:{"" + PRIMARY if pk == key else TEXT2};'
                f'font-weight:700;margin-bottom:4px;">{pk}</div>'
                f'<div style="font-size:{FONT_LABEL};color:{c};font-weight:700;">{pct:+.1f}%</div>'
                f'</div>'
            )
        else:
            strip_items += (
                f'<div style="text-align:center;padding:8px 12px;background:{BG};'
                f'border:1px solid {BORDER};border-radius:6px;min-width:60px;opacity:0.4;">'
                f'<div style="font-size:{FONT_LABEL};color:{TEXT2};font-weight:700;margin-bottom:4px;">{pk}</div>'
                f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">—</div>'
                f'</div>'
            )

    strip = (
        f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;">'
        f'{strip_items}</div>'
    )

    # ── Detail card for selected period ───────────────────────────────────────
    if not s:
        detail = (
            f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;'
            f'padding:20px;text-align:center;color:{TEXT2};font-size:{FONT_VALUE};">'
            f'Not enough data yet for <strong>{key}</strong> — the bot needs more trading history.</div>'
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
            # From → To
            f'<div style="display:flex;align-items:center;gap:12px;">'
            f'<span style="font-size:{FONT_VALUE};color:{TEXT2};">From</span>'
            f'<span style="font-size:{FONT_VALUE};font-weight:700;color:{TEXT1};">${start_val:,.2f}</span>'
            f'<span style="font-size:{FONT_VALUE};color:{TEXT2};">→</span>'
            f'<span style="font-size:{FONT_VALUE};font-weight:700;color:{TEXT1};">${end_val:,.2f}</span>'
            f'</div>'
            # Start date
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};margin-top:10px;">'
            f'Period start: {start_date}</div>'
            f'</div>'
        )

    return f'<div class="nt nt-wrap">{strip}{detail}</div>'


# ── Render: today's trades timeline ─────────────────────────────────────────────
