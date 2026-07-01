"""Overview panel: portfolio health, benchmark, trade frequency, SPY banner."""
from __future__ import annotations
import datetime
from loguru import logger
from dashboard.design_system import (
    BG, SURFACE, SURFACE2, BORDER, TEXT1, TEXT2, TEXT3,
    PRIMARY,
    ACTION_BUY, ACTION_SELL, ACTION_TRIM, ACTION_HOLD, ACTION_WATCH,
    GAIN, LOSS, NEURAL, GAIN_BD, LOSS_BD, NEURAL_BD,
    GAIN_BG, LOSS_BG, NEURAL_BG,
    FONT_HERO, FONT_SECTION, FONT_VALUE, FONT_LABEL,
    WEIGHT_BOLD, WEIGHT_MEDIUM, WEIGHT_NORMAL,
    CARD_PADDING, CARD_RADIUS, ROW_PADDING, SECTION_GAP,
    _card, _label, _hero_value, _section_title, _action_badge, _symbol,
    _confidence_bar, _metric_row, _progress_bar, _divider, _empty_state,
    _action_row, _table, _sym, _badge, _num, _pnl, _section, _wrap,
    _stat_card, TH, TD, TD0,
)
from dashboard.data import get_data, _now_ct, _market_status, safe_query
from dashboard.builders import build_health_vm
from bot.core.error_logger import safe_render, timed, log_exception
_logger = logger


# ── PANEL 1: Portfolio Health Hero ────────────────────────────────────────────
@safe_render("Portfolio Health")
def render_portfolio_health_hero() -> str:
    vm   = build_health_vm()
    d    = get_data()
    mkt_label, mkt_color = _market_status()

    score   = vm.total
    grade   = vm.grade
    gl      = vm.grade_label
    grade_c = GAIN if score >= 80 else (NEURAL if score >= 60 else LOSS)

    grade_pill = (
        f'<div style="display:inline-flex;align-items:center;gap:8px;">'
        f'<span style="font-size:{FONT_HERO};font-weight:800;color:{grade_c};'
        f'letter-spacing:-2px;line-height:1;">{grade}</span>'
        f'<div>'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
        f'letter-spacing:.8px;">{gl}</div>'
        f'<div style="font-size:{FONT_SECTION};font-weight:700;color:{TEXT1};line-height:1.1;">'
        f'{score}<span style="font-size:{FONT_LABEL};color:{TEXT2};font-weight:400;">/100</span></div>'
        f'</div>'
        f'</div>'
    )

    bar = (
        f'<div style="margin:10px 0 6px;background:{BORDER};border-radius:3px;height:5px;">'
        f'<div style="background:{grade_c};height:100%;width:{score}%;border-radius:3px;'
        f'transition:width .4s;"></div></div>'
    )

    comp_rows = []
    for c in vm.components:
        comp_rows.append(
            f'<div style="margin-bottom:7px;">'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:baseline;margin-bottom:2px;">'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">{c.label}</span>'
            f'<span style="font-size:{FONT_LABEL};color:{c.color};font-weight:600;">'
            f'{c.score}/{c.max}&nbsp;<span style="color:{TEXT2};font-weight:400;">{c.detail}</span></span>'
            f'</div>'
            f'<div style="background:{BORDER};border-radius:2px;height:3px;">'
            f'<div style="background:{c.color};height:100%;width:{c.pct}%;border-radius:2px;"></div>'
            f'</div>'
            f'</div>'
        )
    comp_html = "".join(comp_rows)

    risk_icon  = "⚠" if score < 80 else "✓"
    risk_callout = (
        f'<div style="margin-top:8px;padding:7px 10px;background:{SURFACE2};'
        f'border-left:3px solid {vm.biggest_risk_color};border-radius:0 4px 4px 0;">'
        f'<span style="font-size:{FONT_LABEL};color:{vm.biggest_risk_color};font-weight:600;">'
        f'{risk_icon} {vm.biggest_risk}</span>'
        f'</div>'
    )

    str_html = ""
    if vm.strengths:
        items = "".join(
            f'<span style="font-size:{FONT_LABEL};color:{GAIN};background:#0a2010;'
            f'border-radius:3px;padding:2px 6px;">{s}</span> '
            for s in vm.strengths
        )
        str_html = f'<div style="margin-top:6px;">{items}</div>'

    open_pos  = d.get("open_pos", {})
    prices    = d.get("prices", {})
    total_inv = sum(v["invested"] for v in open_pos.values())
    total_cur = sum(v["shares"] * prices.get(s, 0.0) for s, v in open_pos.items())
    total_pnl = total_cur - total_inv
    pnl_pct   = (total_pnl / total_inv * 100) if total_inv > 0 else 0.0
    pnl_c     = GAIN if total_pnl >= 0 else LOSS
    pnl_sign  = "+" if total_pnl >= 0 else ""
    hero_chg  = (f'{pnl_sign}${total_pnl:,.2f} ({pnl_pct:+.2f}%)'
                 if total_inv > 0 else "No open positions")

    def _stat(label, val, color=None):
        c = color or TEXT1
        return (
            f'<div style="text-align:center;padding:10px 14px;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
            f'letter-spacing:.7px;margin-bottom:4px;">{label}</div>'
            f'<div style="font-size:{FONT_SECTION};font-weight:700;color:{c};">{val}</div>'
            f'</div>'
        )

    avg_conf = d.get("avg_confidence", 0.0)
    conf_str = f"{avg_conf*100:.0f}%" if avg_conf > 0 else "&mdash;"
    conf_c   = GAIN if avg_conf >= 0.75 else (NEURAL if avg_conf >= 0.60 else TEXT2)
    vix      = d.get("vix", 0.0)
    vix_str  = f"{vix:.1f}" if vix > 0 else "&mdash;"
    vix_c    = GAIN if vix < 15 else (NEURAL if vix < 25 else LOSS)

    # Total realized gain from all closed trades
    _realized = safe_query(
        "SELECT SUM(pnl_pct * notional) FROM trades"
        " WHERE action LIKE 'SELL%' AND action != 'SELL_RECONCILE'"
        " AND pnl_pct IS NOT NULL AND notional IS NOT NULL",
        default=[(None,)]
    )
    realized_gain  = float((_realized[0][0] or 0) if _realized else 0)
    realized_str   = f"${realized_gain:+,.2f}" if realized_gain != 0 else "$0.00"
    realized_c     = GAIN if realized_gain >= 0 else LOSS

    # Bot status: Scanning / Idle / Market Closed
    _last_scan = safe_query(
        "SELECT MAX(timestamp) FROM signal_log"
        " WHERE timestamp >= datetime('now','-90 minutes')",
        default=[(None,)]
    )
    _has_recent = bool(_last_scan and _last_scan[0][0])
    if "closed" in mkt_label.lower() or "pre" in mkt_label.lower() or "after" in mkt_label.lower():
        bot_status, bot_c = "Market Closed", TEXT3
    elif _has_recent:
        bot_status, bot_c = "Scanning", GAIN
    else:
        bot_status, bot_c = "Waiting", NEURAL

    stats_row = (
        f'<div style="display:flex;flex-wrap:wrap;border-top:1px solid {BORDER};margin-top:10px;">'
        + _stat("Portfolio", d.get("portfolio", "&mdash;"), TEXT1)
        + _stat("Open P&L", hero_chg, pnl_c)
        + _stat("Realized Gain", realized_str, realized_c)
        + _stat("Positions", str(len(open_pos)), TEXT1)
        + _stat("Bot Status", bot_status, bot_c)
        + f'</div>'
    )

    body = (
        f'<div style="display:flex;gap:20px;flex-wrap:wrap;">'
        f'<div style="flex:0 0 auto;min-width:160px;">{grade_pill}{bar}'
        f'{risk_callout}{str_html}</div>'
        f'<div style="flex:1;min-width:200px;">{comp_html}</div>'
        f'</div>'
        f'{stats_row}'
    )

    timestamp = (
        f'<div class="nt-status">'
        f'<span style="color:{TEXT2};font-size:{FONT_LABEL};">'
        f'Updated &nbsp;<strong style="color:{TEXT1};">{_now_ct()}</strong></span>'
        f'<span style="display:inline-flex;align-items:center;gap:5px;">'
        f'<span style="width:6px;height:6px;background:{mkt_color};border-radius:50%;'
        f'display:inline-block;"></span>'
        f'<span style="color:{mkt_color};font-weight:600;font-size:{FONT_LABEL};">'
        f'{mkt_label}</span></span>'
        f'<span class="nt-refresh-label" style="color:{TEXT2};font-size:{FONT_LABEL};">60s refresh</span>'
        f'</div>'
    )
    inner = (
        f'<div class="nt-card" style="padding:20px 18px;">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
        f'letter-spacing:.8px;margin-bottom:12px;">Portfolio Health</div>'
        f'{body}'
        f'</div>'
        f'{timestamp}'
    )
    return f'<div class="nt nt-wrap">{inner}</div>'


# ── Render: benchmark comparison (vs SPY / QQQ) ───────────────────────────────
@timed(_logger)
@safe_render("Benchmark")
def render_benchmark_comparison() -> str:
    from database.services.analytics_service import analytics_service
    from dashboard.data import get_data, get_db_conn, DB_PATH
    import os

    d = get_data()
    pv = 0.0
    try:
        raw = d.get("portfolio", "&mdash;")
        if raw != "&mdash;":
            pv = float(raw.replace("$", "").replace(",", ""))
    except (ValueError, TypeError):
        pass

    port_return = 0.0
    if pv > 0:
        # Compute portfolio return from SQLite portfolio_snapshots (always available)
        try:
            if os.path.exists(DB_PATH):
                with get_db_conn() as _con:
                    first_row = _con.execute(
                        "SELECT portfolio_value FROM portfolio_snapshots "
                        "WHERE portfolio_value > 0 ORDER BY timestamp ASC LIMIT 1"
                    ).fetchone()
                if first_row:
                    first_val = float(first_row[0])
                    if first_val > 0:
                        port_return = (pv - first_val) / first_val * 100
        except Exception as exc:
            log_exception(_logger, "render_benchmark.calc_return", exc)

    bm = analytics_service.get_benchmark_comparison(
        portfolio_return_pct=port_return, period="YTD"
    )

    def _vs_badge(delta: float) -> str:
        if delta > 0:
            return (f'<span style="color:{ACTION_BUY};font-weight:{WEIGHT_BOLD};">'
                    f'+{delta:.1f}% ahead</span>')
        if delta < 0:
            return (f'<span style="color:{ACTION_SELL};font-weight:{WEIGHT_BOLD};">'
                    f'{delta:.1f}% behind</span>')
        return f'<span style="color:{TEXT2};">In line</span>'

    port_c = ACTION_BUY if port_return >= 0 else ACTION_SELL
    spy_c  = ACTION_BUY if bm["spy_return"] >= 0 else ACTION_SELL
    qqq_c  = ACTION_BUY if bm["qqq_return"] >= 0 else ACTION_SELL

    content = (
        _metric_row("Your Portfolio", f'{port_return:+.1f}%', port_c)
        + _metric_row("S&P 500 (SPY)",  f'{bm["spy_return"]:+.1f}%', spy_c,
                      _vs_badge(bm["vs_spy"]))
        + _metric_row("Nasdaq (QQQ)",   f'{bm["qqq_return"]:+.1f}%', qqq_c,
                      _vs_badge(bm["vs_qqq"]))
    )

    note = "YTD · paper trading" if "error" not in bm else "YTD · SPY/QQQ unavailable"
    return (
        f'<div class="nt nt-wrap">'
        f'{_section("📈", "vs Benchmark", note)}'
        f'{_card(content)}'
        f'</div>'
    )


# ── Render: trade frequency (weekly discipline card) ─────────────────────────
@safe_render("Trade Frequency")
def render_trade_frequency() -> str:
    import datetime
    today   = datetime.date.today()
    monday  = today - datetime.timedelta(days=today.weekday())
    sunday  = monday + datetime.timedelta(days=6)
    mon_str = str(monday)
    sun_str = str(sunday)

    rows = safe_query(
        "SELECT action, timestamp, pnl_pct FROM trades "
        "WHERE date(timestamp) BETWEEN ? AND ? AND action != 'SELL_RECONCILE'",
        (mon_str, sun_str), default=[]
    )

    buys  = sum(1 for r in (rows or []) if str(r[0] or "").startswith("BUY"))
    sells = sum(1 for r in (rows or []) if str(r[0] or "").startswith("SELL"))

    # Average hold time: for each sell this week, find matching buy to get duration
    hold_days: list[float] = []
    sell_rows = safe_query(
        """SELECT t_sell.symbol, t_sell.timestamp AS sell_ts,
                  MAX(t_buy.timestamp) AS buy_ts
           FROM trades t_sell
           JOIN trades t_buy ON t_buy.symbol = t_sell.symbol AND t_buy.action = 'BUY'
                             AND t_buy.timestamp < t_sell.timestamp
           WHERE date(t_sell.timestamp) BETWEEN ? AND ?
             AND t_sell.action LIKE 'SELL%' AND t_sell.action != 'SELL_RECONCILE'
           GROUP BY t_sell.symbol, t_sell.timestamp""",
        (mon_str, sun_str), default=[]
    )
    for row in (sell_rows or []):
        try:
            from dashboard.data import _to_ct
            sell_dt = _to_ct(row[1])
            buy_dt  = _to_ct(row[2])
            if sell_dt and buy_dt and len(sell_dt) >= 10 and len(buy_dt) >= 10:
                import datetime as _dt
                d1 = _dt.datetime.strptime(buy_dt[:10],  "%Y-%m-%d")
                d2 = _dt.datetime.strptime(sell_dt[:10], "%Y-%m-%d")
                delta = (d2 - d1).days
                if delta >= 0:
                    hold_days.append(float(delta))
        except Exception:
            pass

    avg_hold = sum(hold_days) / len(hold_days) if hold_days else None

    # Target: 2–3 buys/week
    buys_c = GAIN if 2 <= buys <= 4 else (NEURAL if buys == 1 else (LOSS if buys > 6 else TEXT2))
    sells_c = GAIN if sells >= 1 else TEXT2

    hold_str = f"{avg_hold:.1f} days" if avg_hold is not None else "—"
    hold_c   = GAIN if avg_hold is not None and 1 <= avg_hold <= 7 else (NEURAL if avg_hold is not None else TEXT2)

    week_label = f"{mon_str} – {sun_str}"
    cards = (
        f'<div class="nt-cards">'
        + _stat_card("Buys This Week",  str(buys),  TEXT2, buys_c,
                     "Target: 2–3 per week", 0.00)
        + _stat_card("Sells This Week", str(sells), TEXT2, sells_c,
                     "Exits closed this week", 0.06)
        + _stat_card("Avg Hold Time",   hold_str,   TEXT2, hold_c,
                     "Target: 1–7 days per trade", 0.12)
        + f'</div>'
    )
    return (
        f'<div class="nt nt-wrap">'
        f'{_section("📅", "This Week" + chr(39) + "s Trading", week_label)}'
        f'{cards}'
        f'</div>'
    )


# ── Render: SPY outperformance banner ─────────────────────────────────────────
@safe_render("SPY Banner")
def render_spy_banner() -> str:
    from database.services.analytics_service import analytics_service
    from dashboard.data import get_data, get_db_conn, DB_PATH
    import os

    d = get_data()
    pv = 0.0
    try:
        raw = d.get("portfolio", "—")
        if raw not in ("—", "&mdash;"):
            pv = float(raw.replace("$", "").replace(",", ""))
    except (ValueError, TypeError):
        pass

    port_return = 0.0
    if pv > 0:
        try:
            if os.path.exists(DB_PATH):
                with get_db_conn() as _con:
                    first_row = _con.execute(
                        "SELECT portfolio_value FROM portfolio_snapshots "
                        "WHERE portfolio_value > 0 ORDER BY timestamp ASC LIMIT 1"
                    ).fetchone()
                if first_row:
                    first_val = float(first_row[0])
                    if first_val > 0:
                        port_return = (pv - first_val) / first_val * 100
        except Exception:
            pass

    bm = analytics_service.get_benchmark_comparison(
        portfolio_return_pct=port_return, period="YTD"
    )

    spy_ret = bm.get("spy_return", 0.0)
    vs_spy  = bm.get("vs_spy",    0.0)

    if vs_spy > 0:
        verdict   = f"AHEAD of S&P 500 by {vs_spy:+.1f} pts"
        bar_color = GAIN
        icon      = "🟢"
    elif vs_spy < 0:
        verdict   = f"BEHIND S&P 500 by {abs(vs_spy):.1f} pts"
        bar_color = LOSS
        icon      = "🔴"
    else:
        verdict   = "IN LINE with S&P 500"
        bar_color = NEURAL
        icon      = "🟡"

    return (
        f'<div style="background:{bar_color}18;border:1px solid {bar_color}44;'
        f'border-radius:8px;padding:10px 16px;margin-bottom:8px;'
        f'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">'
        f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">YTD vs Market</span>'
        f'<span style="font-weight:700;color:{bar_color};font-size:{FONT_VALUE};">'
        f'{icon} {verdict}</span>'
        f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">'
        f'You {port_return:+.1f}% &nbsp;·&nbsp; SPY {spy_ret:+.1f}%</span>'
        f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">Goal: beat S&P 500 by 10 pts/year</span>'
        f'</div>'
    )
