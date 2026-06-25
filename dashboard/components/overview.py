"""Overview panel: metrics hero, dashboard hero, portfolio health."""
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
from dashboard.data import get_data, _now_ct, _market_status
from dashboard.builders import build_health_vm
from bot.core.error_logger import safe_render, timed, log_exception
_logger = logger

# ── Render: metrics ───────────────────────────────────────────────────────────
@safe_render("Metrics")
def render_metrics() -> str:
    d = get_data()
    open_syms      = d["open_pos"]
    prices         = d["prices"]
    total_invested = sum(v["invested"] for v in open_syms.values())
    total_cur      = sum(v["shares"] * prices.get(s, 0.0) for s, v in open_syms.items())
    total_pnl      = total_cur - total_invested
    pnl_pct_all    = (total_pnl / total_invested * 100) if total_invested > 0 else 0.0

    pnl_str    = f"${total_pnl:+,.2f}" if total_invested > 0 else "—"
    pnl_sub    = f"{pnl_pct_all:+.2f}% on capital" if total_invested > 0 else "no open positions"
    pnl_color  = GAIN if total_pnl >= 0 else LOSS
    pnl_accent = GAIN_BD if total_pnl >= 0 else LOSS_BD

    invested_str = f"${total_invested:,.2f}" if total_invested > 0 else "—"

    r_lower = d["regime_raw"].lower()
    if any(x in r_lower for x in ["bull", "trending up"]):
        r_color, r_accent = GAIN, GAIN_BD
    elif any(x in r_lower for x in ["bear", "trending down"]):
        r_color, r_accent = LOSS, LOSS_BD
    else:
        r_color, r_accent = NEURAL, NEURAL_BD

    sell_count = d["sell_count"]
    win_count  = d["win_count"]
    win_rate   = (win_count / sell_count * 100) if sell_count > 0 else 0.0
    wr_str     = f"{win_rate:.1f}%" if sell_count > 0 else "—"
    wr_color   = GAIN if win_rate >= 50 else (LOSS if sell_count > 0 else TEXT2)
    wr_accent  = GAIN_BD if win_rate >= 50 else (LOSS_BD if sell_count > 0 else BORDER)

    open_count = len(open_syms)
    mkt_label, mkt_color = _market_status()

    # ── Hero: large portfolio value (Robinhood-style focal point) ────────────
    portfolio_val = d["portfolio"]
    pnl_sign      = "+" if total_pnl >= 0 else ""
    hero_chg      = (f'{pnl_sign}${total_pnl:,.2f} ({pnl_pct_all:+.2f}%)'
                     if total_invested > 0 else "No open positions")
    hero = (
        f'<div class="nt-hero">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
        f'letter-spacing:.8px;margin-bottom:6px;">Alpaca Paper Account Balance</div>'
        f'<div class="nt-hero-val">{portfolio_val}</div>'
        f'<div class="nt-hero-chg" style="color:{pnl_color};">{hero_chg}</div>'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};margin-top:4px;">'
        f'Unrealized gain / loss on open positions vs. what the bot paid</div>'
        f'</div>'
    )

    status = (
        f'<div class="nt-status">'
        f'<span style="color:{TEXT2};font-size:{FONT_LABEL};">'
        f'Updated &nbsp;<strong style="color:{TEXT1};">{_now_ct()}</strong></span>'
        f'<span style="display:inline-flex;align-items:center;gap:5px;">'
        f'<span style="width:6px;height:6px;background:{mkt_color};border-radius:50%;'
        f'display:inline-block;"></span>'
        f'<span style="color:{mkt_color};font-weight:600;font-size:{FONT_LABEL};">'
        f'{mkt_label}</span></span>'
        f'<div style="height:2px;width:100px;background:{BORDER};border-radius:1px;">'
        f'<div style="height:100%;width:100%;background:{PRIMARY};border-radius:1px;'
        f'animation:countdown 60s linear forwards;"></div></div>'
        f'<span style="color:{TEXT2};font-size:{FONT_LABEL};">60s refresh</span>'
        f'</div>'
    )

    legend = (
        f'<div style="display:flex;gap:18px;padding:4px 2px 8px;font-size:{FONT_LABEL};color:{TEXT2};">'
        f'<span><span style="color:{GAIN};">●</span> Gain / Bull regime</span>'
        f'<span><span style="color:{LOSS};">●</span> Loss / Bear regime</span>'
        f'<span><span style="color:{NEURAL};">●</span> Neutral / Ranging</span>'
        f'<span style="margin-left:auto;font-style:italic;">Paper money — no real funds at risk</span>'
        f'</div>'
    )

    row1 = (
        f'<div class="nt-cards">'
        + _stat_card("Unrealized P&amp;L",  pnl_str,                pnl_color, pnl_color,
                "Open trade gain/loss vs. cost basis",         0.00)
        + _stat_card("Total Invested",      invested_str,            TEXT2,     TEXT1,
                "Capital currently deployed in open trades",   0.06)
        + _stat_card("Market Regime",       d["regime_raw"].title(), TEXT2,     r_color,
                "AI-detected trend — drives position sizing",  0.12)
        + _stat_card("Market Session",      mkt_label,               TEXT2,     mkt_color,
                "NYSE/NASDAQ open 9:30am–4pm ET, Mon–Fri",    0.18)
        + f'</div>'
    )

    row2 = (
        f'<div class="nt-cards">'
        + _stat_card("Open Positions", str(open_count),
                TEXT2, TEXT1,
                f"Unique stocks held now (max 8 allowed)", 0.24)
        + _stat_card("Win Rate",       wr_str,
                TEXT2, wr_color,
                f"% of closed trades that made money · {win_count}/{sell_count}", 0.30)
        + _stat_card("Total Trades",   str(d["total_trades"]),
                TEXT2, TEXT1, "All BUY + SELL orders since launch", 0.36)
        + _stat_card("Buys / Sells",   f"{d['buy_count']} / {d['sell_count']}",
                TEXT2, TEXT1, "Entry orders vs. exit orders placed", 0.42)
        + f'</div>'
    )

    return f'<div class="nt nt-wrap">{hero}{status}{legend}{row1}{row2}</div>'


# ── Render: dashboard hero (Bloomberg-style 4-pack + status bar) ─────────────
@safe_render("Dashboard Hero")
def render_dashboard_hero() -> str:
    d = get_data()
    open_syms = d["open_pos"]
    prices    = d["prices"]
    total_invested = sum(v["invested"] for v in open_syms.values())
    total_cur      = sum(v["shares"] * prices.get(s, 0.0) for s, v in open_syms.items())
    total_pnl      = total_cur - total_invested
    pnl_pct_all    = (total_pnl / total_invested * 100) if total_invested > 0 else 0.0
    pnl_color      = GAIN if total_pnl >= 0 else LOSS
    portfolio_val  = d["portfolio"]
    pnl_sign       = "+" if total_pnl >= 0 else ""
    hero_chg       = (f'{pnl_sign}${total_pnl:,.2f} ({pnl_pct_all:+.2f}%)'
                      if total_invested > 0 else "No open positions")

    avg_conf   = d.get("avg_confidence", 0.0)
    conf_str   = f"{avg_conf*100:.0f}%" if avg_conf > 0 else "—"
    conf_color = GAIN if avg_conf >= 0.75 else (NEURAL if avg_conf >= 0.60 else TEXT2)

    vix = d.get("vix", 0.0)
    vix_str = f"{vix:.1f}" if vix > 0 else "—"
    if vix == 0: vix_color = TEXT2
    elif vix < 15: vix_color = GAIN
    elif vix < 25: vix_color = NEURAL
    else: vix_color = LOSS

    mkt_label, mkt_color = _market_status()

    def _big(label, value, sub, color):
        return (
            f'<div class="nt-card" style="padding:20px 18px;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
            f'letter-spacing:.8px;margin-bottom:10px;">{label}</div>'
            f'<div style="font-size:{FONT_HERO};font-weight:700;letter-spacing:-1.5px;'
            f'color:{color};line-height:1;">{value}</div>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};margin-top:6px;">{sub}</div>'
            f'</div>'
        )

    # ── Portfolio Health Score ──────────────────────────────────────────────
    pv_float = 0.0
    try:
        pv_float = float(d["portfolio"].replace("$", "").replace(",", "")) if d["portfolio"] != "—" else 0.0
    except Exception as exc:
        logger.debug(f"parse_portfolio_value render_portfolio_health_hero: {exc}")

    cash_pct_h = ((pv_float - total_invested) / pv_float * 100) if pv_float > 0 else 100.0

    max_dd_h = 0.0
    df_h = d["trades_df"]
    if not df_h.empty and "portfolio_value" in df_h.columns:
        vals_h = df_h["portfolio_value"].dropna()
        if len(vals_h) > 1:
            peak_h  = vals_h.cummax()
            max_dd_h = float(((peak_h - vals_h) / peak_h.replace(0, float("nan"))).max()) * 100

    max_conc_h = 0.0
    if open_syms and pv_float > 0:
        for _s, _p in open_syms.items():
            _cur = d["prices"].get(_s, 0.0)
            _val = _p["shares"] * _cur if _cur > 0 else _p["invested"]
            max_conc_h = max(max_conc_h, _val / pv_float * 100)

    _vix_pts  = 25 if vix < 15  else (15 if vix < 25  else 5)
    _cash_pts = 25 if cash_pct_h > 30 else (15 if cash_pct_h > 15 else 5)
    _conc_pts = 25 if max_conc_h < 15 else (15 if max_conc_h < 25 else 5)
    _dd_pts   = 25 if max_dd_h < 3  else (15 if max_dd_h < 8   else 5)
    health    = _vix_pts + _cash_pts + _conc_pts + _dd_pts

    health_c = GAIN if health >= 75 else (NEURAL if health >= 50 else LOSS)

    _components = [
        ("VIX",          _vix_pts),
        ("Cash Reserve", _cash_pts),
        ("Concentration", _conc_pts),
        ("Drawdown",     _dd_pts),
    ]
    weakest_name, weakest_pts = min(_components, key=lambda x: x[1])
    if weakest_pts == 25:
        weak_sub = "All risk factors look healthy"
    else:
        weak_sub = f"⚠ {weakest_name} is your biggest risk"

    health_card = (
        f'<div class="nt-card" style="padding:20px 18px;">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
        f'letter-spacing:.8px;margin-bottom:10px;">Portfolio Health</div>'
        f'<div style="font-size:{FONT_HERO};font-weight:700;letter-spacing:-1.5px;'
        f'color:{health_c};line-height:1;">{health}<span style="font-size:{FONT_VALUE};'
        f'color:{TEXT2};font-weight:400;">/100</span></div>'
        f'<div style="margin:8px 0 6px;background:{BORDER};border-radius:3px;height:4px;">'
        f'<div style="background:{health_c};height:100%;width:{health}%;'
        f'border-radius:3px;transition:width .4s;"></div></div>'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">{weak_sub}</div>'
        f'</div>'
    )

    val_color = pnl_color if total_invested > 0 else TEXT1
    cards = (
        f'<div class="nt-cards">'
        + _big("Portfolio Value",  portfolio_val,     f"Unrealized: {hero_chg}", val_color)
        + _big("Open Positions",   str(len(open_syms)), "Stocks held now (max 8)",  TEXT1)
        + _big("AI Confidence",    conf_str,          "Avg signal strength · last 5 buys", conf_color)
        + _big("VIX",              vix_str,           "Fear gauge · <15 calm · >30 fear",  vix_color)
        + health_card
        + f'</div>'
    )

    status = (
        f'<div class="nt-status">'
        f'<span style="color:{TEXT2};font-size:{FONT_LABEL};">'
        f'Updated &nbsp;<strong style="color:{TEXT1};">{_now_ct()}</strong></span>'
        f'<span style="display:inline-flex;align-items:center;gap:5px;">'
        f'<span style="width:6px;height:6px;background:{mkt_color};border-radius:50%;'
        f'display:inline-block;"></span>'
        f'<span style="color:{mkt_color};font-weight:600;font-size:{FONT_LABEL};">'
        f'{mkt_label}</span></span>'
        f'<span style="color:{TEXT2};font-size:{FONT_LABEL};">60s refresh &nbsp;·&nbsp; Paper money only</span>'
        f'</div>'
    )
    return f'<div class="nt nt-wrap">{cards}{status}</div>'


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
    conf_str = f"{avg_conf*100:.0f}%" if avg_conf > 0 else "—"
    conf_c   = GAIN if avg_conf >= 0.75 else (NEURAL if avg_conf >= 0.60 else TEXT2)
    vix      = d.get("vix", 0.0)
    vix_str  = f"{vix:.1f}" if vix > 0 else "—"
    vix_c    = GAIN if vix < 15 else (NEURAL if vix < 25 else LOSS)

    stats_row = (
        f'<div style="display:flex;flex-wrap:wrap;border-top:1px solid {BORDER};margin-top:10px;">'
        + _stat("Portfolio", d.get("portfolio", "—"), TEXT1)
        + _stat("P&L", hero_chg, pnl_c)
        + _stat("Positions", str(len(open_pos)), TEXT1)
        + _stat("AI Conf.", conf_str, conf_c)
        + _stat("VIX", vix_str, vix_c)
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
        f'<span style="color:{TEXT2};font-size:{FONT_LABEL};">60s refresh</span>'
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
        raw = d.get("portfolio", "—")
        if raw != "—":
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
