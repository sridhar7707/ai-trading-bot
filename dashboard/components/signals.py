"""Watchlist, signals tab, and trade timeline."""
from __future__ import annotations
import datetime
from loguru import logger
from dashboard.design_system import (
    BG, SURFACE, SURFACE2, BORDER, TEXT1, TEXT2, TEXT3,
    ACTION_BUY, ACTION_SELL, ACTION_TRIM,
    GAIN, LOSS, NEURAL,
    FONT_HERO, FONT_SECTION, FONT_VALUE, FONT_LABEL,
    WEIGHT_BOLD, WEIGHT_MEDIUM, WEIGHT_NORMAL,
    CARD_PADDING, SECTION_GAP,
    _card, _label, _section_title, _action_badge, _symbol,
    _metric_row, _divider, _empty_state, _section, _wrap,
    _sym, _badge, _num, _pnl, TH, TD, TD0,
)
import pandas as pd
from dashboard.data import get_data, _to_ct
from dashboard.charts import _FI_LABELS
from bot.core.error_logger import safe_render
_logger = logger

# ── Render: watchlist (open positions with live return vs avg cost) ────────────
@safe_render("Watchlist")
def render_watchlist() -> str:
    d        = get_data()
    open_pos = d["open_pos"]
    prices   = d["prices"]

    if not open_pos:
        return (f'<div class="nt nt-wrap">'
                f'{_section("👁","Watchlist","open positions · vs avg cost")}'
                f'{_card(_empty_state("💰", "Fully in cash", "Bot enters trades during market hours when signals align."))}</div>')

    rows  = ""
    items = list(open_pos.items())[:8]
    for i, (sym, pos) in enumerate(items):
        cur      = prices.get(sym, 0.0)
        shares   = pos["shares"]
        invested = pos["invested"]
        avg_cost = invested / shares if shares > 0 else 0
        chg_pct  = ((cur - avg_cost) / avg_cost * 100) if avg_cost > 0 and cur > 0 else 0.0
        arrow    = "↑" if chg_pct >= 0 else "↓"
        chg_c    = GAIN if chg_pct >= 0 else LOSS
        td = TD if i < len(items) - 1 else TD0
        rows += (
            f'<tr>'
            f'<td {td}>{_sym(sym)}</td>'
            f'<td {td}><span style="font-family:Courier New,monospace;font-weight:600;'
            f'color:{TEXT1};">${cur:.2f}</span></td>'
            f'<td {td}><span style="font-weight:700;font-size:{FONT_VALUE};color:{chg_c};">'
            f'{arrow} {chg_pct:+.1f}%</span></td>'
            f'</tr>'
        )
    table = _wrap(
        f'<table class="nt-tbl" style="width:100%"><thead><tr>'
        f'<th {TH}>Symbol</th><th {TH}>Price</th><th {TH}>vs Avg Cost</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    return (f'<div class="nt nt-wrap">'
            f'{_section("👁","Watchlist","vs avg cost · live")}{table}</div>')


# ── Render: signals tab (recent BUY signals with confidence + SHAP) ───────────
@safe_render("Signals")
def render_signals_tab() -> str:
    d    = get_data()
    buys = d.get("today_buy_signals", [])

    if not buys:
        _es = _empty_state("⚡", "No signals yet",
                           "The AI generates signals Mon–Fri 9:30am–4pm ET "
                           "when all entry gates pass.")
        return (f'<div class="nt nt-wrap">'
                f'{_section("⚡","AI Buy Signals","recent")}'
                f'{_card(_es)}'
                f'</div>')

    rows  = ""
    shown = buys[:20]
    for i, sig in enumerate(shown):
        ts      = sig.get("timestamp", "")
        sym     = sig.get("symbol", "—")
        price   = float(sig.get("price",          0.0) or 0.0)
        conf    = float(sig.get("ensemble_score",  0.0) or 0.0)
        regime  = str(sig.get("regime") or "—").replace("_", " ").title()
        drv_raw = sig.get("feature_drivers")
        driver_text = "—"
        try:
            import json as _j
            ds = _j.loads(drv_raw) if isinstance(drv_raw, str) else (drv_raw or [])
            parts = [
                f"{_FI_LABELS.get(f, f)}{'↑' if float(v) > 0 else '↓'}"
                for f, v in (ds or [])[:2]
            ]
            driver_text = " · ".join(parts) if parts else "—"
        except Exception as exc:
            logger.debug(f"parse_driver_text: {exc}")
        conf_pct = f"{conf*100:.0f}%" if conf > 0 else "—"
        conf_c   = GAIN if conf >= 0.75 else (NEURAL if conf >= 0.60 else TEXT2)
        td   = TD if i < len(shown) - 1 else TD0
        anim = f'style="animation:slideInRow .3s ease both;animation-delay:{i*0.04:.2f}s;"'
        rows += (
            f'<tr {anim}>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{_to_ct(ts)}</span></td>'
            f'<td {td}>{_sym(sym)}</td>'
            f'<td {td}>{_badge("BUY")}</td>'
            f'<td {td}><span style="font-family:Courier New,monospace;color:{TEXT1};">'
            f'${price:.2f}</span></td>'
            f'<td {td}><span style="font-weight:700;color:{conf_c};">{conf_pct}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{regime}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{driver_text}</span></td>'
            f'</tr>'
        )
    note = f"last {len(shown)} signals · confidence = XGBoost + LSTM + sentiment ensemble"
    help_block = (
        f'<div style="background:{BG};border-top:1px solid {BORDER};'
        f'padding:8px 14px;font-size:{FONT_LABEL};color:{TEXT2};line-height:1.7;">'
        f'<b>Confidence</b> ≥75% strong · 60–75% moderate · &lt;60% weak &nbsp;·&nbsp;'
        f'<b>Top Drivers</b> show which indicators pushed the AI to BUY &nbsp;·&nbsp;'
        f'<b>Regime</b> = macro trend when signal fired'
        f'</div>'
    )
    table_inner = (
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Time (CT)</th><th {TH}>Symbol</th>'
        f'<th {TH}>Signal</th><th {TH}>Entry</th>'
        f'<th {TH}>Confidence</th><th {TH}>Regime</th>'
        f'<th {TH}>Top Drivers</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>' + help_block
    )
    return (f'<div class="nt nt-wrap">'
            f'{_section("⚡","AI Buy Signals", note)}'
            f'{_wrap(table_inner)}</div>')


# ── Render: risk controls panel ──────────────────────────────────────────────
@safe_render("Risk Panel")
def render_risk_panel() -> str:
    d        = get_data()
    open_pos = d["open_pos"]
    prices   = d["prices"]
    vix      = d.get("vix", 0.0)
    df       = d["trades_df"]

    # Portfolio value as float
    pv = 0.0
    try:
        pv = float(d["portfolio"].replace("$", "").replace(",", "")) if d["portfolio"] != "—" else 0.0
    except Exception as exc:
        logger.debug(f"parse_portfolio_value render_risk_panel: {exc}")

    total_invested = sum(v["invested"] for v in open_pos.values())
    cash_pct = ((pv - total_invested) / pv * 100) if pv > 0 else 100.0

    # Max drawdown from portfolio history
    max_dd = 0.0
    if not df.empty and "portfolio_value" in df.columns:
        vals = df["portfolio_value"].dropna()
        if len(vals) > 1:
            peak  = vals.cummax()
            max_dd = float(((peak - vals) / peak.replace(0, float("nan"))).max()) * 100

    # Daily loss (today's sells, average pnl_pct)
    daily_pnl = 0.0
    if not df.empty:
        today_str  = str(datetime.date.today())
        sells_today = df[
            df["action"].str.startswith("SELL") & (df["action"] != "SELL_RECONCILE") &
            (df["date"].astype(str) == today_str)
        ]
        if not sells_today.empty and "pnl_pct" in sells_today.columns:
            daily_pnl = float(sells_today["pnl_pct"].mean()) * 100

    # Sector exposure
    sector_exp: dict[str, float] = {}
    for sym, pos in open_pos.items():
        cur = prices.get(sym, 0.0)
        val = pos["shares"] * cur if cur > 0 else pos["invested"]
        sector = _SECTOR_MAP.get(sym.upper(), "Other")
        sector_exp[sector] = sector_exp.get(sector, 0.0) + val
    total_eq = sum(sector_exp.values()) or 1.0
    sector_pcts = {s: v / total_eq * 100 for s, v in sorted(sector_exp.items(), key=lambda x: -x[1])}

    # Largest position concentration
    max_conc = 0.0
    if open_pos and pv > 0:
        for sym, pos in open_pos.items():
            cur = prices.get(sym, 0.0)
            val = pos["shares"] * cur if cur > 0 else pos["invested"]
            max_conc = max(max_conc, val / pv * 100)

    # Overall risk
    risk_pts = sum([vix > 25, max_dd > 8, cash_pct < 15, max_conc > 20])
    if risk_pts >= 3: overall_risk, risk_c = "High",   LOSS
    elif risk_pts >= 1: overall_risk, risk_c = "Medium", NEURAL
    else: overall_risk, risk_c = "Low", GAIN

    dd_c  = GAIN if max_dd < 5 else (NEURAL if max_dd < 12 else LOSS)
    dl_c  = GAIN if daily_pnl >= 0 else (NEURAL if daily_pnl > -2 else LOSS)
    cc_c  = GAIN if max_conc < 15 else (NEURAL if max_conc < 20 else LOSS)
    ca_c  = GAIN if cash_pct > 30 else (NEURAL if cash_pct > 15 else LOSS)

    cards = (
        f'<div class="nt-cards">'
        + _stat_card("Portfolio Risk",  overall_risk,       TEXT2, risk_c,
                "VIX + drawdown + concentration", 0.00)
        + _stat_card("Max Drawdown",    f"{max_dd:.1f}%",   TEXT2, dd_c,
                "Peak-to-trough all-time",        0.06)
        + _stat_card("Today's P&L",     f"{daily_pnl:+.2f}%", TEXT2, dl_c,
                "Realised from closed trades",    0.12)
        + _stat_card("Cash Reserve",    f"{cash_pct:.1f}%", TEXT2, ca_c,
                "Uninvested capital buffer",      0.18)
        + f'</div>'
    )

    sector_rows = ""
    for sector, pct in list(sector_pcts.items())[:5]:
        bar_c = LOSS if pct > 50 else (NEURAL if pct > 30 else GAIN)
        sector_rows += (
            f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0;">'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT2};width:70px;flex-shrink:0;">{sector}</span>'
            f'<div style="background:{BORDER};border-radius:2px;height:5px;flex:1;overflow:hidden;">'
            f'<div style="background:{bar_c};height:100%;width:{min(pct,100):.0f}%;"></div></div>'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT1};width:36px;text-align:right;">{pct:.0f}%</span>'
            f'</div>'
        )
    if not sector_rows:
        sector_rows = f'<div style="color:{TEXT2};font-size:{FONT_LABEL};">No open positions — fully in cash</div>'

    note = (f'Concentration: <span style="color:{cc_c};font-weight:700;">{max_conc:.1f}%</span>'
            f' largest position')
    return (f'<div class="nt nt-wrap">'
            f'{_section("🛡","Risk Controls","real-time")}'
            f'{cards}'
            f'<div style="background:{SURFACE};border:1px solid {BORDER};'
            f'border-radius:8px;padding:14px 16px;margin-top:8px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;letter-spacing:.8px;">Sector Exposure</div>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">{note}</div>'
            f'</div>'
            f'{sector_rows}</div></div>')


# ── Render: institutional metrics ─────────────────────────────────────────────
@safe_render("Institutional Metrics")
def render_institutional_metrics() -> str:
    d  = get_data()
    df = d["trades_df"]

    if df.empty or "portfolio_value" not in df.columns:
        msg = f'<div style="color:{TEXT2};text-align:center;padding:28px;font-size:{FONT_LABEL};">No trade history yet.</div>'
        return f'<div class="nt nt-wrap">{_section("📐","Institutional Metrics")}{_wrap(msg)}</div>'

    daily = (df.dropna(subset=["portfolio_value"])
               .groupby("date")["portfolio_value"].last()
               .reset_index()
               .sort_values("date"))
    daily.columns = ["date", "value"]

    if len(daily) < 3:
        msg = f'<div style="color:{TEXT2};text-align:center;padding:28px;font-size:{FONT_LABEL};">Need ≥ 3 days of history.</div>'
        return f'<div class="nt nt-wrap">{_section("📐","Institutional Metrics")}{_wrap(msg)}</div>'

    rets   = daily["value"].pct_change().dropna()
    mean_r = float(rets.mean())
    std_r  = float(rets.std())

    # Sharpe (annualised, 252 trading days)
    sharpe = (mean_r / std_r * (252 ** 0.5)) if std_r > 0 else 0.0

    # Sortino (downside std only)
    neg_rets = rets[rets < 0]
    down_std = float(neg_rets.std()) if len(neg_rets) > 1 else std_r
    sortino  = (mean_r / down_std * (252 ** 0.5)) if down_std > 0 else 0.0

    # Max drawdown
    vals  = daily["value"]
    peak  = vals.cummax()
    max_dd = float(((peak - vals) / peak.replace(0, float("nan"))).max())

    # CAGR
    n_days  = (pd.to_datetime(daily["date"].iloc[-1]) - pd.to_datetime(daily["date"].iloc[0])).days
    start_v = float(daily["value"].iloc[0])
    end_v   = float(daily["value"].iloc[-1])
    cagr    = ((end_v / start_v) ** (365.0 / n_days) - 1) if n_days > 0 and start_v > 0 else 0.0

    # Calmar
    calmar = (cagr / max_dd) if max_dd > 0 else 0.0

    # VaR 95% (1-day)
    var_95 = float(rets.quantile(0.05)) if len(rets) >= 5 else 0.0

    # Win rate
    sells    = df[df["action"].str.startswith("SELL") & (df["action"] != "SELL_RECONCILE")]
    win_rate = float((sells["pnl_pct"] > 0).sum() / len(sells)) if len(sells) > 0 else 0.0

    def _row(label, val_str, color, desc):
        return (
            f'<tr><td style="padding:10px 14px;border-bottom:1px solid {BORDER};'
            f'background:{SURFACE};color:{TEXT2};font-size:{FONT_LABEL};font-weight:600;">{label}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid {BORDER};'
            f'background:{SURFACE};font-family:-apple-system,monospace;'
            f'color:{color};font-weight:700;">{val_str}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid {BORDER};'
            f'background:{SURFACE};color:{TEXT2};font-size:{FONT_LABEL};">{desc}</td></tr>'
        )

    sh_c = GAIN if sharpe > 1 else (NEURAL if sharpe > 0.5 else LOSS)
    so_c = GAIN if sortino > 1.5 else (NEURAL if sortino > 0.8 else LOSS)
    dd_c = GAIN if max_dd < 0.05 else (NEURAL if max_dd < 0.12 else LOSS)
    ca_c = GAIN if calmar > 2 else (NEURAL if calmar > 1 else LOSS)
    vr_c = GAIN if var_95 > -0.02 else (NEURAL if var_95 > -0.04 else LOSS)
    wr_c = GAIN if win_rate > 0.55 else (NEURAL if win_rate > 0.45 else LOSS)

    rows = (
        _row("Sharpe Ratio",    f"{sharpe:.2f}",  sh_c, ">1.0 = good · >2.0 = excellent")
        + _row("Sortino Ratio", f"{sortino:.2f}", so_c, "Like Sharpe but penalises only downside vol")
        + _row("Max Drawdown",  f"{max_dd:.1%}",  dd_c, "Worst peak-to-trough in account history")
        + _row("CAGR",          f"{cagr:.1%}",    (GAIN if cagr > 0.15 else (NEURAL if cagr > 0 else LOSS)),
               "Compound Annual Growth Rate over tracked period")
        + _row("Calmar Ratio",  f"{calmar:.2f}",  ca_c, "CAGR ÷ max drawdown — higher is better")
        + _row("VaR (95%, 1d)", f"{var_95:.2%}",  vr_c, "Worst expected 1-day loss at 95% confidence")
        + _row("Win Rate",      f"{win_rate:.1%}", wr_c, "% of closed trades that returned a profit")
    )
    help_block = (
        f'<div style="background:{BG};border-top:1px solid {BORDER};'
        f'padding:8px 14px;font-size:{FONT_LABEL};color:{TEXT2};line-height:1.6;">'
        f'Metrics computed from all trade history since launch. '
        f'Short history (&lt;30 days) may produce unreliable Sharpe / Sortino estimates.'
        f'</div>'
    )
    n_str = f"{n_days} days of history" if n_days > 0 else "—"
    table = _wrap(f'<table class="nt-tbl" style="width:100%">{rows}</table>' + help_block)
    return (f'<div class="nt nt-wrap">'
            f'{_section("📐","Institutional Metrics", n_str)}{table}</div>')


# ── Render: AI decision feed (trade timeline) ────────────────────────────────
@safe_render("Timeline")
def render_timeline() -> str:
    d  = get_data()
    df = d["trades_df"]
    if df.empty:
        empty = (f'<div style="color:{TEXT2};text-align:center;padding:40px;font-size:{FONT_VALUE};">'
                 f'No decisions yet. The AI trades Mon–Fri 9:30am–4pm ET.</div>')
        return f'<div class="nt nt-wrap">{_section("🕐","AI Decision Feed","live")}{_wrap(empty)}</div>'

    recent = df.tail(30).iloc[::-1]
    items  = ""
    for i, (_, row) in enumerate(recent.iterrows()):
        action  = str(row.get("action", ""))
        sym     = str(row.get("symbol", "—"))
        ts      = row.get("timestamp", "")
        conf    = float(row.get("ensemble_score",  0.0) or 0.0)
        regime  = str(row.get("regime") or "").replace("_", " ").title()
        sent    = float(row.get("sentiment_score", 0.0) or 0.0)
        pnl     = float(row.get("pnl_pct",         0.0) or 0.0)
        drv_raw = row.get("feature_drivers")
        is_last = i == len(recent) - 1
        dot_c   = GAIN if action == "BUY" else LOSS

        ts_full  = _to_ct(ts)
        time_lbl = ts_full[11:16] if len(ts_full) >= 16 else ts_full[:5]
        tz_lbl   = ts_full[17:20] if len(ts_full) >= 20 else ""
        date_lbl = ts_full[:10]

        if action == "BUY":
            parts = []
            if conf > 0:        parts.append(f"Confidence {conf*100:.0f}%")
            if sent > 0.05:     parts.append("Positive sentiment")
            elif sent < -0.05:  parts.append("Negative sentiment")
            if regime:          parts.append(regime)
            try:
                import json as _j
                ds  = _j.loads(drv_raw) if isinstance(drv_raw, str) else (drv_raw or [])
                pos = [(f, float(v)) for f, v in (ds or []) if float(v) > 0]
                if pos:
                    best = max(pos, key=lambda x: x[1])
                    w = _WHY_MAP.get(best[0])
                    parts.append(w[0] if w else _FI_LABELS.get(best[0], best[0]))
            except Exception as exc:
                logger.debug(f"parse_detail_parts: {exc}")
            detail = " · ".join(parts)
        else:
            reason  = _SELL_REASON.get(action, "Exit")
            pnl_str = f"{pnl:+.1%}" if pnl != 0 else ""
            detail  = f"{reason} · {pnl_str}" if pnl_str else reason

        conf_badge = ""
        if action == "BUY" and conf > 0:
            c_c = GAIN if conf >= 0.75 else (NEURAL if conf >= 0.60 else TEXT2)
            conf_badge = (f'<span style="font-size:{FONT_LABEL};color:{c_c};font-weight:700;">'
                          f'{conf*100:.0f}%</span>')

        line = f'border-bottom:1px solid {BORDER};' if not is_last else ''
        connector = (f'<div style="width:1px;flex:1;background:{BORDER};min-height:14px;"></div>'
                     if not is_last else '')
        items += (
            f'<div style="display:flex;gap:14px;padding:10px 0;{line}">'
            f'<div style="flex-shrink:0;width:58px;text-align:right;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT1};font-family:monospace;font-weight:600;">{time_lbl}</div>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">{tz_lbl}</div>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">{date_lbl}</div>'
            f'</div>'
            f'<div style="display:flex;flex-direction:column;align-items:center;padding-top:4px;">'
            f'<div style="width:10px;height:10px;border-radius:50%;background:{dot_c};flex-shrink:0;'
            f'box-shadow:0 0 6px {dot_c}44;"></div>'
            f'{connector}'
            f'</div>'
            f'<div style="flex:1;min-width:0;">'
            f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:3px;">'
            f'{_badge(action)}{_sym(sym)}{conf_badge}'
            f'</div>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};white-space:nowrap;overflow:hidden;'
            f'text-overflow:ellipsis;">{detail}</div>'
            f'</div></div>'
        )
    return (f'<div class="nt nt-wrap">'
            f'{_section("🕐","AI Decision Feed",f"last {len(recent)} decisions · newest first")}'
            f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;padding:0 16px;">'
            f'{items}</div></div>')


# ── Render: investor view (plain-language Models tab) ────────────────────────
