"""Risk panel and market intelligence."""
from __future__ import annotations
import datetime
from loguru import logger
from dashboard.design_system import (
    BG, SURFACE, SURFACE2, BORDER, TEXT1, TEXT2, TEXT3,
    ACTION_BUY, ACTION_SELL, ACTION_TRIM, ACTION_HOLD, ACTION_WATCH,
    GAIN, LOSS, NEURAL,
    FONT_HERO, FONT_SECTION, FONT_VALUE, FONT_LABEL,
    WEIGHT_BOLD, WEIGHT_MEDIUM, WEIGHT_NORMAL,
    CARD_PADDING, SECTION_GAP,
    _card, _label, _section_title, _action_badge, _symbol,
    _metric_row, _divider, _empty_state, _section, _wrap, _stat_card,
    TH, TD, TD0,
)
from dashboard.data import get_data, _now_ct
from bot.core.error_logger import safe_render
_logger = logger

_SECTOR_MAP: dict[str, str] = {
    "NVDA": "Tech",    "MSFT": "Tech",    "AAPL": "Tech",  "GOOGL": "Tech",
    "META": "Tech",    "AMZN": "Consumer","TSLA": "Auto",  "AMD":   "Tech",
    "INTC": "Tech",    "QCOM": "Tech",    "MU":   "Tech",  "AVGO":  "Tech",
    "CRM":  "Tech",    "NOW":  "Tech",    "SNOW": "Tech",  "PLTR":  "Tech",
    "JPM":  "Finance", "BAC":  "Finance", "GS":   "Finance","MS":   "Finance",
    "XOM":  "Energy",  "CVX":  "Energy",  "SPY":  "Index", "QQQ":   "Index",
}

_SELL_REASON: dict[str, str] = {
    "SELL":           "Target exit",
    "SELL_STOP":      "Stop-loss triggered",
    "SELL_TP":        "Take-profit hit",
    "SELL_TRAIL":     "Trailing stop hit",
    "SELL_TRIM":      "Oversize trim",
    "SELL_TIME":      "Time-based exit",
    "SELL_ENSEMBLE":  "Signal deteriorated",
    "SELL_RECONCILE": "Reconciled on startup",
}


def _risk_level(vix: float, regime: str) -> tuple[str, str]:
    r = regime.lower()
    if vix > 30 or "bear" in r:
        return "High", LOSS
    elif vix > 20 or any(x in r for x in ["ranging", "neutral"]):
        return "Medium", NEURAL
    return "Low", GAIN


# ── Render: market intelligence (VIX / regime / confidence / sentiment) ───────
@safe_render("Market Intelligence")
def render_market_intelligence() -> str:
    d        = get_data()
    vix      = d.get("vix", 0.0)
    regime   = d.get("regime_raw", "—")
    avg_conf = d.get("avg_confidence", 0.0)
    sent     = d.get("sentiment_avg", 0.0)

    if vix == 0: vix_label, vix_color = "N/A", TEXT2
    elif vix < 15: vix_label, vix_color = "Low Fear", GAIN
    elif vix < 25: vix_label, vix_color = "Moderate", NEURAL
    elif vix < 35: vix_label, vix_color = "High Fear", LOSS
    else: vix_label, vix_color = "Extreme Fear", LOSS

    r_lower = regime.lower()
    if any(x in r_lower for x in ["bull", "trending up"]):   r_color = GAIN
    elif any(x in r_lower for x in ["bear", "trending down"]): r_color = LOSS
    else: r_color = NEURAL

    conf_color = GAIN if avg_conf >= 0.75 else (NEURAL if avg_conf >= 0.60 else TEXT2)

    if sent == 0: sent_label, sent_color = "No data", TEXT2
    elif sent > 0.05: sent_label, sent_color = "Positive", GAIN
    elif sent < -0.05: sent_label, sent_color = "Negative", LOSS
    else: sent_label, sent_color = "Neutral", NEURAL

    cards = (
        f'<div class="nt-cards">'
        + _stat_card("VIX", f"{vix:.1f}" if vix > 0 else "—",
                TEXT2, vix_color, f"{vix_label} · <15=calm, >30=fear", 0.00)
        + _stat_card("Market Regime", regime.replace("_", " ").title(),
                TEXT2, r_color, "AI-detected trend · drives position size", 0.06)
        + _stat_card("Signal Strength", f"{avg_conf*100:.0f}%" if avg_conf > 0 else "—",
                TEXT2, conf_color, "Avg confidence · last 5 buy signals", 0.12)
        + _stat_card("News Sentiment", sent_label,
                TEXT2, sent_color, "FinBERT score · recent headlines", 0.18)
        + f'</div>'
    )
    return f'<div class="nt nt-wrap">{_section("📡","Market Intelligence","live")}{cards}</div>'


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
