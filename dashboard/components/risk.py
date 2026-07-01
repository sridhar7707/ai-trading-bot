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
    _sym, _badge, TH, TD, TD0,
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
    regime   = d.get("regime_raw", "&mdash;")
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

    if sent > 0.05:    sent_label, sent_color = "Positive", GAIN
    elif sent < -0.05: sent_label, sent_color = "Negative", LOSS
    elif sent != 0.0:  sent_label, sent_color = "Neutral",  NEURAL
    else:              sent_label, sent_color = "No data",   TEXT2

    cards = (
        f'<div class="nt-cards">'
        + _stat_card("VIX", f"{vix:.1f}" if vix > 0 else "&mdash;",
                TEXT2, vix_color, f"{vix_label} · <15=calm, >30=fear", 0.00)
        + _stat_card("Market Regime", regime.replace("_", " ").title(),
                TEXT2, r_color, "AI-detected trend · drives position size", 0.06)
        + _stat_card("Signal Strength", f"{avg_conf*100:.0f}%" if avg_conf > 0 else "&mdash;",
                TEXT2, conf_color, "Avg confidence · last 5 buy signals", 0.12)
        + _stat_card("News Sentiment", sent_label,
                TEXT2, sent_color, "FinBERT score · recent headlines", 0.18)
        + f'</div>'
    )
    return f'<div class="nt nt-wrap">{_section("📡","Market Intelligence","live")}{cards}</div>'


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
        pv = float(d["portfolio"].replace("$", "").replace(",", "")) if d["portfolio"] != "&mdash;" else 0.0
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
        sector_rows = f'<div style="color:{TEXT2};font-size:{FONT_LABEL};">No open positions &mdash; fully in cash</div>'

    note = (f'Concentration: <span style="color:{cc_c};font-weight:700;">{max_conc:.1f}%</span>'
            f' largest position')

    # ── Risk Limits Tracker ───────────────────────────────────────────────────
    # Shows current value vs hard limit for each guardrail — green=headroom, red=at limit
    MAX_POS      = 8       # max simultaneous positions
    CASH_FLOOR   = 15.0    # bot stops buying if cash < 15%
    VIX_CAP      = 28.0    # bot reduces size above VIX 28
    DAILY_LOSS_CAP = 5.0   # daily loss limit in % (bot halts buys if hit)

    pos_count   = len(open_pos)
    pos_pct     = pos_count / MAX_POS * 100
    cash_used   = max(0.0, CASH_FLOOR - cash_pct) / CASH_FLOOR * 100  # how close to floor
    vix_pct     = min(vix / VIX_CAP * 100, 100) if vix > 0 else 0
    loss_pct    = min(abs(daily_pnl) / DAILY_LOSS_CAP * 100, 100) if daily_pnl < 0 else 0

    pos_c  = GAIN if pos_count < 6 else (NEURAL if pos_count < 8 else LOSS)
    csh_c  = GAIN if cash_pct > 30 else (NEURAL if cash_pct > CASH_FLOOR else LOSS)
    vix_c2 = GAIN if vix < 20 else (NEURAL if vix < VIX_CAP else LOSS)
    dl_c2  = GAIN if daily_pnl >= -2 else (NEURAL if daily_pnl > -DAILY_LOSS_CAP else LOSS)

    def _limit_bar(label, current_str, limit_str, fill_pct, bar_c, note_str):
        safe_pct = min(max(fill_pct, 0), 100)
        return (
            f'<div style="margin-bottom:10px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:3px;">'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">{label}</span>'
            f'<span style="font-size:{FONT_LABEL};color:{bar_c};font-weight:600;">'
            f'{current_str} <span style="color:{TEXT2};font-weight:400;">/ {limit_str}</span></span>'
            f'</div>'
            f'<div style="background:{BORDER};border-radius:2px;height:4px;">'
            f'<div style="background:{bar_c};height:100%;width:{safe_pct:.0f}%;border-radius:2px;transition:width .4s;"></div>'
            f'</div>'
            f'<div style="font-size:10px;color:{TEXT2};margin-top:2px;">{note_str}</div>'
            f'</div>'
        )

    limits_html = (
        _limit_bar("Open positions",   f"{pos_count}",        f"{MAX_POS} max",
                   pos_pct,  pos_c,
                   "Bot stops buying new stocks when full" if pos_count >= MAX_POS else "Room for more buys")
        + _limit_bar("Cash reserve",   f"{cash_pct:.0f}%",   f"{CASH_FLOOR:.0f}% floor",
                   max(0, (CASH_FLOOR - cash_pct) / (100 - CASH_FLOOR) * 100 + 50),
                   csh_c,
                   "Bot holds cash buffer to avoid over-investing")
        + _limit_bar("Fear gauge (VIX)", f"{vix:.1f}" if vix > 0 else "N/A", f"{VIX_CAP:.0f} cap",
                   vix_pct, vix_c2,
                   "Bot reduces position size when markets are fearful (VIX > 20)")
        + _limit_bar("Today's losses",  f"{daily_pnl:+.1f}%", f"{DAILY_LOSS_CAP:.0f}% limit",
                   loss_pct, dl_c2,
                   "Bot halts all new buys if daily losses hit 5% — protects against bad days")
    )

    return (f'<div class="nt nt-wrap">'
            f'{_section("🛡","Risk Controls","real-time — bot guardrails")}'
            f'{cards}'
            f'<div style="background:{SURFACE};border:1px solid {BORDER};'
            f'border-radius:8px;padding:14px 16px;margin-top:8px;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
            f'letter-spacing:.8px;margin-bottom:10px;">Bot Safety Limits</div>'
            f'{limits_html}'
            f'</div>'
            f'<div style="background:{SURFACE};border:1px solid {BORDER};'
            f'border-radius:8px;padding:14px 16px;margin-top:8px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;letter-spacing:.8px;">Sector Exposure</div>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">{note}</div>'
            f'</div>'
            f'{sector_rows}</div></div>')


# ── Render: institutional metrics ─────────────────────────────────────────────
