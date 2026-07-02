"""Morning Executive Brief — daily portfolio summary panel (req 11.1)."""
from __future__ import annotations
import datetime
from loguru import logger
from dashboard.design_system import (
    BG, SURFACE, SURFACE2, BORDER, TEXT1, TEXT2, TEXT3,
    ACTION_BUY, ACTION_SELL, ACTION_HOLD,
    PRIMARY, GAIN, LOSS, NEURAL,
    FONT_HERO, FONT_SECTION, FONT_VALUE, FONT_LABEL,
    WEIGHT_BOLD, _section, _wrap, _card, _metric_row, _stat_card,
)
from dashboard.data import get_data, get_db_conn, DB_PATH, safe_query, _now_ct, _market_status
from bot.core.error_logger import safe_render, timed, log_exception
from bot.core.recommendation_engine import get_portfolio_health, get_sell_analysis
import os

_logger = logger


def _portfolio_health_score() -> tuple[int, str]:
    """Return (0-100 score, label). Uses cached health VM if available."""
    try:
        from dashboard.builders import build_health_vm
        vm = build_health_vm()
        return int(vm.total), vm.grade_label
    except Exception:
        return 0, "Unknown"


def _market_sentiment() -> str:
    """Classify current market as Bullish / Neutral / Cautious / Bearish from VIX + SPY."""
    d = get_data()
    vix = d.get("vix", 0.0)
    regime = d.get("regime", "")
    macro = d.get("macro_score", 0.5)
    if vix > 0:
        if vix < 15 and macro >= 0.65:
            return "Bullish"
        if vix < 20 and macro >= 0.50:
            return "Neutral"
        if vix < 28:
            return "Cautious"
        return "Bearish"
    if "trending up" in str(regime).lower():
        return "Bullish"
    if "trending down" in str(regime).lower():
        return "Bearish"
    return "Neutral"


def _cash_pct() -> float:
    """Available cash as % of total portfolio value."""
    d = get_data()
    try:
        pv = float(d.get("portfolio", "0").replace("$", "").replace(",", ""))
        cash = d.get("cash", 0.0)
        if pv > 0:
            return round(cash / pv * 100, 1)
    except Exception:
        pass
    return 0.0


def _opportunity_and_risk() -> tuple[str, str]:
    """Return (highest_opportunity_symbol, highest_risk_symbol) from open positions."""
    d = get_data()
    open_pos = d.get("open_pos", {})
    if not open_pos:
        return "—", "—"

    # Best opportunity: highest ensemble score in recent signals
    scores: dict[str, float] = {}
    pnls: dict[str, float] = {}
    prices = d.get("prices", {})
    for sym, pos in open_pos.items():
        cur = prices.get(sym, 0.0)
        invested = pos.get("invested", 0.0)
        shares = pos.get("shares", 0.0)
        pnl_pct = ((shares * cur - invested) / invested * 100) if invested > 0 and cur > 0 else 0.0
        pnls[sym] = pnl_pct

    # Fetch latest ensemble scores from signal_log
    try:
        if os.path.exists(DB_PATH):
            with get_db_conn() as con:
                rows = con.execute(
                    "SELECT sl.symbol, sl.ensemble_score FROM signal_log sl "
                    "INNER JOIN (SELECT symbol, MAX(id) mid FROM signal_log GROUP BY symbol) m "
                    "ON sl.id=m.mid"
                ).fetchall()
            for sym, sc in rows:
                if sym in open_pos:
                    scores[sym] = float(sc or 0.0)
    except Exception:
        pass

    if scores:
        best_sym = max(scores, key=lambda s: scores[s])
    else:
        best_sym = max(pnls, key=lambda s: pnls[s]) if pnls else "—"

    # Highest risk: worst pnl or sell analysis
    worst_sym = min(pnls, key=lambda s: pnls[s]) if pnls else "—"
    return best_sym, worst_sym


def _action_items() -> list[str]:
    """Return up to 5 AI-generated action items from sell analysis."""
    d = get_data()
    open_pos = d.get("open_pos", {})
    items: list[str] = []
    for sym in list(open_pos.keys())[:8]:
        try:
            sa = get_sell_analysis(sym, d)
            rec = sa.get("recommendation", "HOLD")
            if rec in ("EXIT", "SELL"):
                items.append(f"Sell {sym}")
            elif rec == "TRIM":
                pct = sa.get("trim_amount_pct", 25)
                items.append(f"Trim {sym} {pct}%")
        except Exception:
            pass
    return items[:5]


def _no_action_count(action_items: list[str]) -> int:
    d = get_data()
    total = len(d.get("open_pos", {}))
    return max(0, total - len(action_items))


@timed(_logger)
@safe_render("Morning Brief")
def render_morning_brief() -> str:
    d = get_data()
    mkt_label, mkt_color = _market_status()
    now_ct = _now_ct()

    health_score, health_label = _portfolio_health_score()
    sentiment = _market_sentiment()
    cash_pct = _cash_pct()
    opp_sym, risk_sym = _opportunity_and_risk()
    action_items = _action_items()
    no_action = _no_action_count(action_items)

    # ── Greeting ──────────────────────────────────────────────────────────────
    hour = datetime.datetime.now().hour
    greeting = "Good Morning" if hour < 12 else ("Good Afternoon" if hour < 17 else "Good Evening")
    date_str = datetime.date.today().strftime("%B %d, %Y")

    mkt_closed = "closed" in mkt_label.lower() or "pre" in mkt_label.lower()
    closed_badge = (
        f'<span style="background:{NEURAL}22;border:1px solid {NEURAL};color:{NEURAL};'
        f'padding:2px 8px;border-radius:4px;font-size:{FONT_LABEL};font-weight:700;">Market Closed</span>'
        if mkt_closed else ""
    )

    # ── Sentiment color ───────────────────────────────────────────────────────
    sent_color = {
        "Bullish": GAIN, "Neutral": NEURAL,
        "Cautious": "#f59e0b", "Bearish": LOSS,
    }.get(sentiment, NEURAL)

    # ── Health color ──────────────────────────────────────────────────────────
    health_color = GAIN if health_score >= 80 else (NEURAL if health_score >= 60 else LOSS)

    # ── Top scorecards row ────────────────────────────────────────────────────
    def _brief_card(label: str, val: str, color: str, sublabel: str = "") -> str:
        sub = (f'<div style="font-size:{FONT_LABEL};color:{TEXT3};margin-top:2px;">{sublabel}</div>'
               if sublabel else "")
        return (
            f'<div style="flex:1;text-align:center;padding:16px 12px;background:{SURFACE};'
            f'border-radius:8px;border:1px solid {BORDER};">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
            f'letter-spacing:.8px;margin-bottom:6px;">{label}</div>'
            f'<div style="font-size:{FONT_SECTION};font-weight:800;color:{color};">{val}</div>'
            f'{sub}'
            f'</div>'
        )

    top_row = (
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;">'
        + _brief_card("Portfolio Health", f"{health_score}/100", health_color, health_label)
        + _brief_card("Today's Market", sentiment, sent_color)
        + _brief_card("Cash Position", f"{cash_pct:.0f}%",
                      GAIN if 2 <= cash_pct <= 20 else NEURAL,
                      "Buying power available")
        + f'</div>'
    )

    # ── Second row: Opportunity / Risk / Action ───────────────────────────────
    action_label = action_items[0] if action_items else "No action needed"
    action_color = LOSS if action_items else GAIN
    mid_row = (
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;">'
        + _brief_card("Highest Opportunity", opp_sym, GAIN, "Best signal strength")
        + _brief_card("Highest Risk", risk_sym, LOSS, "Worst unrealized P&L")
        + _brief_card("Action Required", action_label, action_color,
                      f"{len(action_items)} item{'s' if len(action_items) != 1 else ''}" if action_items else "All positions stable")
        + f'</div>'
    )

    # ── Action items list ─────────────────────────────────────────────────────
    if action_items:
        items_html = "".join(
            f'<div style="display:flex;align-items:center;gap:8px;padding:6px 0;'
            f'border-bottom:1px solid {BORDER};">'
            f'<span style="color:{LOSS};font-weight:700;">!</span>'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT1};">{item}</span>'
            f'</div>'
            for item in action_items
        )
    else:
        items_html = (
            f'<div style="color:{GAIN};font-size:{FONT_LABEL};padding:8px 0;">'
            f'&#10003; No action required for {no_action} position{"s" if no_action != 1 else ""}.</div>'
        )

    # ── AI Summary ───────────────────────────────────────────────────────────
    if action_items:
        ai_summary = (
            f'{len(action_items)} position{"s need" if len(action_items) != 1 else " needs"} '
            f'attention today. {no_action} position{"s are" if no_action != 1 else " is"} stable.'
        )
    else:
        ai_summary = (
            f'No action required for {no_action} position{"s" if no_action != 1 else ""}. '
            f'Portfolio is {health_label.lower()} with {sentiment.lower()} market conditions.'
        )

    review_mins = max(2, len(action_items) + no_action)

    summary_block = (
        f'<div style="background:{SURFACE2};border-radius:6px;padding:12px 14px;margin-top:8px;">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};margin-bottom:4px;font-weight:600;">'
        f'AI Summary</div>'
        f'<div style="font-size:{FONT_VALUE};color:{TEXT1};line-height:1.5;">{ai_summary}</div>'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT3};margin-top:6px;">'
        f'Estimated review time: {review_mins} minute{"s" if review_mins != 1 else ""}.</div>'
        f'</div>'
    )

    header = (
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
        f'flex-wrap:wrap;gap:8px;margin-bottom:14px;">'
        f'<span style="font-size:{FONT_HERO};font-weight:800;color:{TEXT1};">{greeting}</span>'
        f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">{date_str} &nbsp; {closed_badge}</span>'
        f'</div>'
    )

    body = header + top_row + mid_row + items_html + summary_block

    timestamp = (
        f'<div class="nt-status">'
        f'<span style="color:{TEXT2};font-size:{FONT_LABEL};">Updated <strong style="color:{TEXT1};">'
        f'{now_ct}</strong></span>'
        f'<span style="display:inline-flex;align-items:center;gap:5px;">'
        f'<span style="width:6px;height:6px;background:{mkt_color};border-radius:50%;'
        f'display:inline-block;"></span>'
        f'<span style="color:{mkt_color};font-weight:600;font-size:{FONT_LABEL};">'
        f'{mkt_label}</span></span>'
        f'</div>'
    )
    inner = (
        f'<div class="nt-card" style="padding:20px 18px;">{body}</div>'
        + timestamp
    )
    return f'<div class="nt nt-wrap">{inner}</div>'
