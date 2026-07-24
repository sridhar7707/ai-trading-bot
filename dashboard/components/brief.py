"""Morning Executive Brief — daily portfolio summary panel (req 11.1)."""
from __future__ import annotations
import datetime
import time
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

_CRON_INTERVAL_MINS = 5  # expected cron cadence

_logger = logger

# ── SPY daily % change with 5-min cache ───────────────────────────────────────
_spy_cache: dict = {"pct": 0.0, "ts": 0.0}


def _spy_pct_today() -> float:
    now = time.time()
    if now - _spy_cache["ts"] < 300:
        return _spy_cache["pct"]
    # Use the main data cache — _refresh_cache() already fetches SPY in the same
    # yfinance batch as open positions, so this avoids a duplicate network call.
    cached_pct = get_data().get("spy_pct", 0.0)
    if cached_pct != 0.0:
        _spy_cache["pct"] = float(cached_pct)
        _spy_cache["ts"] = now
        return _spy_cache["pct"]
    # Fallback: direct yfinance (first load before cache populates, or truly flat day)
    try:
        import yfinance as yf
        df = yf.download("SPY", period="5d", progress=False, auto_adjust=True)
        if len(df) >= 2:
            _spy_cache["pct"] = float(
                (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
            )
        else:
            _spy_cache["pct"] = 0.0
    except Exception:
        pass
    _spy_cache["ts"] = now
    return _spy_cache["pct"]


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
        return "&mdash;", "&mdash;"

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
        best_sym = max(pnls, key=lambda s: pnls[s]) if pnls else "&mdash;"

    # Highest risk: worst pnl or sell analysis
    worst_sym = min(pnls, key=lambda s: pnls[s]) if pnls else "&mdash;"
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
@safe_render("Three-Question Summary")
def render_three_question_summary() -> str:
    """Three cards: Portfolio Health · Today's Action · Benchmark (req 2.1)."""
    health_score, health_label = _portfolio_health_score()
    action_items = _action_items()
    spy_pct = _spy_pct_today()

    health_color = GAIN if health_score >= 80 else (NEURAL if health_score >= 60 else LOSS)

    if not action_items:
        action_val, action_sub, action_color = "No Action Needed", "Portfolio on track", GAIN
    elif len(action_items) == 1:
        action_val, action_sub, action_color = action_items[0], "1 action", LOSS
    else:
        action_val = f"{len(action_items)} actions"
        action_sub, action_color = "Tap to review", NEURAL

    sign = "+" if spy_pct >= 0 else ""
    spy_color = GAIN if spy_pct >= 0 else LOSS

    def _q_card(label: str, val: str, color: str, sub: str = "") -> str:
        sub_html = (
            f'<div style="font-size:11px;color:{TEXT3};margin-top:4px;">{sub}</div>'
            if sub else ""
        )
        return (
            f'<div style="flex:1;text-align:center;padding:18px 12px;'
            f'background:{SURFACE};border-radius:10px;border:1px solid {BORDER};'
            f'min-width:140px;">'
            f'<div style="font-size:10px;color:{TEXT3};text-transform:uppercase;'
            f'letter-spacing:.8px;margin-bottom:8px;">{label}</div>'
            f'<div style="font-size:{FONT_SECTION};font-weight:800;color:{color};">{val}</div>'
            f'{sub_html}'
            f'</div>'
        )

    row = (
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;padding:4px 0;">'
        + _q_card("Portfolio Health", f"{health_score}/100", health_color, health_label)
        + _q_card("Today's Action", action_val, action_color, action_sub)
        + _q_card("Benchmark", f"SPY {sign}{spy_pct:.1f}%", spy_color, "Today vs SPY")
        + f'</div>'
    )
    return f'<div class="nt nt-wrap" style="padding:0;">{row}</div>'


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
    from zoneinfo import ZoneInfo
    _now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    hour     = _now_et.hour
    greeting = "Good Morning" if hour < 12 else ("Good Afternoon" if hour < 17 else "Good Evening")
    date_str = _now_et.strftime("%B %d, %Y")

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


@timed(_logger)
@safe_render("Scheduler Status")
def render_scheduler_status() -> str:
    """Operational status panel — always visible, above the morning brief."""
    from scheduler.health_monitor import get_health_summary
    from scheduler.session_manager import get_today_session
    from scheduler.market_calendar import is_trading_day, now_et

    mkt_label, mkt_color = _market_status()
    today_et = now_et().date()
    is_trading = is_trading_day(today_et)

    # Load health summary from execution_log
    try:
        health = get_health_summary()
    except Exception:
        health = None

    # Load today's session
    try:
        session = get_today_session()
    except Exception:
        session = None

    # ── Status dot + label ────────────────────────────────────────────────────
    if not is_trading:
        dot_color   = TEXT2
        status_text = "Market Closed &mdash; No Trading Today"
        session_state = "HOLIDAY" if today_et.weekday() < 5 else "WEEKEND"
    elif health is None or health.cron_status == "Unknown":
        dot_color   = TEXT2
        status_text = "Pending First Run"
        session_state = session.state if session else "UNKNOWN"
    elif health.cron_status == "Down":
        dot_color   = LOSS
        status_text = "Down"
        session_state = session.state if session else "UNKNOWN"
    elif health.cron_status == "Degraded":
        dot_color   = NEURAL
        status_text = "Degraded"
        session_state = session.state if session else "UNKNOWN"
    else:
        dot_color   = GAIN
        status_text = "Running"
        session_state = session.state if session else "UNKNOWN"

    def _et_label(iso: str | None) -> str:
        """Convert UTC ISO string to ET display."""
        if not iso:
            return "&mdash;"
        try:
            from zoneinfo import ZoneInfo
            dt = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            et = dt.astimezone(ZoneInfo("America/New_York"))
            return et.strftime("%I:%M %p ET")
        except Exception:
            return str(iso)[:16]

    last_cycle  = _et_label(health.last_execution_time.isoformat() if health and health.last_execution_time else None)
    next_cycle  = "&mdash;"
    if health and health.last_execution_time:
        try:
            nxt = health.last_execution_time + datetime.timedelta(minutes=_CRON_INTERVAL_MINS)
            next_cycle = _et_label(nxt.isoformat())
        except Exception:
            pass

    trades_today = session.trades_today if session else 0
    avg_ms       = int(health.avg_execution_time_ms) if health else 0
    cron_lbl     = health.cron_status if health else "Unknown"
    cron_color   = GAIN if cron_lbl == "Healthy" else (NEURAL if cron_lbl == "Degraded" else LOSS)

    def _chip(label: str, val: str, val_color: str = TEXT1) -> str:
        return (
            f'<span style="display:inline-flex;flex-direction:column;align-items:center;'
            f'gap:1px;padding:6px 14px;background:{SURFACE2};border-radius:6px;'
            f'border:1px solid {BORDER};min-width:90px;">'
            f'<span style="font-size:10px;color:{TEXT3};text-transform:uppercase;'
            f'letter-spacing:.6px;">{label}</span>'
            f'<span style="font-size:{FONT_VALUE};font-weight:700;color:{val_color};">{val}</span>'
            f'</span>'
        )

    # Server-side CT time chip (CDT in summer, CST in winter)
    from zoneinfo import ZoneInfo as _ZI
    _now_ct = datetime.datetime.now(_ZI("America/Chicago"))
    _now_ct_str = _now_ct.strftime("%I:%M %p %Z")
    local_time_chip = _chip("Now", _now_ct_str)

    chips = (
        f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;">'
        + local_time_chip
        + _chip("Session", session_state, PRIMARY)
        + _chip("Cron", cron_lbl, cron_color)
        + _chip("Last Cycle", last_cycle)
        + _chip("Next ~", next_cycle)
        + _chip("Trades Today", str(trades_today), GAIN if trades_today else TEXT2)
        + _chip("Avg Run", f"{avg_ms}ms" if avg_ms else "&mdash;")
        + f'</div>'
    )

    header_row = (
        f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
        f'<span style="width:10px;height:10px;border-radius:50%;background:{dot_color};'
        f'display:inline-block;flex-shrink:0;"></span>'
        f'<span style="font-weight:700;font-size:{FONT_VALUE};color:{TEXT1};">{status_text}</span>'
        f'<span style="margin-left:auto;font-size:{FONT_LABEL};color:{mkt_color};">'
        f'{mkt_label}</span>'
        f'</div>'
    )

    body = (
        f'<div class="nt-card" style="padding:14px 18px;">'
        f'{header_row}{chips}'
        f'</div>'
    )
    return f'<div class="nt nt-wrap">{body}</div>'


from dashboard.registry import ComponentSpec, RefreshGroup, register
register(ComponentSpec("morning_brief_out",    RefreshGroup.FAST, render_morning_brief,           priority=12))
register(ComponentSpec("scheduler_status_out", RefreshGroup.FAST, render_scheduler_status,        priority=13))
register(ComponentSpec("three_q_out",          RefreshGroup.SLOW, render_three_question_summary,  priority=10))
