"""Investment Thesis Engine — record, track, and validate position theses (req 11.3)."""
from __future__ import annotations
import datetime
import sqlite3
from loguru import logger
from dashboard.design_system import (
    SURFACE, SURFACE2, BORDER, TEXT1, TEXT2, TEXT3,
    GAIN, LOSS, NEURAL, PRIMARY,
    FONT_SECTION, FONT_VALUE, FONT_LABEL, WEIGHT_BOLD,
    _section, _wrap, _card, _empty_state,
    TH, TD, TD0,
)
from dashboard.data import get_data, get_db_conn, DB_PATH, safe_query
from bot.core.error_logger import safe_render, timed, log_exception
import os

_logger = logger

_VALIDITY_COLORS = {
    "valid":       GAIN,
    "weakening":   NEURAL,
    "invalidated": LOSS,
}


def _ai_generate_thesis(symbol: str, d: dict) -> str:
    """Generate a plain-English thesis for a symbol from available signals."""
    open_pos = d.get("open_pos", {})
    prices   = d.get("prices", {})
    pos = open_pos.get(symbol, {})
    cur_price = prices.get(symbol, 0.0)
    entry = pos.get("entry_price", cur_price)
    pnl_pct = ((cur_price - entry) / entry * 100) if entry > 0 and cur_price > 0 else 0.0

    # Fetch latest signals
    ensemble = 0.0
    regime   = "Unknown"
    try:
        if os.path.exists(DB_PATH):
            with get_db_conn() as con:
                row = con.execute(
                    "SELECT ensemble_score, regime FROM signal_log "
                    "WHERE symbol=? ORDER BY id DESC LIMIT 1", (symbol,)
                ).fetchone()
            if row:
                ensemble = float(row[0] or 0.0)
                regime   = row[1] or "Unknown"
    except Exception as exc:
        _logger.debug(f"get_current_thesis_text: signal_log read: {exc}")

    conf_pct = int(ensemble * 100)
    trend = "positive price momentum" if pnl_pct > 0 else "recovering position"
    return (
        f"Holding {symbol} with {conf_pct}% AI confidence. "
        f"Market regime: {regime}. "
        f"Position shows {trend} ({pnl_pct:+.1f}% from entry). "
        f"Thesis valid while AI ensemble remains above 60%."
    )


def save_thesis(symbol: str, thesis_text: str, price_target: float,
                invalidation: str, review_trigger: str = "quarterly",
                confidence: int = 75) -> bool:
    """Upsert investment thesis for a symbol."""
    if not symbol or not symbol.strip():
        return False
    if review_trigger not in ("weekly", "monthly", "quarterly"):
        review_trigger = "quarterly"
    price_target = max(0.0, float(price_target or 0.0))
    confidence = max(0, min(100, int(confidence or 75)))
    if not os.path.exists(DB_PATH):
        return False
    today = datetime.date.today().isoformat()
    review_date = _next_review_date(review_trigger)
    try:
        with get_db_conn() as con:
            existing = con.execute(
                "SELECT thesis_id FROM investment_theses WHERE symbol=? "
                "ORDER BY created_at DESC LIMIT 1", (symbol,)
            ).fetchone()
            if existing:
                con.execute(
                    "UPDATE investment_theses SET thesis_text=?, price_target=?,"
                    "invalidation_criteria=?, review_trigger=?, next_review_date=?,"
                    "confidence_at_entry=?, current_validity='valid',"
                    "last_evaluated_date=? WHERE thesis_id=?",
                    (thesis_text, price_target, invalidation, review_trigger,
                     review_date, confidence, today, existing[0])
                )
            else:
                con.execute(
                    "INSERT INTO investment_theses "
                    "(symbol, thesis_text, price_target, invalidation_criteria,"
                    "review_trigger, next_review_date, confidence_at_entry,"
                    "current_validity, last_evaluated_date) "
                    "VALUES (?,?,?,?,?,?,?,'valid',?)",
                    (symbol, thesis_text, price_target, invalidation,
                     review_trigger, review_date, confidence, today)
                )
            con.commit()
        return True
    except Exception as exc:
        log_exception(_logger, "save_thesis", exc, {"symbol": symbol})
        return False


def _next_review_date(trigger: str) -> str:
    today = datetime.date.today()
    if trigger == "monthly":
        return (today + datetime.timedelta(days=30)).isoformat()
    if trigger == "quarterly":
        return (today + datetime.timedelta(days=90)).isoformat()
    if trigger == "weekly":
        return (today + datetime.timedelta(days=7)).isoformat()
    return (today + datetime.timedelta(days=90)).isoformat()


def _load_theses() -> list[dict]:
    if not os.path.exists(DB_PATH):
        return []
    try:
        with get_db_conn() as con:
            rows = con.execute(
                "SELECT thesis_id, symbol, thesis_text, price_target,"
                "invalidation_criteria, review_trigger, next_review_date,"
                "confidence_at_entry, current_validity, last_evaluated_date,"
                "ai_evaluation_notes FROM investment_theses ORDER BY created_at DESC"
            ).fetchall()
        cols = ["thesis_id", "symbol", "thesis_text", "price_target",
                "invalidation_criteria", "review_trigger", "next_review_date",
                "confidence_at_entry", "current_validity", "last_evaluated_date",
                "ai_evaluation_notes"]
        return [dict(zip(cols, r)) for r in rows]
    except Exception as exc:
        log_exception(_logger, "_load_theses", exc)
        return []


def _evaluate_thesis_validity(thesis: dict, d: dict) -> str:
    """Return 'valid'/'weakening'/'invalidated' based on current signals."""
    symbol = thesis["symbol"]
    prices = d.get("prices", {})
    cur = prices.get(symbol, 0.0)
    target = thesis.get("price_target") or 0.0

    # Check if price target already exceeded (invalidated by success)
    if target > 0 and cur > 0 and cur >= target * 0.98:
        return "valid"

    # Fetch current ensemble score
    ensemble = 0.0
    try:
        if os.path.exists(DB_PATH):
            with get_db_conn() as con:
                row = con.execute(
                    "SELECT ensemble_score FROM signal_log WHERE symbol=? ORDER BY id DESC LIMIT 1",
                    (symbol,)
                ).fetchone()
            if row:
                ensemble = float(row[0] or 0.0)
    except Exception as exc:
        _logger.debug(f"_check_thesis_validity: signal_log read: {exc}")

    if ensemble >= 0.65:
        return "valid"
    if ensemble >= 0.50:
        return "weakening"
    return "invalidated"


def _auto_populate_missing(d: dict) -> None:
    """Auto-generate theses for open positions that don't have one."""
    open_pos = d.get("open_pos", {})
    if not open_pos or not os.path.exists(DB_PATH):
        return
    try:
        with get_db_conn() as con:
            existing = {r[0] for r in con.execute(
                "SELECT DISTINCT symbol FROM investment_theses"
            ).fetchall()}
    except Exception:
        return
    prices = d.get("prices", {})
    for sym in open_pos:
        if sym not in existing:
            cur = prices.get(sym, 0.0)
            target = round(cur * 1.15, 2) if cur > 0 else 0.0
            save_thesis(
                symbol=sym,
                thesis_text=_ai_generate_thesis(sym, d),
                price_target=target,
                invalidation="AI ensemble score drops below 50% for 3+ consecutive days",
                review_trigger="quarterly",
                confidence=65,
            )


@timed(_logger)
@safe_render("Investment Thesis")
def render_thesis_tracker() -> str:
    d = get_data()
    _auto_populate_missing(d)
    theses = _load_theses()
    open_pos = d.get("open_pos", {})

    if not theses:
        return (
            f'<div class="nt nt-wrap">'
            f'{_section("📋", "Investment Thesis", "Position rationale tracker")}'
            f'{_card(_empty_state("📋", "No theses recorded yet", "Theses are auto-generated when the bot opens positions."))}'
            f'</div>'
        )

    today = datetime.date.today().isoformat()
    overdue_reviews = [t for t in theses if t.get("next_review_date", "9999") <= today]
    note = (f"{len(overdue_reviews)} review{'s' if len(overdue_reviews) != 1 else ''} overdue"
            if overdue_reviews else f"{len(theses)} active thesis{'es' if len(theses) != 1 else ''}")

    # Update validity for open positions
    rows_html = ""
    n = len(theses)
    for i, t in enumerate(theses):
        sym = t["symbol"]
        validity = t["current_validity"] or "valid"
        if sym in open_pos:
            validity = _evaluate_thesis_validity(t, d)
        vc = _VALIDITY_COLORS.get(validity, TEXT2)
        conf = t.get("confidence_at_entry") or 75
        review = t.get("next_review_date") or "—"
        overdue = review <= today if review != "—" else False
        review_c = LOSS if overdue else TEXT2
        price_tgt = t.get("price_target") or 0.0
        prices = d.get("prices", {})
        cur_p = prices.get(sym, 0.0)
        upside = (f"+{(price_tgt/cur_p - 1)*100:.0f}%"
                  if price_tgt > 0 and cur_p > 0 else "—")
        thesis_short = (t.get("thesis_text") or "Auto-generated")[:80]
        if len(t.get("thesis_text") or "") > 80:
            thesis_short += "…"
        td = TD if i < n - 1 else TD0
        rows_html += (
            f'<tr>'
            f'<td {td}><span style="font-family:Courier New,monospace;font-weight:700;color:{TEXT1};">'
            f'{sym}</span></td>'
            f'<td {td}><span style="color:{vc};font-weight:{WEIGHT_BOLD};text-transform:capitalize;">'
            f'{validity}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{conf}%</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{GAIN};">{upside}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{review_c};">'
            f'{"⚠ " if overdue else ""}{review}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{thesis_short}</span></td>'
            f'</tr>'
        )

    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Symbol</th><th {TH}>Validity</th><th {TH}>Confidence</th>'
        f'<th {TH}>Upside</th><th {TH}>Next Review</th><th {TH}>Thesis</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table>'
    )
    return f'<div class="nt nt-wrap">{_section("📋", "Investment Thesis", note)}{table}</div>'


from dashboard.registry import ComponentSpec, RefreshGroup, register
register(ComponentSpec("thesis_out", RefreshGroup.SLOW, render_thesis_tracker, priority=45))
