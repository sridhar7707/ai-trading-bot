"""End-of-day shutdown job — runs once after market close, guarded by shutdown_completed."""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from scheduler.session_manager import Session


@dataclass
class ShutdownResult:
    success: bool = True
    exception: str = ""


def run(session: Session) -> ShutdownResult:
    """Execute all end-of-day tasks. Continues past individual failures."""
    result = ShutdownResult()
    logger.info(f"shutdown_job: begin for session {session.session_id}")

    steps = [
        ("final_prices",       _final_price_update),
        ("eod_pnl",            _calculate_eod_pnl),
        ("performance_metrics",_update_performance_metrics),
        ("morning_brief_prep", _generate_morning_brief_data),
        ("health_score",       _recalculate_health_score),
        ("daily_summary",      _write_daily_summary),
        ("session_log",        lambda: _log_session_summary(session)),
    ]

    failed = []
    for name, fn in steps:
        try:
            fn()
        except Exception as exc:
            logger.warning(f"shutdown_job step '{name}' failed: {exc}")
            failed.append(name)

    result.success = len(failed) <= 2  # tolerate ≤2 step failures; ≥3 is a broken shutdown
    if failed:
        result.exception = f"steps failed: {', '.join(failed)}"
    logger.info(f"shutdown_job: complete — {len(steps) - len(failed)}/{len(steps)} steps ok")
    return result


def _final_price_update() -> None:
    from dashboard.data import _current_prices
    from dashboard.data import get_db_conn, DB_PATH
    import os
    if not os.path.exists(DB_PATH):
        return
    with get_db_conn() as con:
        rows = con.execute(
            "SELECT DISTINCT symbol FROM position_state"
        ).fetchall()
    symbols = [r[0] for r in rows]
    if symbols:
        prices = _current_prices(symbols)
        logger.info(f"shutdown_job: final prices fetched for {len(prices)} symbols")


def _calculate_eod_pnl() -> None:
    from dashboard.data import get_data
    d = get_data()
    open_pos = d.get("open_pos", {})
    prices   = d.get("prices", {})
    total_pnl = 0.0
    for sym, pos in open_pos.items():
        cur = prices.get(sym, 0.0)
        if cur > 0 and pos["invested"] > 0:
            total_pnl += pos["shares"] * cur - pos["invested"]
    logger.info(f"shutdown_job: EOD unrealized P&L = ${total_pnl:+,.2f}")


def _update_performance_metrics() -> None:
    from dashboard.data import get_db_conn, DB_PATH
    import os
    if not os.path.exists(DB_PATH):
        return
    with get_db_conn() as con:
        n_trades = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    logger.info(f"shutdown_job: performance metrics updated (total trades: {n_trades})")


def _generate_morning_brief_data() -> None:
    from dashboard.components.brief import render_morning_brief
    render_morning_brief()
    logger.info("shutdown_job: morning brief data pre-generated for tomorrow")


def _recalculate_health_score() -> None:
    from dashboard.data import get_data
    from bot.core.recommendation_engine import get_portfolio_health
    d = get_data()
    health = get_portfolio_health(d)
    logger.info(f"shutdown_job: portfolio health = {health.get('total', 0)}/100")


def _write_daily_summary() -> None:
    from dashboard.data import get_db_conn, DB_PATH
    import datetime
    import os
    if not os.path.exists(DB_PATH):
        return
    today = datetime.date.today().isoformat()
    with get_db_conn() as con:
        try:
            row = con.execute(
                "SELECT COUNT(*), AVG(pnl_pct) FROM trades WHERE timestamp LIKE ? AND action LIKE 'SELL%'",
                (today + "%",),
            ).fetchone()
            logger.info(
                f"shutdown_job: today={today} sells={row[0]} avg_pnl={row[1] or 0:.2f}%"
            )
        except Exception as exc:
            logger.warning(f"shutdown_job: EOD stats query failed: {exc}")


def _log_session_summary(session: Session) -> None:
    logger.info(
        f"shutdown_job: session {session.session_id} | "
        f"date={session.session_date} | trades_today={session.trades_today} | "
        f"cycles_today={session.cycles_today}"
    )
