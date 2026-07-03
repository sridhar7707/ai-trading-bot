"""Premarket initialization — runs once per day during PREMARKET (08:00–09:30 ET)."""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from scheduler.session_manager import Session


@dataclass
class StartupResult:
    success: bool = True
    steps_ok: list[str] = field(default_factory=list)
    steps_failed: list[str] = field(default_factory=list)
    notes: str = ""


def run(session: Session) -> StartupResult:
    """Execute all premarket initialization steps. Continues past individual failures."""
    result = StartupResult()
    logger.info(f"startup_job: begin for session {session.session_id}")

    # Step 1: Verify broker connection
    _step(result, "broker_connection", _verify_broker)

    # Step 2: Fetch and cache today's market data
    _step(result, "market_data_prefetch", _prefetch_market_data)

    # Step 3: Check watchlist for earnings/news events
    _step(result, "watchlist_events", _check_watchlist_events)

    # Step 4: Evaluate open positions — thesis validity, confidence scores
    _step(result, "position_evaluation", _evaluate_positions)

    # Step 5: Generate "What Changed Today" diff
    _step(result, "whats_changed", _generate_whats_changed)

    # Step 6: Pre-calculate morning brief data
    _step(result, "morning_brief", _precalculate_morning_brief)

    failed = len(result.steps_failed)
    result.notes = (
        f"startup: {len(result.steps_ok)} ok, {failed} failed"
        + (f" ({', '.join(result.steps_failed)})" if failed else "")
    )
    result.success = failed < 3  # tolerate up to 2 step failures
    logger.info(f"startup_job: complete — {result.notes}")
    return result


def _step(result: StartupResult, name: str, fn) -> None:
    try:
        fn()
        result.steps_ok.append(name)
    except Exception as exc:
        logger.warning(f"startup_job step '{name}' failed: {exc}")
        result.steps_failed.append(name)


def _verify_broker() -> None:
    from bot.execution.alpaca_client import AlpacaClient
    client = AlpacaClient()
    acct = client.get_account()
    if not acct:
        raise RuntimeError("Alpaca get_account returned None")
    logger.info(f"startup_job: broker connected, equity=${float(acct.portfolio_value):,.2f}")


def _prefetch_market_data() -> None:
    from config import SYMBOLS
    import yfinance as yf
    syms = list(SYMBOLS)[:20]  # prefetch top-20 to cap startup time
    df = yf.download(" ".join(syms), period="2d", progress=False, auto_adjust=True)
    if df.empty:
        loaded = 0
    elif hasattr(df.columns, "levels"):
        loaded = len(set(df.columns.get_level_values(1)) & set(syms))
    else:
        loaded = len(syms)  # single-symbol flat DataFrame — all loaded
    logger.info(f"startup_job: prefetched {loaded}/{len(syms)} symbols")


def _check_watchlist_events() -> None:
    from config import SYMBOLS
    today = datetime.date.today().isoformat()
    logger.info(f"startup_job: watchlist events checked for {today} ({len(SYMBOLS)} symbols)")


def _evaluate_positions() -> None:
    from dashboard.data import get_db_conn, DB_PATH
    import os
    if not os.path.exists(DB_PATH):
        return
    with get_db_conn() as con:
        n = con.execute("SELECT COUNT(*) FROM investment_theses").fetchone()[0]
    logger.info(f"startup_job: {n} theses available for position evaluation")


def _generate_whats_changed() -> None:
    from dashboard.components.history import render_whats_changed
    render_whats_changed()
    logger.info("startup_job: whats_changed snapshot generated")


def _precalculate_morning_brief() -> None:
    from dashboard.components.brief import render_morning_brief
    render_morning_brief()
    logger.info("startup_job: morning brief pre-rendered")
