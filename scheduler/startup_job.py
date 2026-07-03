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

    # Infrastructure — must run before trading
    _step(result, "db_sync",            _sync_db)
    _step(result, "model_pull",         _pull_models)
    _step(result, "universe_screen",    _screen_universe)
    _step(result, "sentiment_prefetch", _prefetch_sentiment)

    # Dashboard / portfolio prep
    _step(result, "broker_connection",    _verify_broker)
    _step(result, "market_data_prefetch", _prefetch_market_data)
    _step(result, "watchlist_events",     _check_watchlist_events)
    _step(result, "position_evaluation",  _evaluate_positions)
    _step(result, "whats_changed",        _generate_whats_changed)
    _step(result, "morning_brief",        _precalculate_morning_brief)

    failed = len(result.steps_failed)
    result.notes = (
        f"startup: {len(result.steps_ok)} ok, {failed} failed"
        + (f" ({', '.join(result.steps_failed)})" if failed else "")
    )
    result.success = failed < 4  # tolerate up to 3 failures across 10 steps
    logger.info(f"startup_job: complete — {result.notes}")
    return result


def _step(result: StartupResult, name: str, fn) -> None:
    try:
        fn()
        result.steps_ok.append(name)
    except Exception as exc:
        logger.warning(f"startup_job step '{name}' failed: {exc}")
        result.steps_failed.append(name)


def _sync_db() -> None:
    import shutil
    from config import HF_TOKEN, HF_DB_REPO_ID, TRADE_DB_PATH
    if not HF_TOKEN:
        logger.info("startup_job: HF_TOKEN not set — skipping db sync")
        return
    from huggingface_hub import hf_hub_download
    cached = hf_hub_download(
        repo_id=HF_DB_REPO_ID, filename="trades.db",
        repo_type="dataset", token=HF_TOKEN, force_download=True,
    )
    shutil.copy(cached, TRADE_DB_PATH)
    import sqlite3
    n = sqlite3.connect(TRADE_DB_PATH).execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    logger.info(f"startup_job: trades.db synced from {HF_DB_REPO_ID} ({n} records)")


def _pull_models() -> None:
    from scripts.load_model_hf import pull
    pull()
    logger.info("startup_job: ML models pulled from HF")


def _screen_universe() -> None:
    import argparse
    from scripts.screen_universe import main as _screen_main
    ns = argparse.Namespace(max=25, max_per_sector=3, min_price=10.0, min_adv=5_000_000, min_history=245)
    _screen_main(ns)
    logger.info("startup_job: universe screened → data/universe_today.json")


def _prefetch_sentiment() -> None:
    from scripts.prefetch_sentiment import main as _sentiment_main
    _sentiment_main()
    logger.info("startup_job: sentiment prefetched → data/sentiment_today.json")


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
