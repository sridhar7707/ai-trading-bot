"""Dispatcher — the sole cron entry point. Routes to exactly one job per execution."""
from __future__ import annotations

import datetime
import sys
import os

from loguru import logger

# Allow running as `python scheduler/dispatcher.py` from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler import market_calendar, session_manager, startup_job, trading_job, shutdown_job
from scheduler.health_monitor import ExecutionLog, save as _save_log

PORTFOLIO_ID = 1  # single-portfolio Phase 1


def main() -> None:
    started_at = datetime.datetime.now(datetime.timezone.utc)
    log = ExecutionLog(started_at=started_at, portfolio_id=PORTFOLIO_ID)

    try:
        et_now = market_calendar.now_et()
        today  = et_now.date()

        # 1. Skip non-trading days immediately
        if not market_calendar.is_trading_day(today):
            log.job_executed = "skip"
            log.market_state_at_execution = "CLOSED"
            log.success = True
            logger.info(f"dispatcher: {today} is not a trading day — skip")
            return

        # 2. Load or create today's session
        session = session_manager.get_today_session(PORTFOLIO_ID)
        if session is None:
            session = session_manager.create_session(PORTFOLIO_ID, today)
        log.session_id = session.session_id

        # 3. Get current market state and persist it
        state = market_calendar.get_market_state(et_now)
        session_manager.update_state(session, state)
        log.market_state_at_execution = state

        logger.info(
            f"dispatcher: {et_now.strftime('%H:%M ET')} | state={state} | "
            f"session={session.session_id} | initialized={session.initialized} | "
            f"startup_completed={session.startup_completed} | "
            f"shutdown_completed={session.shutdown_completed}"
        )

        # 4. Route to exactly one job
        if state == "PREMARKET" and session_manager.is_startup_needed(session):
            # Guard BEFORE running — crash-safe idempotency
            session_manager.mark_initialized(session)
            result = startup_job.run(session)
            session_manager.mark_startup_complete(session)
            log.job_executed = "startup"
            log.success = result.success
            if not result.success:
                log.exception = result.notes

        elif state == "OPEN" and session.startup_completed:
            result = trading_job.run(session)
            if result.trades_executed:
                session_manager.increment_trades(session, result.trades_executed)
            log.job_executed = "trading"
            log.trades_executed = result.trades_executed
            log.success = result.success
            if not result.success:
                log.exception = result.exception

        elif state == "CLOSED" and session_manager.is_shutdown_needed(session):
            # Guard BEFORE running — crash-safe idempotency
            session_manager.mark_shutdown_complete(session)
            result = shutdown_job.run(session)
            log.job_executed = "shutdown"
            log.success = result.success
            if not result.success:
                log.exception = result.exception

        else:
            log.job_executed = "skip"
            log.success = True
            logger.info(f"dispatcher: no job to run (state={state}, session consistent)")

    except Exception as exc:
        log.success = False
        log.exception = str(exc)
        logger.error(f"dispatcher: unhandled exception — {exc}")

    finally:
        finished_at = datetime.datetime.now(datetime.timezone.utc)
        log.finished_at = finished_at
        elapsed_ms = int((finished_at - started_at).total_seconds() * 1000)
        log.execution_time_ms = elapsed_ms
        _save_log(log)
        logger.info(
            f"dispatcher: done in {elapsed_ms}ms | job={log.job_executed} | success={log.success}"
        )


if __name__ == "__main__":
    main()
