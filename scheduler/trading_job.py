"""Market-hours trading cycle — stateless; called by dispatcher every cron tick."""
from __future__ import annotations

import datetime
from dataclasses import dataclass

from loguru import logger

from scheduler import market_calendar
from scheduler.session_manager import Session

_CUTOFF_MINS_BEFORE_CLOSE = 5  # no new trades within 5 min of close
_TIMEOUT_SECS = 55


@dataclass
class TradingResult:
    success: bool = True
    trades_executed: int = 0
    exception: str = ""


def run(session: Session) -> TradingResult:
    """Run one trading cycle. Must complete within _TIMEOUT_SECS seconds."""
    import signal as _signal

    result = TradingResult()
    now_et = market_calendar.now_et()

    # Hard cutoff: no new trades in the last 5 minutes before close
    mins_left = market_calendar.minutes_until_close(now_et)
    if mins_left <= _CUTOFF_MINS_BEFORE_CLOSE:
        logger.info(
            f"trading_job: skipped — {mins_left} min to close (cutoff={_CUTOFF_MINS_BEFORE_CLOSE})"
        )
        return result

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"trading_job exceeded {_TIMEOUT_SECS}s")

    # SIGALRM is Unix-only; on Windows we skip the hard timeout
    try:
        _signal.signal(_signal.SIGALRM, _timeout_handler)
        _signal.alarm(_TIMEOUT_SECS)
    except (AttributeError, OSError):
        pass  # Windows — no SIGALRM

    try:
        trades_before = _count_trades_today()
        _run_cycle()
        trades_after = _count_trades_today()
        result.trades_executed = max(0, trades_after - trades_before)
        result.success = True
        logger.info(f"trading_job: cycle complete, {result.trades_executed} new trade(s)")
    except TimeoutError as exc:
        result.success = False
        result.exception = str(exc)
        logger.warning(f"trading_job: {exc}")
    except Exception as exc:
        result.success = False
        result.exception = str(exc)
        logger.error(f"trading_job: unhandled exception — {exc}")
    finally:
        try:
            _signal.alarm(0)
        except (AttributeError, OSError):
            pass

    return result


def _run_cycle() -> None:
    """Delegate to the main trading engine."""
    from bot.main import run as bot_run
    bot_run(mode="paper")


def _count_trades_today() -> int:
    """Count trades placed so far today (used to diff before/after cycle)."""
    from dashboard.data import get_db_conn, DB_PATH
    import os
    if not os.path.exists(DB_PATH):
        return 0
    today = datetime.date.today().isoformat()
    try:
        with get_db_conn() as con:
            return con.execute(
                "SELECT COUNT(*) FROM trades WHERE timestamp LIKE ?",
                (today + "%",),
            ).fetchone()[0]
    except Exception:
        return 0
