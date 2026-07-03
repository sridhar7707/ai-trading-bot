"""Record every cron execution to execution_log; expose health summary."""
from __future__ import annotations

import datetime
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from config import TRADE_DB_PATH


@dataclass
class ExecutionLog:
    started_at: datetime.datetime
    session_id: Optional[int] = None
    portfolio_id: int = 1
    finished_at: Optional[datetime.datetime] = None
    execution_time_ms: int = 0
    job_executed: str = "skip"
    success: bool = False
    exception: Optional[str] = None
    trades_executed: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    market_state_at_execution: str = "CLOSED"


@dataclass
class HealthSummary:
    last_execution_time: Optional[datetime.datetime] = None
    last_success_time: Optional[datetime.datetime] = None
    consecutive_failures: int = 0
    avg_execution_time_ms: float = 0.0
    cron_status: str = "Down"


def _conn(db_path: str = TRADE_DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def _ensure_table(db_path: str = TRADE_DB_PATH) -> None:
    con = _conn(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS execution_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            portfolio_id INTEGER DEFAULT 1,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            execution_time_ms INTEGER DEFAULT 0,
            job_executed TEXT DEFAULT 'skip',
            success INTEGER DEFAULT 0,
            exception TEXT,
            trades_executed INTEGER DEFAULT 0,
            orders_submitted INTEGER DEFAULT 0,
            orders_filled INTEGER DEFAULT 0,
            market_state_at_execution TEXT DEFAULT 'CLOSED'
        )
    """)
    con.commit()
    con.close()


def save(log: ExecutionLog, db_path: str = TRADE_DB_PATH) -> None:
    _ensure_table(db_path)
    con = _conn(db_path)
    con.execute(
        "INSERT INTO execution_log (session_id, portfolio_id, started_at, finished_at,"
        " execution_time_ms, job_executed, success, exception,"
        " trades_executed, orders_submitted, orders_filled, market_state_at_execution)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            log.session_id, log.portfolio_id,
            log.started_at.isoformat() if log.started_at else None,
            log.finished_at.isoformat() if log.finished_at else None,
            log.execution_time_ms, log.job_executed,
            int(log.success), log.exception,
            log.trades_executed, log.orders_submitted,
            log.orders_filled, log.market_state_at_execution,
        ),
    )
    con.commit()
    con.close()


def get_recent_executions(n: int = 20, db_path: str = TRADE_DB_PATH) -> list[dict]:
    _ensure_table(db_path)
    con = _conn(db_path)
    rows = con.execute(
        "SELECT * FROM execution_log ORDER BY log_id DESC LIMIT ?", (n,)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_health_summary(db_path: str = TRADE_DB_PATH) -> HealthSummary:
    _ensure_table(db_path)
    con = _conn(db_path)
    rows = con.execute(
        "SELECT started_at, success, execution_time_ms FROM execution_log"
        " ORDER BY log_id DESC LIMIT 20"
    ).fetchall()
    con.close()

    summary = HealthSummary()
    if not rows:
        return summary

    # Last execution time
    try:
        summary.last_execution_time = datetime.datetime.fromisoformat(rows[0]["started_at"])
    except Exception:
        pass

    # Last success time
    for r in rows:
        if r["success"]:
            try:
                summary.last_success_time = datetime.datetime.fromisoformat(r["started_at"])
            except Exception:
                pass
            break

    # Consecutive failures from most recent
    for r in rows:
        if r["success"]:
            break
        summary.consecutive_failures += 1

    # Average execution time (last 20)
    times = [r["execution_time_ms"] for r in rows if r["execution_time_ms"]]
    summary.avg_execution_time_ms = sum(times) / len(times) if times else 0.0

    # Cron status — compare in UTC-aware space
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    last_utc = summary.last_execution_time
    if last_utc is not None and last_utc.tzinfo is None:
        last_utc = last_utc.replace(tzinfo=datetime.timezone.utc)
    mins_since = (
        (now_utc - last_utc).total_seconds() / 60 if last_utc else 9999
    )
    if summary.consecutive_failures >= 3 or mins_since > 30:
        summary.cron_status = "Down"
    elif summary.consecutive_failures >= 2 or mins_since > 10:
        summary.cron_status = "Degraded"
    else:
        summary.cron_status = "Healthy"

    return summary
