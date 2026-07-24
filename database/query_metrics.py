"""In-memory query timing accumulator.

Records avg_ms / max_ms / calls per named query in memory during the trading
session. flush_to_db() upserts the session's stats into the query_metrics table
at EOD — one write, not one write per query execution.

Usage:
    from database.query_metrics import timed_query

    @timed_query("dashboard.get_trades")
    def get_trades_df() -> pd.DataFrame:
        ...
"""
from __future__ import annotations

import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from typing import Callable

from loguru import logger

SLOW_MS: float = 500.0  # log a warning for any query exceeding this


@dataclass
class _Stats:
    calls: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    last_run: str = ""


_stats: dict[str, _Stats] = defaultdict(_Stats)
_lock = threading.Lock()


def record(query_name: str, elapsed_ms: float) -> None:
    """Thread-safe accumulation of one timing sample."""
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        s = _stats[query_name]
        s.calls += 1
        s.total_ms += elapsed_ms
        s.max_ms = max(s.max_ms, elapsed_ms)
        s.last_run = now
    if elapsed_ms > SLOW_MS:
        logger.warning(f"Slow query [{query_name}]: {elapsed_ms:.0f} ms")


def timed_query(query_name: str) -> Callable:
    """Decorator — wraps a function and records its wall-clock time."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                record(query_name, (time.perf_counter() - t0) * 1000.0)
        return wrapper
    return decorator


def flush_to_db(conn: sqlite3.Connection) -> None:
    """Upsert accumulated session stats into query_metrics. Call once at EOD."""
    with _lock:
        snapshot = {k: (_Stats(s.calls, s.total_ms, s.max_ms, s.last_run))
                    for k, s in _stats.items()}
    if not snapshot:
        return
    for name, s in snapshot.items():
        avg_ms = s.total_ms / s.calls if s.calls else 0.0
        conn.execute(
            """INSERT INTO query_metrics (query_name, avg_ms, max_ms, calls, last_run)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(query_name) DO UPDATE SET
                   avg_ms   = (avg_ms * calls + excluded.avg_ms * excluded.calls)
                              / (calls + excluded.calls),
                   max_ms   = MAX(max_ms, excluded.max_ms),
                   calls    = calls + excluded.calls,
                   last_run = excluded.last_run""",
            (name, avg_ms, s.max_ms, s.calls, s.last_run),
        )
    conn.commit()
    logger.info(f"query_metrics flushed: {len(snapshot)} entries")


def current_stats() -> dict[str, dict]:
    """Return live stats snapshot (for logging / health checks)."""
    with _lock:
        return {
            name: {
                "calls":   s.calls,
                "avg_ms":  round(s.total_ms / s.calls, 1) if s.calls else 0.0,
                "max_ms":  round(s.max_ms, 1),
                "last_run": s.last_run,
            }
            for name, s in _stats.items()
        }
