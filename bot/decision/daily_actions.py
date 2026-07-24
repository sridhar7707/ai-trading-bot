"""Daily action recorder for the Decision Workspace.

The table is created lazily on first write so init_db() needs no changes.
Records one row per pending AI recommendation per trading session.
"""
from __future__ import annotations

import sqlite3
from datetime import date
from loguru import logger as _logger

_DDL = """
CREATE TABLE IF NOT EXISTS daily_actions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_date      TEXT NOT NULL,
    symbol            TEXT,
    action_type       TEXT NOT NULL,
    reasoning         TEXT,
    confidence        INTEGER NOT NULL DEFAULT 0,
    expected_impact   TEXT,
    recommended_time  TEXT NOT NULL DEFAULT 'Today',
    status            TEXT NOT NULL DEFAULT 'pending',
    estimated_minutes INTEGER NOT NULL DEFAULT 2,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_daily_actions_date "
    "ON daily_actions (session_date, status)"
)


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(_DDL)
    conn.execute(_INDEX)


def record(
    conn: sqlite3.Connection,
    action_type: str,
    symbol: str | None = None,
    reasoning: str = "",
    confidence: int = 0,
    expected_impact: str = "",
    recommended_time: str = "Today",
    estimated_minutes: int = 2,
    status: str = "pending",
) -> None:
    """Insert a daily action. Pending actions are upserted; executed actions are always inserted."""
    _ensure_table(conn)
    today = str(date.today())
    if status == "pending":
        existing = conn.execute(
            "SELECT id FROM daily_actions "
            "WHERE session_date = ? AND symbol IS ? AND action_type = ? AND status = 'pending'",
            (today, symbol, action_type),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE daily_actions SET reasoning = ?, confidence = ?, "
                "expected_impact = ?, estimated_minutes = ? WHERE id = ?",
                (reasoning, confidence, expected_impact, estimated_minutes, existing[0]),
            )
            conn.commit()
            return
    conn.execute(
        "INSERT INTO daily_actions "
        "(session_date, symbol, action_type, reasoning, confidence, "
        "expected_impact, recommended_time, estimated_minutes, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (today, symbol, action_type, reasoning, confidence,
         expected_impact, recommended_time, estimated_minutes, status),
    )
    conn.commit()


def mark_executed(conn: sqlite3.Connection, action_id: int) -> bool:
    try:
        _ensure_table(conn)
        cur = conn.execute(
            "UPDATE daily_actions SET status = 'executed' WHERE id = ?",
            (action_id,),
        )
        conn.commit()
        return cur.rowcount > 0
    except Exception as exc:
        _logger.error(f"daily_actions.mark_executed({action_id}): {exc}")
        return False


def get_pending(conn: sqlite3.Connection) -> list[dict]:
    """Return today's pending actions ordered by confidence desc."""
    today = str(date.today())
    try:
        rows = conn.execute(
            "SELECT id, symbol, action_type, reasoning, confidence, "
            "expected_impact, recommended_time, estimated_minutes "
            "FROM daily_actions WHERE session_date = ? AND status = 'pending' "
            "ORDER BY confidence DESC",
            (today,),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return []
        raise
    return [
        {
            "id": r[0], "symbol": r[1] or "", "action_type": r[2],
            "reasoning": r[3] or "", "confidence": r[4] or 0,
            "expected_impact": r[5] or "", "recommended_time": r[6] or "Today",
            "minutes": r[7] or 2,
        }
        for r in rows
    ]
