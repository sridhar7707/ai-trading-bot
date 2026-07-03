"""Trading session state machine — one DB row per calendar day."""
from __future__ import annotations

import datetime
import sqlite3
from dataclasses import dataclass
from typing import Optional

from config import TRADE_DB_PATH


@dataclass
class Session:
    session_id: int
    portfolio_id: int
    session_date: datetime.date
    state: str
    initialized: bool
    startup_completed: bool
    shutdown_completed: bool
    last_cycle_time: Optional[str]
    next_cycle_time: Optional[str]
    trades_today: int
    cycles_today: int
    market_open: bool
    notes: Optional[str]


def _conn(db_path: str = TRADE_DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def _ensure_table(db_path: str = TRADE_DB_PATH) -> None:
    con = _conn(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS trading_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER DEFAULT 1,
            session_date DATE NOT NULL,
            state TEXT DEFAULT 'PREMARKET',
            initialized INTEGER DEFAULT 0,
            startup_completed INTEGER DEFAULT 0,
            shutdown_completed INTEGER DEFAULT 0,
            last_cycle_time TEXT,
            next_cycle_time TEXT,
            trades_today INTEGER DEFAULT 0,
            cycles_today INTEGER DEFAULT 0,
            market_open INTEGER DEFAULT 0,
            notes TEXT,
            UNIQUE(portfolio_id, session_date)
        )
    """)
    con.commit()
    con.close()


def _row_to_session(row: sqlite3.Row) -> Session:
    return Session(
        session_id=row["session_id"],
        portfolio_id=row["portfolio_id"],
        session_date=datetime.date.fromisoformat(str(row["session_date"])),
        state=row["state"] or "PREMARKET",
        initialized=bool(row["initialized"]),
        startup_completed=bool(row["startup_completed"]),
        shutdown_completed=bool(row["shutdown_completed"]),
        last_cycle_time=row["last_cycle_time"],
        next_cycle_time=row["next_cycle_time"],
        trades_today=int(row["trades_today"] or 0),
        cycles_today=int(row["cycles_today"] or 0),
        market_open=bool(row["market_open"]),
        notes=row["notes"],
    )


def get_today_session(portfolio_id: int = 1, db_path: str = TRADE_DB_PATH) -> Optional[Session]:
    _ensure_table(db_path)
    today = datetime.date.today().isoformat()
    con = _conn(db_path)
    row = con.execute(
        "SELECT * FROM trading_sessions WHERE portfolio_id=? AND session_date=?",
        (portfolio_id, today),
    ).fetchone()
    con.close()
    return _row_to_session(row) if row else None


def create_session(portfolio_id: int = 1,
                   session_date: Optional[datetime.date] = None,
                   db_path: str = TRADE_DB_PATH) -> Session:
    _ensure_table(db_path)
    d = (session_date or datetime.date.today()).isoformat()
    con = _conn(db_path)
    con.execute(
        "INSERT OR IGNORE INTO trading_sessions"
        " (portfolio_id, session_date, state, initialized, startup_completed, shutdown_completed)"
        " VALUES (?,?,'PREMARKET',0,0,0)",
        (portfolio_id, d),
    )
    con.commit()
    row = con.execute(
        "SELECT * FROM trading_sessions WHERE portfolio_id=? AND session_date=?",
        (portfolio_id, d),
    ).fetchone()
    con.close()
    return _row_to_session(row)


def update_state(session: Session, state: str, db_path: str = TRADE_DB_PATH) -> None:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    con = _conn(db_path)
    con.execute(
        "UPDATE trading_sessions SET state=?, last_cycle_time=?, cycles_today=cycles_today+1"
        " WHERE session_id=?",
        (state, now, session.session_id),
    )
    con.commit()
    con.close()
    session.state = state
    session.last_cycle_time = now


def increment_trades(session: Session, count: int = 1, db_path: str = TRADE_DB_PATH) -> None:
    con = _conn(db_path)
    con.execute(
        "UPDATE trading_sessions SET trades_today=trades_today+? WHERE session_id=?",
        (count, session.session_id),
    )
    con.commit()
    con.close()
    session.trades_today += count


def is_startup_needed(session: Session) -> bool:
    return not session.initialized and not session.startup_completed


def is_shutdown_needed(session: Session) -> bool:
    return session.state == "CLOSED" and not session.shutdown_completed


def mark_initialized(session: Session, db_path: str = TRADE_DB_PATH) -> None:
    con = _conn(db_path)
    con.execute(
        "UPDATE trading_sessions SET initialized=1 WHERE session_id=?",
        (session.session_id,),
    )
    con.commit()
    con.close()
    session.initialized = True


def mark_startup_complete(session: Session, db_path: str = TRADE_DB_PATH) -> None:
    con = _conn(db_path)
    con.execute(
        "UPDATE trading_sessions SET startup_completed=1 WHERE session_id=?",
        (session.session_id,),
    )
    con.commit()
    con.close()
    session.startup_completed = True


def mark_shutdown_complete(session: Session, db_path: str = TRADE_DB_PATH) -> None:
    con = _conn(db_path)
    con.execute(
        "UPDATE trading_sessions SET shutdown_completed=1 WHERE session_id=?",
        (session.session_id,),
    )
    con.commit()
    con.close()
    session.shutdown_completed = True
