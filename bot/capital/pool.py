"""CapitalPool — first-class managed capital concept.

The bot sizes positions against `pool.tradeable_cash`, not the full Alpaca account,
so the AI can never exceed its managed allocation. The ledger is append-only;
pool balance fields are kept in sync but can always be recomputed from ledger events.

Tables are created lazily — no changes to _main_db.py required.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

_DDL_POOLS = """
CREATE TABLE IF NOT EXISTS capital_pools (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL DEFAULT 'default',
    status           TEXT NOT NULL DEFAULT 'active',
    allocated_amount REAL NOT NULL DEFAULT 0.0,
    available_cash   REAL NOT NULL DEFAULT 0.0,
    invested_amount  REAL NOT NULL DEFAULT 0.0,
    reserve          REAL NOT NULL DEFAULT 0.0,
    realized_profit  REAL NOT NULL DEFAULT 0.0,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_LEDGER = """
CREATE TABLE IF NOT EXISTS capital_ledger (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    pool_id       INTEGER NOT NULL,
    event_type    TEXT NOT NULL,
    amount        REAL NOT NULL,
    balance_after REAL NOT NULL,
    symbol        TEXT,
    notes         TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_IDX_LEDGER = (
    "CREATE INDEX IF NOT EXISTS idx_capital_ledger_pool "
    "ON capital_ledger (pool_id, created_at)"
)


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(_DDL_POOLS)
    conn.execute(_DDL_LEDGER)
    conn.execute(_IDX_LEDGER)


@dataclass
class CapitalPool:
    id: int
    name: str
    allocated_amount: float
    available_cash: float
    invested_amount: float
    reserve: float
    realized_profit: float

    @property
    def tradeable_cash(self) -> float:
        """Cash available for new positions (available minus reserve)."""
        return max(0.0, self.available_cash - self.reserve)

    @property
    def total_value(self) -> float:
        """Available cash + current open-position cost basis."""
        return self.available_cash + self.invested_amount


def load_active_pool(
    conn: sqlite3.Connection, initial_amount: float = 1000.0
) -> CapitalPool:
    """Load the active pool, creating a default one if none exists."""
    _ensure_tables(conn)
    row = conn.execute(
        "SELECT id, name, allocated_amount, available_cash, invested_amount, "
        "reserve, realized_profit FROM capital_pools "
        "WHERE status = 'active' ORDER BY id ASC LIMIT 1"
    ).fetchone()
    if not row:
        cur = conn.execute(
            "INSERT INTO capital_pools "
            "(name, status, allocated_amount, available_cash) VALUES (?, 'active', ?, ?)",
            ("default", initial_amount, initial_amount),
        )
        append_ledger(conn, cur.lastrowid, "deposit", initial_amount, initial_amount,
                      notes="Initial allocation")
        conn.commit()
        row = conn.execute(
            "SELECT id, name, allocated_amount, available_cash, invested_amount, "
            "reserve, realized_profit FROM capital_pools "
            "WHERE status = 'active' ORDER BY id ASC LIMIT 1"
        ).fetchone()
    return CapitalPool(
        id=row[0], name=row[1], allocated_amount=row[2], available_cash=row[3],
        invested_amount=row[4], reserve=row[5], realized_profit=row[6],
    )


def update_on_buy(
    conn: sqlite3.Connection, pool_id: int, notional: float
) -> None:
    """Move `notional` from available_cash to invested_amount on a BUY fill."""
    conn.execute(
        "UPDATE capital_pools SET "
        "available_cash  = available_cash  - ?, "
        "invested_amount = invested_amount + ?, "
        "updated_at = datetime('now') WHERE id = ?",
        (notional, notional, pool_id),
    )
    row = conn.execute(
        "SELECT available_cash FROM capital_pools WHERE id = ?", (pool_id,)
    ).fetchone()
    append_ledger(conn, pool_id, "buy", -notional, row[0] if row else 0.0)
    conn.commit()


def update_on_sell(
    conn: sqlite3.Connection, pool_id: int, cost_basis: float, fill_value: float
) -> None:
    """Return fill proceeds to available_cash; book realized P&L."""
    pnl = fill_value - cost_basis
    conn.execute(
        "UPDATE capital_pools SET "
        "available_cash  = available_cash  + ?, "
        "invested_amount = MAX(0.0, invested_amount - ?), "
        "realized_profit = realized_profit + ?, "
        "updated_at = datetime('now') WHERE id = ?",
        (fill_value, cost_basis, pnl, pool_id),
    )
    row = conn.execute(
        "SELECT available_cash FROM capital_pools WHERE id = ?", (pool_id,)
    ).fetchone()
    append_ledger(conn, pool_id, "sell", fill_value, row[0] if row else 0.0)
    conn.commit()


def append_ledger(
    conn: sqlite3.Connection,
    pool_id: int,
    event_type: str,
    amount: float,
    balance_after: float,
    symbol: str | None = None,
    notes: str | None = None,
) -> None:
    """Append an immutable event to the capital ledger."""
    conn.execute(
        "INSERT INTO capital_ledger "
        "(pool_id, event_type, amount, balance_after, symbol, notes) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (pool_id, event_type, amount, balance_after, symbol, notes),
    )
