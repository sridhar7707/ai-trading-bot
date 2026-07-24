"""Tests for bot/capital/pool.py — all tests use an in-memory SQLite DB."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ.setdefault("_BOT_LOG_HANDLER_ADDED", "1")

from bot.capital.pool import (
    CapitalPool,
    append_ledger,
    deposit,
    load_active_pool,
    set_reserve,
    update_on_buy,
    update_on_sell,
    withdraw,
)


@pytest.fixture
def mem_db():
    con = sqlite3.connect(":memory:")
    yield con
    con.close()


@pytest.fixture
def pool_db(mem_db):
    """mem_db with an active pool of $5000 already loaded."""
    pool = load_active_pool(mem_db, 5000.0)
    return mem_db, pool


# ── 1. load_active_pool creates default ───────────────────────────────────────

def test_load_active_pool_creates_default(mem_db):
    pool = load_active_pool(mem_db, 5000.0)
    assert pool.allocated_amount == 5000.0
    assert pool.available_cash == 5000.0
    row = mem_db.execute("SELECT event_type FROM capital_ledger").fetchone()
    assert row is not None
    assert row[0] == "deposit"


# ── 2. load_active_pool returns existing ──────────────────────────────────────

def test_load_active_pool_returns_existing(mem_db):
    p1 = load_active_pool(mem_db, 5000.0)
    p2 = load_active_pool(mem_db, 9999.0)   # different amount — must be ignored
    assert p2.id == p1.id
    assert p2.allocated_amount == 5000.0
    count = mem_db.execute("SELECT COUNT(*) FROM capital_pools").fetchone()[0]
    assert count == 1


# ── 3. update_on_buy moves cash to invested ───────────────────────────────────

def test_update_on_buy_moves_cash_to_invested(pool_db):
    con, pool = pool_db
    update_on_buy(con, pool.id, 1000.0)
    row = con.execute(
        "SELECT available_cash, invested_amount FROM capital_pools WHERE id=?",
        (pool.id,),
    ).fetchone()
    assert row[0] == pytest.approx(4000.0)
    assert row[1] == pytest.approx(1000.0)


# ── 4. update_on_sell books profit ────────────────────────────────────────────

def test_update_on_sell_books_pnl(pool_db):
    con, pool = pool_db
    update_on_buy(con, pool.id, 1000.0)
    update_on_sell(con, pool.id, cost_basis=1000.0, fill_value=1200.0)
    row = con.execute(
        "SELECT available_cash, invested_amount, realized_profit FROM capital_pools WHERE id=?",
        (pool.id,),
    ).fetchone()
    assert row[0] == pytest.approx(5200.0)
    assert row[1] == pytest.approx(0.0)
    assert row[2] == pytest.approx(200.0)


# ── 5. update_on_sell books loss ──────────────────────────────────────────────

def test_update_on_sell_loss(pool_db):
    con, pool = pool_db
    update_on_buy(con, pool.id, 1000.0)
    update_on_sell(con, pool.id, cost_basis=1000.0, fill_value=800.0)
    row = con.execute(
        "SELECT realized_profit FROM capital_pools WHERE id=?", (pool.id,)
    ).fetchone()
    assert row[0] == pytest.approx(-200.0)


# ── 6. ledger entry on buy ────────────────────────────────────────────────────

def test_ledger_entry_on_buy(pool_db):
    con, pool = pool_db
    con.execute("DELETE FROM capital_ledger")
    update_on_buy(con, pool.id, 1000.0)
    row = con.execute(
        "SELECT event_type, amount FROM capital_ledger WHERE event_type='buy'"
    ).fetchone()
    assert row is not None
    assert row[0] == "buy"
    assert row[1] == pytest.approx(-1000.0)


# ── 7. ledger entry on sell ───────────────────────────────────────────────────

def test_ledger_entry_on_sell(pool_db):
    con, pool = pool_db
    con.execute("DELETE FROM capital_ledger")
    update_on_buy(con, pool.id, 1000.0)
    update_on_sell(con, pool.id, cost_basis=1000.0, fill_value=1200.0)
    row = con.execute(
        "SELECT event_type FROM capital_ledger WHERE event_type='sell'"
    ).fetchone()
    assert row is not None
    assert row[0] == "sell"


# ── 8. deposit increases cash and allocated ───────────────────────────────────

def test_deposit_increases_cash_and_allocated(pool_db):
    con, pool = pool_db
    deposit(con, pool.id, 500.0)
    row = con.execute(
        "SELECT available_cash, allocated_amount FROM capital_pools WHERE id=?",
        (pool.id,),
    ).fetchone()
    assert row[0] == pytest.approx(5500.0)
    assert row[1] == pytest.approx(5500.0)


# ── 9. withdraw decreases cash ────────────────────────────────────────────────

def test_withdraw_decreases_cash(pool_db):
    con, pool = pool_db
    withdraw(con, pool.id, 300.0)
    row = con.execute(
        "SELECT available_cash FROM capital_pools WHERE id=?", (pool.id,)
    ).fetchone()
    assert row[0] == pytest.approx(4700.0)


# ── 10. withdraw tracks profit_withdrawn ──────────────────────────────────────

def test_withdraw_tracks_profit_withdrawn(pool_db):
    con, pool = pool_db
    update_on_buy(con, pool.id, 1000.0)
    update_on_sell(con, pool.id, cost_basis=1000.0, fill_value=1200.0)  # realized_profit = 200
    withdraw(con, pool.id, 200.0)
    row = con.execute(
        "SELECT profit_withdrawn FROM capital_pools WHERE id=?", (pool.id,)
    ).fetchone()
    assert row[0] == pytest.approx(200.0)


# ── 11. withdrawable_profit property ──────────────────────────────────────────

def test_withdrawable_profit_property():
    pool = CapitalPool(
        id=1, name="test", allocated_amount=5000.0, available_cash=5000.0,
        invested_amount=0.0, reserve=0.0,
        realized_profit=500.0, profit_withdrawn=200.0,
    )
    assert pool.withdrawable_profit == pytest.approx(300.0)


# ── 12. tradeable_cash respects reserve ───────────────────────────────────────

def test_tradeable_cash_respects_reserve():
    pool = CapitalPool(
        id=1, name="test", allocated_amount=5000.0, available_cash=3000.0,
        invested_amount=0.0, reserve=500.0,
        realized_profit=0.0, profit_withdrawn=0.0,
    )
    assert pool.tradeable_cash == pytest.approx(2500.0)


# ── 13. set_reserve updates DB ────────────────────────────────────────────────

def test_set_reserve_updates_db(pool_db):
    con, pool = pool_db
    set_reserve(con, pool.id, 750.0)
    reloaded = load_active_pool(con)
    assert reloaded.reserve == pytest.approx(750.0)


# ── 14. ledger symbol tracking ────────────────────────────────────────────────

def test_ledger_symbol_tracking(pool_db):
    con, pool = pool_db
    update_on_buy(con, pool.id, 500.0, symbol="AAPL")
    row = con.execute(
        "SELECT symbol FROM capital_ledger WHERE event_type='buy'"
    ).fetchone()
    assert row is not None
    assert row[0] == "AAPL"


# ── 15. atomic buy/sell sequence ──────────────────────────────────────────────

def test_atomic_buy_sell_sequence(pool_db):
    con, pool = pool_db
    update_on_buy(con, pool.id, 1000.0, symbol="TSLA")
    update_on_sell(con, pool.id, cost_basis=1000.0, fill_value=1100.0, symbol="TSLA")
    row = con.execute(
        "SELECT available_cash, invested_amount FROM capital_pools WHERE id=?",
        (pool.id,),
    ).fetchone()
    assert row[1] == pytest.approx(0.0)       # invested back to zero
    assert row[0] == pytest.approx(5100.0)    # original 5000 + 100 profit
