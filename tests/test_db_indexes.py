"""Verify that init_db() creates all required indexes and the query_metrics table.

Index tests use :memory: (fast, no filesystem). The WAL test requires a real
file because WAL mode is not supported on in-memory databases.
"""
from __future__ import annotations

import sqlite3

import pytest

from bot._main_db import init_db


@pytest.fixture
def mem_db() -> sqlite3.Connection:
    conn = init_db(":memory:")
    yield conn
    conn.close()


def _index_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {r[0] for r in rows}


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


# ── Index existence ────────────────────────────────────────────────────────────

def test_idx_signal_log_sym_ts(mem_db):
    assert "idx_signal_log_sym_ts" in _index_names(mem_db)


def test_idx_trades_sym_ts(mem_db):
    assert "idx_trades_sym_ts" in _index_names(mem_db)


def test_idx_trades_action(mem_db):
    assert "idx_trades_action" in _index_names(mem_db)


def test_idx_signal_log_action(mem_db):
    assert "idx_signal_log_action" in _index_names(mem_db)


# ── query_metrics table ────────────────────────────────────────────────────────

def test_query_metrics_table_exists(mem_db):
    assert "query_metrics" in _table_names(mem_db)


def test_query_metrics_columns(mem_db):
    cols = {
        row[1]
        for row in mem_db.execute("PRAGMA table_info(query_metrics)").fetchall()
    }
    assert {"query_name", "avg_ms", "max_ms", "calls", "last_run"} <= cols


def test_query_metrics_primary_key(mem_db):
    # query_name is the PK — duplicate upsert should succeed (ON CONFLICT update)
    mem_db.execute(
        "INSERT INTO query_metrics (query_name, avg_ms, max_ms, calls, last_run) "
        "VALUES ('test.query', 10.0, 20.0, 1, '2026-01-01')"
    )
    mem_db.execute(
        "INSERT INTO query_metrics (query_name, avg_ms, max_ms, calls, last_run) "
        "VALUES ('test.query', 15.0, 25.0, 2, '2026-01-02') "
        "ON CONFLICT(query_name) DO UPDATE SET calls = calls + excluded.calls"
    )
    row = mem_db.execute(
        "SELECT calls FROM query_metrics WHERE query_name = 'test.query'"
    ).fetchone()
    assert row[0] == 3


# ── WAL mode (requires real file) ─────────────────────────────────────────────

def test_wal_mode_is_set(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    assert mode == "wal", f"Expected WAL mode, got {mode!r}"
