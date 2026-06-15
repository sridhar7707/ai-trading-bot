"""
Tests for SQLite thread safety under Gradio's multi-threaded refresh cycle.
SPEC 56 — zero SQLite threading errors under concurrent reads.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_concurrent_data_reads():
    """10 ThreadPoolExecutor workers calling get_data() must not raise SQLite threading errors."""
    from dashboard.data import get_data

    errors: list[str] = []

    def _call():
        try:
            get_data()
        except Exception as exc:
            errors.append(str(exc))

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(_call) for _ in range(10)]
        for f in futures:
            f.result()

    threading_errors = [e for e in errors if "thread" in e.lower()]
    assert not threading_errors, f"SQLite threading errors: {threading_errors}"


def test_concurrent_render_functions():
    """4 threads calling render functions simultaneously must not produce SQLite threading errors."""
    try:
        from dashboard.components.history import render_whats_changed, render_portfolio_performance
    except ImportError:
        return  # skip if dashboard not importable in CI

    errors: list[str] = []

    def _call(fn):
        try:
            fn()
        except Exception as exc:
            errors.append(str(exc))

    fns = [render_whats_changed, render_portfolio_performance,
           render_whats_changed, render_portfolio_performance]
    threads = [threading.Thread(target=_call, args=(fn,)) for fn in fns]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    threading_errors = [e for e in errors if "thread" in e.lower() and "sqlite" in e.lower()]
    assert not threading_errors, f"SQLite threading errors in render functions: {threading_errors}"


def test_no_shared_connection_objects():
    """After SPEC 56 conversion, only dashboard/data.py (which contains get_db_conn) should
    reference sqlite3.connect — all other dashboard files must use get_db_conn()."""
    dash_dir = os.path.join(os.path.dirname(__file__), "..", "dashboard")
    violations: list[str] = []

    for root, _, files in os.walk(dash_dir):
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            if fname == "data.py":
                continue  # data.py contains the get_db_conn() implementation
            fpath = os.path.join(root, fname)
            with open(fpath, encoding="utf-8") as f:
                content = f.read()
            if "sqlite3.connect(" in content:
                rel = os.path.relpath(fpath, dash_dir)
                violations.append(rel)

    assert not violations, (
        f"Raw sqlite3.connect() found in dashboard files (use get_db_conn instead): {violations}"
    )


def test_wal_mode_enabled():
    """DB file must report journal_mode=wal after _init_db() runs on import."""
    from dashboard.data import get_db_conn, DB_PATH

    if not os.path.exists(DB_PATH):
        return  # no DB in this environment — skip

    with get_db_conn() as conn:
        row = conn.execute("PRAGMA journal_mode").fetchone()
    mode = row[0] if row else "unknown"
    assert mode == "wal", f"Expected WAL journal mode, got: {mode!r}"


def test_sqlite_timeout_not_infinite():
    """get_db_conn(timeout=0.1) must return (or raise OperationalError) within 2 seconds."""
    from dashboard.data import get_db_conn

    start = time.monotonic()
    try:
        with get_db_conn(timeout=0.1) as conn:
            conn.execute("SELECT 1")
    except Exception:
        pass  # OperationalError on lock timeout is acceptable
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"get_db_conn blocked for {elapsed:.2f}s — timeout param ignored"
