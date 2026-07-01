"""User settings persistence — key/value store backed by SQLite user_settings table."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Generator

_DB_PATH = os.getenv("DB_PATH", "trades.db")

_DEFAULTS: dict[str, str] = {
    "risk_tolerance":         "Moderate",
    "benchmark":              "SPY",
    "max_position_pct":       "0.20",
    "max_drawdown_pct":       "0.12",
    "max_sector_pct":         "0.30",
    "stop_loss_pct":          "0.04",
    "notifications_enabled":  "false",
}


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    con = sqlite3.connect(_DB_PATH, timeout=5.0)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_table() -> None:
    """Create user_settings table — called by init_db(); safe to call multiple times."""
    if not os.path.exists(_DB_PATH):
        return
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)


def get_setting(key: str, default: str | None = None) -> str:
    """Return the stored setting, falling back to _DEFAULTS then `default`."""
    fallback = _DEFAULTS.get(key, default or "")
    if not os.path.exists(_DB_PATH):
        return fallback
    try:
        with _conn() as con:
            row = con.execute(
                "SELECT value FROM user_settings WHERE key = ?", (key,)
            ).fetchone()
            return row[0] if row else fallback
    except Exception:
        return fallback


def save_setting(key: str, value: str) -> None:
    """Upsert a setting into user_settings."""
    if not os.path.exists(_DB_PATH):
        return
    try:
        with _conn() as con:
            con.execute("""
                INSERT INTO user_settings (key, value, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(key) DO UPDATE
                    SET value      = excluded.value,
                        updated_at = excluded.updated_at
            """, (key, str(value)))
    except Exception:
        pass


def get_all_settings() -> dict[str, str]:
    """Return all settings merged with defaults (DB values take precedence)."""
    result = dict(_DEFAULTS)
    if not os.path.exists(_DB_PATH):
        return result
    try:
        with _conn() as con:
            for k, v in con.execute("SELECT key, value FROM user_settings").fetchall():
                result[k] = v
    except Exception:
        pass
    return result
