"""User settings persistence — key/value store backed by SQLite user_settings table."""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Generator

from loguru import logger

from config import TRADE_DB_PATH as _DB_PATH   # single source of truth — eliminates env-var split-brain

_DEFAULTS: dict[str, str] = {
    "risk_tolerance":        "Moderate",
    "benchmark":             "SPY",
    "max_position_pct":      "0.20",
    "max_drawdown_pct":      "0.12",
    "stop_loss_pct":         "0.04",
    "notifications_enabled": "false",
}

# In-memory cache — refreshed every 30 s or immediately after any write
_cache: dict[str, str] | None = None
_cache_ts: float = 0.0
_CACHE_TTL: float = 30.0


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    con = sqlite3.connect(_DB_PATH, timeout=5.0)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
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


def _warm_cache() -> dict[str, str]:
    """Fetch all settings from DB into the module-level cache."""
    global _cache, _cache_ts
    result = dict(_DEFAULTS)
    if os.path.exists(_DB_PATH):
        try:
            with _conn() as con:
                for k, v in con.execute("SELECT key, value FROM user_settings").fetchall():
                    result[k] = v
        except Exception as exc:
            logger.warning(f"user_settings: cache refresh failed — {exc}")
    _cache = result
    _cache_ts = time.time()
    return result


def _get_cache() -> dict[str, str]:
    """Return cached settings, refreshing if stale (> 30 s) or cold."""
    if _cache is None or time.time() - _cache_ts > _CACHE_TTL:
        return _warm_cache()
    return _cache


def get_setting(key: str, default: str | None = None) -> str:
    """Return the stored setting, falling back to _DEFAULTS then `default`."""
    return _get_cache().get(key, _DEFAULTS.get(key, default or ""))


def save_setting(key: str, value: str) -> bool:
    """Upsert a setting. Returns True on success, False on failure (always logs errors)."""
    global _cache
    if not os.path.exists(_DB_PATH):
        logger.warning(f"user_settings.save_setting: DB not found at {_DB_PATH}")
        return False
    try:
        with _conn() as con:
            con.execute("""
                INSERT INTO user_settings (key, value, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(key) DO UPDATE
                    SET value      = excluded.value,
                        updated_at = excluded.updated_at
            """, (key, str(value)))
        _cache = None   # invalidate so next read fetches fresh values
        return True
    except Exception as exc:
        logger.error(f"user_settings.save_setting({key!r}): {exc}")
        return False


def get_all_settings() -> dict[str, str]:
    """Return all settings merged with defaults (DB values take precedence)."""
    return _warm_cache()
