"""Shared mutable state for dashboard modules — avoids circular imports."""
from __future__ import annotations
import threading

_last_sync: dict = {"ok": None, "ts": None, "err": ""}
_pull_lock = threading.Lock()
_spy_cache: dict = {}   # {start_iso: (today_str, ret)} — multi-slot, one entry per lookback date
