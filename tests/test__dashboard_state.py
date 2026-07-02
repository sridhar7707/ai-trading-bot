import threading
from bot.monitor._dashboard_state import _last_sync, _pull_lock, _spy_cache


def test_last_sync_is_dict():
    assert isinstance(_last_sync, dict)
    assert "ok" in _last_sync
    assert "ts" in _last_sync
    assert "err" in _last_sync


def test_pull_lock_is_lock():
    assert isinstance(_pull_lock, type(threading.Lock()))


def test_spy_cache_is_dict():
    assert isinstance(_spy_cache, dict)


def test_state_is_mutable():
    original = _last_sync["ok"]
    _last_sync["ok"] = True
    assert _last_sync["ok"] is True
    _last_sync["ok"] = original
