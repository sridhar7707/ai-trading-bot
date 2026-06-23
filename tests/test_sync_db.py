from unittest.mock import patch
from bot.monitor.sync_db import push_db, pull_db


def test_push_db_skips_when_no_token():
    with patch("bot.monitor.sync_db._get_cfg", return_value=("trades.db", "repo/id", "")):
        result = push_db()
    assert result is False


def test_push_db_skips_when_no_repo():
    with patch("bot.monitor.sync_db._get_cfg", return_value=("trades.db", "", "token")):
        result = push_db()
    assert result is False


def test_push_db_skips_when_db_missing(tmp_path):
    missing = str(tmp_path / "nonexistent.db")
    with patch("bot.monitor.sync_db._get_cfg", return_value=(missing, "repo/id", "token")):
        result = push_db()
    assert result is False


def test_pull_db_skips_when_no_repo():
    with patch("bot.monitor.sync_db._get_cfg", return_value=("trades.db", "", "token")):
        result = pull_db()
    assert result is False


def test_pull_db_returns_true_for_fresh_local_db(tmp_path):
    db = tmp_path / "trades.db"
    db.touch()
    with patch("bot.monitor.sync_db._get_cfg", return_value=(str(db), "repo/id", "token")):
        result = pull_db(force=False)
    assert result is True
