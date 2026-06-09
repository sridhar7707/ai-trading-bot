import sqlite3
from datetime import datetime, timezone, timedelta
import pytest
from bot.main import _opened_today, log_trade, init_db


@pytest.fixture
def db():
    """In-memory SQLite DB with the trades table."""
    con = sqlite3.connect(":memory:")
    con.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            action TEXT,
            shares REAL,
            price REAL,
            notional REAL,
            regime TEXT,
            portfolio_value REAL,
            pnl_pct REAL
        )
    """)
    con.commit()
    yield con
    con.close()


# --- _opened_today ---

def test_opened_today_true_when_buy_today(db):
    today = datetime.now(timezone.utc).isoformat()
    db.execute(
        "INSERT INTO trades VALUES (NULL,?,?,?,?,?,?,?,?,?)",
        (today, "AAPL", "BUY", 1.0, 150.0, 150.0, "RANGING", 10000.0, 0.0),
    )
    db.commit()
    assert _opened_today(db, "AAPL") is True


def test_opened_today_false_when_no_buy(db):
    assert _opened_today(db, "AAPL") is False


def test_opened_today_false_when_only_sell_today(db):
    today = datetime.now(timezone.utc).isoformat()
    db.execute(
        "INSERT INTO trades VALUES (NULL,?,?,?,?,?,?,?,?,?)",
        (today, "AAPL", "SELL", 1.0, 160.0, 0.0, "RANGING", 10160.0, 0.06),
    )
    db.commit()
    assert _opened_today(db, "AAPL") is False


def test_opened_today_false_for_yesterday_buy(db):
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    db.execute(
        "INSERT INTO trades VALUES (NULL,?,?,?,?,?,?,?,?,?)",
        (yesterday, "AAPL", "BUY", 1.0, 150.0, 150.0, "RANGING", 10000.0, 0.0),
    )
    db.commit()
    assert _opened_today(db, "AAPL") is False


def test_opened_today_symbol_isolation(db):
    today = datetime.now(timezone.utc).isoformat()
    db.execute(
        "INSERT INTO trades VALUES (NULL,?,?,?,?,?,?,?,?,?)",
        (today, "MSFT", "BUY", 1.0, 300.0, 300.0, "RANGING", 10000.0, 0.0),
    )
    db.commit()
    assert _opened_today(db, "AAPL") is False
    assert _opened_today(db, "MSFT") is True


# --- log_trade ---

def test_log_trade_stores_record(db):
    log_trade(db, "AAPL", "BUY", 1.0, 150.0, 150.0, "RANGING", 10000.0, 0.0)
    row = db.execute("SELECT symbol, action, price FROM trades").fetchone()
    assert row == ("AAPL", "BUY", 150.0)


def test_log_trade_timestamp_is_utc_iso(db):
    log_trade(db, "AAPL", "BUY", 1.0, 150.0, 150.0, "RANGING", 10000.0, 0.0)
    ts = db.execute("SELECT timestamp FROM trades").fetchone()[0]
    # Should be parseable and contain UTC offset (+00:00) or 'Z'
    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


# --- init_db ---

def test_init_db_creates_trades_table(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_trades.db")
    monkeypatch.setattr("bot.main.TRADE_DB_PATH", db_path)
    con = init_db()
    tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert ("trades",) in tables
    con.close()
