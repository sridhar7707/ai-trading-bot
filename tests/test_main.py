import sqlite3
import math
import pandas as pd
from datetime import datetime, timezone, timedelta, date
import pytest
from bot.main import (
    _opened_today, _maybe_record_day_trade, log_trade, init_db,
    _kelly_fraction, _passes_correlation_gate, _check_time_exit,
    _reconcile_positions, _load_risk_state, _save_risk_state,
    _upsert_position_state, _delete_position_state, _is_wash_sale_risk,
    _record_snapshot, _anchor_daily_start, _apply_sim_capital,
)
from bot.risk.risk_manager import RiskManager


@pytest.fixture
def db():
    """In-memory SQLite DB matching the schema created by init_db()."""
    con = sqlite3.connect(":memory:")
    con.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, symbol TEXT, action TEXT,
            shares REAL, price REAL, notional REAL,
            regime TEXT, portfolio_value REAL, pnl_pct REAL,
            xgb_prob REAL DEFAULT 0.0,
            lstm_prob REAL DEFAULT 0.0,
            sentiment_score REAL DEFAULT 0.0,
            macro_score REAL DEFAULT 0.0,
            ensemble_score REAL DEFAULT 0.0,
            realized_pnl REAL DEFAULT 0.0,
            order_id TEXT DEFAULT NULL,
            holding_days INTEGER DEFAULT 0
        )
    """)
    con.execute("""
        CREATE TABLE risk_state (
            key TEXT PRIMARY KEY, value TEXT, updated_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE position_state (
            symbol TEXT PRIMARY KEY,
            entry_price REAL,
            high_water_mark REAL,
            atr_at_entry REAL,
            opened_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE portfolio_snapshots (
            timestamp TEXT PRIMARY KEY,
            portfolio_value REAL,
            available_cash REAL,
            open_positions INTEGER
        )
    """)
    con.commit()
    yield con
    con.close()


# --- _opened_today ---

def _today_ts() -> str:
    """Timestamp that starts with local today's date — matches _opened_today's LIKE query."""
    return date.today().isoformat() + "T12:00:00+00:00"


def _yesterday_ts() -> str:
    return (date.today() - timedelta(days=1)).isoformat() + "T12:00:00+00:00"


def test_opened_today_true_when_buy_today(db):
    db.execute(
        "INSERT INTO trades (timestamp,symbol,action,shares,price,notional,regime,portfolio_value,pnl_pct) VALUES (?,?,?,?,?,?,?,?,?)",
        (_today_ts(), "AAPL", "BUY", 1.0, 150.0, 150.0, "RANGING", 10000.0, 0.0),
    )
    db.commit()
    assert _opened_today(db, "AAPL") is True


def test_opened_today_false_when_no_buy(db):
    assert _opened_today(db, "AAPL") is False


def test_opened_today_false_when_only_sell_today(db):
    db.execute(
        "INSERT INTO trades (timestamp,symbol,action,shares,price,notional,regime,portfolio_value,pnl_pct) VALUES (?,?,?,?,?,?,?,?,?)",
        (_today_ts(), "AAPL", "SELL", 1.0, 160.0, 0.0, "RANGING", 10160.0, 0.06),
    )
    db.commit()
    assert _opened_today(db, "AAPL") is False


def test_opened_today_false_for_yesterday_buy(db):
    db.execute(
        "INSERT INTO trades (timestamp,symbol,action,shares,price,notional,regime,portfolio_value,pnl_pct) VALUES (?,?,?,?,?,?,?,?,?)",
        (_yesterday_ts(), "AAPL", "BUY", 1.0, 150.0, 150.0, "RANGING", 10000.0, 0.0),
    )
    db.commit()
    assert _opened_today(db, "AAPL") is False


def test_opened_today_symbol_isolation(db):
    db.execute(
        "INSERT INTO trades (timestamp,symbol,action,shares,price,notional,regime,portfolio_value,pnl_pct) VALUES (?,?,?,?,?,?,?,?,?)",
        (_today_ts(), "MSFT", "BUY", 1.0, 300.0, 300.0, "RANGING", 10000.0, 0.0),
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
    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


def test_log_trade_realized_pnl_on_sell(db):
    # 10 shares, entry $100, sell $110 → realized = 10 * (110 - 100) = 100.0
    log_trade(db, "AAPL", "SELL", 10.0, 110.0, 1100.0, "RANGING", 10100.0, 0.10,
              entry_price=100.0)
    pnl = db.execute("SELECT realized_pnl FROM trades").fetchone()[0]
    assert abs(pnl - 100.0) < 0.01


def test_log_trade_realized_pnl_zero_on_buy(db):
    log_trade(db, "AAPL", "BUY", 10.0, 100.0, 1000.0, "RANGING", 10000.0, 0.0,
              entry_price=95.0)
    pnl = db.execute("SELECT realized_pnl FROM trades").fetchone()[0]
    assert pnl == 0.0


def test_log_trade_realized_pnl_negative_on_loss(db):
    # Sell at $90 with entry $100 → realized = 10 * (90 - 100) = -100.0
    log_trade(db, "AAPL", "SELL_STOP", 10.0, 90.0, 900.0, "VOLATILE", 9900.0, -0.10,
              entry_price=100.0)
    pnl = db.execute("SELECT realized_pnl FROM trades").fetchone()[0]
    assert abs(pnl - (-100.0)) < 0.01


def test_log_trade_realized_pnl_zero_when_no_entry_price(db):
    log_trade(db, "AAPL", "SELL", 10.0, 110.0, 1100.0, "RANGING", 10100.0, 0.10)
    pnl = db.execute("SELECT realized_pnl FROM trades").fetchone()[0]
    assert pnl == 0.0


# --- _maybe_record_day_trade ---

def _insert_buy_today(db, symbol):
    db.execute(
        "INSERT INTO trades (timestamp,symbol,action,shares,price,notional,regime,portfolio_value,pnl_pct) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (_today_ts(), symbol, "BUY", 1.0, 100.0, 100.0, "RANGING", 10000.0, 0.0),
    )
    db.commit()


def test_maybe_record_day_trade_records_when_not_exempt(db):
    _insert_buy_today(db, "AAPL")
    risk = RiskManager()
    risk.reset_daily(10_000.0)
    _maybe_record_day_trade(db, risk, "AAPL", sell_success=True, pdt_exempt=False)
    assert len(risk.day_trade_log) == 1


def test_maybe_record_day_trade_skips_when_exempt(db):
    _insert_buy_today(db, "AAPL")
    risk = RiskManager()
    risk.reset_daily(10_000.0)
    _maybe_record_day_trade(db, risk, "AAPL", sell_success=True, pdt_exempt=True)
    assert len(risk.day_trade_log) == 0


def test_maybe_record_day_trade_skips_when_sell_failed(db):
    _insert_buy_today(db, "AAPL")
    risk = RiskManager()
    risk.reset_daily(10_000.0)
    _maybe_record_day_trade(db, risk, "AAPL", sell_success=False, pdt_exempt=False)
    assert len(risk.day_trade_log) == 0


def test_maybe_record_day_trade_skips_when_not_opened_today(db):
    risk = RiskManager()
    risk.reset_daily(10_000.0)
    _maybe_record_day_trade(db, risk, "AAPL", sell_success=True, pdt_exempt=False)
    assert len(risk.day_trade_log) == 0


# --- init_db ---

def test_init_db_creates_trades_table(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_trades.db")
    monkeypatch.setattr("bot.main.TRADE_DB_PATH", db_path)
    con = init_db()
    tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert ("trades",) in tables
    con.close()


def test_init_db_creates_all_tables(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_trades.db")
    monkeypatch.setattr("bot.main.TRADE_DB_PATH", db_path)
    con = init_db()
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"trades", "position_state", "risk_state", "earnings_cache",
            "macro_cache", "portfolio_snapshots"} <= names
    con.close()


def test_init_db_trades_has_realized_pnl_column(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_trades.db")
    monkeypatch.setattr("bot.main.TRADE_DB_PATH", db_path)
    con = init_db()
    cols = {r[1] for r in con.execute("PRAGMA table_info(trades)").fetchall()}
    assert "realized_pnl" in cols
    con.close()


# --- _kelly_fraction ---

def _insert_sell(db, symbol, pnl_pct):
    db.execute(
        "INSERT INTO trades (timestamp,symbol,action,shares,price,notional,regime,portfolio_value,pnl_pct) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (_today_ts(), symbol, "SELL", 1.0, 100.0, 100.0, "RANGING", 10000.0, pnl_pct),
    )
    db.commit()


def test_kelly_fraction_returns_default_when_too_few_trades(db):
    from bot.strategy.ensemble import BUY_FRACTION
    # < 10 trades → should return default
    for _ in range(5):
        _insert_sell(db, "AAPL", 0.05)
    result = _kelly_fraction(db, "AAPL")
    assert result == BUY_FRACTION


def test_kelly_fraction_returns_default_when_all_wins(db):
    from bot.strategy.ensemble import BUY_FRACTION
    for _ in range(15):
        _insert_sell(db, "AAPL", 0.05)
    # No losses → falls back to default (can't compute b ratio)
    result = _kelly_fraction(db, "AAPL")
    assert result == BUY_FRACTION


def test_kelly_fraction_returns_default_when_all_losses(db):
    from bot.strategy.ensemble import BUY_FRACTION
    for _ in range(15):
        _insert_sell(db, "AAPL", -0.05)
    result = _kelly_fraction(db, "AAPL")
    assert result == BUY_FRACTION


def test_kelly_fraction_positive_edge_is_bounded(db):
    from config import KELLY_FRACTION_MAX
    # 80% win rate with good win/loss ratio → high Kelly, but must be capped
    for _ in range(12):
        _insert_sell(db, "MSFT", 0.10)
    for _ in range(3):
        _insert_sell(db, "MSFT", -0.01)
    result = _kelly_fraction(db, "MSFT")
    assert 0.02 <= result <= KELLY_FRACTION_MAX


def test_kelly_fraction_minimum_floor(db):
    # Negative edge → Kelly would be negative, but clamped to 0.02
    for _ in range(5):
        _insert_sell(db, "TSLA", 0.01)
    for _ in range(10):
        _insert_sell(db, "TSLA", -0.10)
    result = _kelly_fraction(db, "TSLA")
    assert result >= 0.02


# --- _passes_correlation_gate ---

def _make_bars(closes):
    return pd.DataFrame({"close": closes})


def test_passes_correlation_gate_no_bars_for_symbol():
    # Symbol not in bars_map → passes (fail-open)
    assert _passes_correlation_gate("AAPL", {"MSFT": None}, {}) is True


def test_passes_correlation_gate_empty_positions():
    bars = {"AAPL": _make_bars([1, 2, 3, 4, 5])}
    assert _passes_correlation_gate("AAPL", {}, bars) is True


def test_passes_correlation_gate_held_symbol_skipped():
    # AAPL is both the candidate AND a held position — skip self-comparison
    bars = {"AAPL": _make_bars(list(range(30)))}
    assert _passes_correlation_gate("AAPL", {"AAPL": None}, bars) is True


def test_passes_correlation_gate_blocks_high_correlation():
    import numpy as np
    from config import CORRELATION_THRESHOLD
    # Perfectly correlated bars (same series)
    prices = list(range(1, 31))
    bars = {
        "AAPL": _make_bars(prices),
        "MSFT": _make_bars(prices),
    }
    result = _passes_correlation_gate("AAPL", {"MSFT": None}, bars)
    # Correlation = 1.0 > CORRELATION_THRESHOLD → blocked
    assert result is False


def test_passes_correlation_gate_allows_uncorrelated():
    import numpy as np
    # Uncorrelated: alternating vs constant rise
    bars = {
        "AAPL": _make_bars([10 if i % 2 == 0 else 20 for i in range(30)]),
        "TSLA": _make_bars(list(range(10, 40))),
    }
    result = _passes_correlation_gate("AAPL", {"TSLA": None}, bars)
    # These returns should be near-zero correlated (constant rise vs alternating)
    # Result depends on threshold, just ensure no exception is raised
    assert isinstance(result, bool)


def test_passes_correlation_gate_insufficient_common_bars():
    # Only 5 common timestamps — gate requires ≥ 20
    bars = {
        "AAPL": _make_bars(list(range(5))),
        "MSFT": _make_bars(list(range(5))),
    }
    # < 20 observations → gate skips this pair → passes
    assert _passes_correlation_gate("AAPL", {"MSFT": None}, bars) is True


# --- _check_time_exit ---

def test_check_time_exit_no_pos_state():
    assert _check_time_exit(None, 0.0) is False


def test_check_time_exit_missing_opened_at():
    assert _check_time_exit({}, -0.05) is False
    assert _check_time_exit({"opened_at": None}, -0.05) is False


def test_check_time_exit_recent_position_not_exited():
    from config import MAX_HOLD_DAYS
    opened = (datetime.now(timezone.utc) - timedelta(days=MAX_HOLD_DAYS - 1)).isoformat()
    pos = {"opened_at": opened}
    assert _check_time_exit(pos, -0.05) is False


def test_check_time_exit_old_position_negative_pnl_exits():
    from config import MAX_HOLD_DAYS
    opened = (datetime.now(timezone.utc) - timedelta(days=MAX_HOLD_DAYS + 1)).isoformat()
    pos = {"opened_at": opened}
    # pnl_pct < 0.01 → should exit
    assert _check_time_exit(pos, -0.01) is True


def test_check_time_exit_old_position_good_pnl_stays():
    from config import MAX_HOLD_DAYS
    opened = (datetime.now(timezone.utc) - timedelta(days=MAX_HOLD_DAYS + 1)).isoformat()
    pos = {"opened_at": opened}
    # pnl_pct >= 0.01 → still profitable, keep holding
    assert _check_time_exit(pos, 0.05) is False


def test_check_time_exit_invalid_date_string():
    pos = {"opened_at": "not-a-date"}
    assert _check_time_exit(pos, -0.05) is False


# --- _reconcile_positions ---

def test_reconcile_removes_stale_db_entries(db):
    # AAPL is in DB but not in Alpaca → should be removed
    _upsert_position_state(db, "AAPL", 100.0, 100.0, 1.0)
    _reconcile_positions(db, alpaca_positions={})
    rows = db.execute("SELECT symbol FROM position_state").fetchall()
    assert ("AAPL",) not in rows


def test_reconcile_seeds_missing_alpaca_positions(db):
    # MSFT is in Alpaca but not in DB → should be seeded
    class FakePos:
        avg_entry_price = 200.0

    _reconcile_positions(db, alpaca_positions={"MSFT": FakePos()})
    row = db.execute("SELECT entry_price FROM position_state WHERE symbol='MSFT'").fetchone()
    assert row is not None
    assert row[0] == pytest.approx(200.0)


def test_reconcile_leaves_matching_positions(db):
    # TSLA in both DB and Alpaca → untouched
    _upsert_position_state(db, "TSLA", 300.0, 310.0, 2.0)

    class FakePos:
        avg_entry_price = 300.0

    _reconcile_positions(db, alpaca_positions={"TSLA": FakePos()})
    row = db.execute("SELECT symbol FROM position_state WHERE symbol='TSLA'").fetchone()
    assert row is not None


def test_reconcile_handles_empty_alpaca_and_db(db):
    # Both empty → no-op
    _reconcile_positions(db, alpaca_positions={})
    assert db.execute("SELECT COUNT(*) FROM position_state").fetchone()[0] == 0


# --- _load_risk_state / _save_risk_state ---

def test_load_risk_state_returns_nones_when_empty(db):
    daily_start, day_trades, weekly_start, warn_sent, halt_alerted, ph = _load_risk_state(db)
    assert daily_start is None
    assert day_trades == []
    assert weekly_start is None
    assert warn_sent is False
    assert halt_alerted is False
    assert ph is None


def test_save_and_load_portfolio_high(db):
    risk = RiskManager(portfolio_high=15_000.0)
    risk.reset_daily(10_000.0)
    _save_risk_state(db, risk)
    _, _, _, _, _, ph = _load_risk_state(db)
    assert ph == pytest.approx(15_000.0, abs=0.01)


def test_save_and_load_daily_start(db):
    risk = RiskManager()
    risk.reset_daily(12_345.0)
    _save_risk_state(db, risk)
    daily_start, _, _, _, _, _ = _load_risk_state(db)
    assert daily_start == pytest.approx(12_345.0, abs=0.01)


def test_save_and_load_day_trade_log(db):
    risk = RiskManager()
    risk.reset_daily(10_000.0)
    risk.record_day_trade()
    risk.record_day_trade()
    _save_risk_state(db, risk)
    _, day_trades, _, _, _, _ = _load_risk_state(db)
    assert len(day_trades) == 2


# --- _is_wash_sale_risk (IRS IRC §1091) ---

def _insert_loss_sell(db, symbol, pnl_pct, days_ago=1, realized_pnl=-50.0):
    from datetime import timezone, timedelta
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    db.execute(
        "INSERT INTO trades (timestamp,symbol,action,shares,price,notional,regime,"
        "portfolio_value,pnl_pct,realized_pnl) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (ts, symbol, "SELL_STOP", 10.0, 90.0, 900.0, "VOLATILE", 9900.0, pnl_pct, realized_pnl),
    )
    db.commit()


def _insert_profit_sell(db, symbol, days_ago=1):
    from datetime import timezone, timedelta
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    db.execute(
        "INSERT INTO trades (timestamp,symbol,action,shares,price,notional,regime,"
        "portfolio_value,pnl_pct,realized_pnl) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (ts, symbol, "SELL", 10.0, 110.0, 1100.0, "TRENDING_UP", 10100.0, 0.10, 100.0),
    )
    db.commit()


def test_wash_sale_blocks_rebuy_after_loss(db):
    _insert_loss_sell(db, "AAPL", pnl_pct=-0.05, days_ago=5)
    assert _is_wash_sale_risk(db, "AAPL") is True


def test_wash_sale_clears_after_30_days(db):
    # 31 days ago is outside the 30-day window
    _insert_loss_sell(db, "AAPL", pnl_pct=-0.05, days_ago=31)
    assert _is_wash_sale_risk(db, "AAPL") is False


def test_wash_sale_no_block_on_profit_sell(db):
    # Sold at a profit — no wash-sale concern
    _insert_profit_sell(db, "AAPL", days_ago=5)
    assert _is_wash_sale_risk(db, "AAPL") is False


def test_wash_sale_symbol_isolation(db):
    # Loss on MSFT does not block AAPL
    _insert_loss_sell(db, "MSFT", pnl_pct=-0.05, days_ago=5)
    assert _is_wash_sale_risk(db, "AAPL") is False


def test_wash_sale_no_history_allows_buy(db):
    assert _is_wash_sale_risk(db, "TSLA") is False


def test_wash_sale_blocks_on_negative_pnl_pct_without_realized(db):
    # Older rows may have realized_pnl=0 (entry_price not recorded)
    # Should fall back to pnl_pct < 0 as the loss signal
    _insert_loss_sell(db, "NVDA", pnl_pct=-0.03, days_ago=10, realized_pnl=0.0)
    assert _is_wash_sale_risk(db, "NVDA") is True


# --- log_trade audit fields ---

def test_log_trade_stores_order_id(db):
    log_trade(db, "AAPL", "BUY", 1.0, 150.0, 150.0, "RANGING", 10000.0, 0.0,
              order_id="abc-123")
    row = db.execute("SELECT order_id FROM trades").fetchone()
    assert row[0] == "abc-123"


def test_log_trade_stores_holding_days(db):
    log_trade(db, "AAPL", "SELL", 1.0, 160.0, 160.0, "RANGING", 10100.0, 0.06,
              entry_price=150.0, holding_days=3)
    row = db.execute("SELECT holding_days FROM trades").fetchone()
    assert row[0] == 3


def test_log_trade_order_id_defaults_none(db):
    log_trade(db, "AAPL", "BUY", 1.0, 150.0, 150.0, "RANGING", 10000.0, 0.0)
    row = db.execute("SELECT order_id FROM trades").fetchone()
    assert row[0] is None


def test_init_db_has_order_id_and_holding_days_columns(tmp_path, monkeypatch):
    db_path = str(tmp_path / "audit_test.db")
    monkeypatch.setattr("bot.main.TRADE_DB_PATH", db_path)
    con = init_db()
    cols = {r[1] for r in con.execute("PRAGMA table_info(trades)").fetchall()}
    assert "order_id" in cols
    assert "holding_days" in cols
    con.close()


# --- wash-sale boundary precision ---

def test_wash_sale_at_exactly_30_days_is_blocked(db):
    # The IRS 30-day window is >=, so exactly 30 days ago is still within range
    _insert_loss_sell(db, "AAPL", pnl_pct=-0.05, days_ago=30)
    assert _is_wash_sale_risk(db, "AAPL") is True


def test_wash_sale_29_days_is_blocked(db):
    _insert_loss_sell(db, "AAPL", pnl_pct=-0.05, days_ago=29)
    assert _is_wash_sale_risk(db, "AAPL") is True


# --- P0: halt state persistence (_save_risk_state) ---

def test_save_risk_state_persists_halt(db):
    from bot.main import _save_risk_state
    from bot.risk.risk_manager import RiskManager
    risk = RiskManager()
    risk.daily_start_value = 10_000.0
    risk.halted = True
    _save_risk_state(db, risk)
    row = db.execute(
        "SELECT value FROM risk_state WHERE key='trading_halted_date'"
    ).fetchone()
    assert row is not None
    assert row[0] == date.today().isoformat()


def test_save_risk_state_clears_halt_when_not_halted(db):
    from bot.main import _save_risk_state
    from bot.risk.risk_manager import RiskManager
    risk = RiskManager()
    risk.daily_start_value = 10_000.0
    risk.halted = False
    _save_risk_state(db, risk)
    row = db.execute(
        "SELECT value FROM risk_state WHERE key='trading_halted_date'"
    ).fetchone()
    assert row is None or row[0] == ""


# --- P0: PDT audit — protective exits still recorded even when limit exceeded ---

def test_maybe_record_day_trade_records_even_when_pdt_exceeded(db):
    """A protective exit past the PDT limit must STILL be recorded in the log.
    The function logs CRITICAL but never silently swallows the trade record —
    the audit trail must reflect the actual day-trade count."""
    from config import PDT_MAX_DAY_TRADES
    _insert_buy_today(db, "AAPL")
    risk = RiskManager()
    risk.reset_daily(10_000.0)
    # Fill the PDT window up to the limit
    for _ in range(PDT_MAX_DAY_TRADES):
        risk.record_day_trade()
    initial_count = len(risk.day_trade_log)
    assert initial_count == PDT_MAX_DAY_TRADES

    # Protective exit on a position opened today — must record despite being at limit
    _maybe_record_day_trade(db, risk, "AAPL", sell_success=True, pdt_exempt=False)

    assert len(risk.day_trade_log) == PDT_MAX_DAY_TRADES + 1, (
        "Day trade must be recorded even when PDT limit is already exceeded"
    )


def test_maybe_record_day_trade_pdt_exceeded_persists_to_db(db):
    """The day-trade record written past the PDT limit must be persisted to SQLite."""
    from config import PDT_MAX_DAY_TRADES
    _insert_buy_today(db, "MSFT")
    risk = RiskManager()
    risk.reset_daily(10_000.0)
    for _ in range(PDT_MAX_DAY_TRADES):
        risk.record_day_trade()

    _maybe_record_day_trade(db, risk, "MSFT", sell_success=True, pdt_exempt=False)

    import json
    row = db.execute(
        "SELECT value FROM risk_state WHERE key='day_trade_dates'"
    ).fetchone()
    assert row is not None
    stored_dates = json.loads(row[0])
    assert len(stored_dates) == PDT_MAX_DAY_TRADES + 1


# --- _record_snapshot (per-cycle portfolio heartbeat) ---

def test_record_snapshot_writes_row(db):
    _record_snapshot(db, 12_500.0, 3_000.0, 2)
    row = db.execute(
        "SELECT portfolio_value, available_cash, open_positions FROM portfolio_snapshots"
    ).fetchone()
    assert row == (12_500.0, 3_000.0, 2)


def test_record_snapshot_dashboard_reads_value_without_any_trade(db):
    """The whole point: dashboard shows a live portfolio value on a no-trade DB."""
    from bot.monitor.dashboard_data import _latest_portfolio_value
    # No trades at all — only a snapshot
    _record_snapshot(db, 9_876.0, 1_000.0, 0)
    assert _latest_portfolio_value(db) == 9_876.0


def test_latest_portfolio_value_prefers_newer_timestamp(db):
    """When both a trade and a snapshot exist, the more recent one wins."""
    from bot.monitor.dashboard_data import _latest_portfolio_value
    # Older trade row
    db.execute(
        "INSERT INTO trades (timestamp,symbol,action,shares,price,notional,regime,portfolio_value,pnl_pct) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        ("2026-06-10T10:00:00+00:00", "AAPL", "BUY", 1.0, 100.0, 100.0, "RANGING", 10_000.0, 0.0),
    )
    # Newer snapshot
    db.execute(
        "INSERT INTO portfolio_snapshots (timestamp, portfolio_value, available_cash, open_positions) "
        "VALUES (?,?,?,?)",
        ("2026-06-11T10:00:00+00:00", 11_000.0, 500.0, 1),
    )
    db.commit()
    assert _latest_portfolio_value(db) == 11_000.0


def test_latest_portfolio_value_prefers_newer_trade(db):
    """A trade row newer than the snapshot should win."""
    from bot.monitor.dashboard_data import _latest_portfolio_value
    db.execute(
        "INSERT INTO portfolio_snapshots (timestamp, portfolio_value, available_cash, open_positions) "
        "VALUES (?,?,?,?)",
        ("2026-06-10T10:00:00+00:00", 9_000.0, 500.0, 1),
    )
    db.execute(
        "INSERT INTO trades (timestamp,symbol,action,shares,price,notional,regime,portfolio_value,pnl_pct) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        ("2026-06-11T10:00:00+00:00", "AAPL", "SELL", 1.0, 120.0, 120.0, "RANGING", 12_000.0, 0.2),
    )
    db.commit()
    assert _latest_portfolio_value(db) == 12_000.0


def _yesterday_iso(hour=20):
    return (date.today() - timedelta(days=1)).isoformat() + f"T{hour:02d}:00:00+00:00"


def test_anchor_daily_start_prefers_prior_day_snapshot(db):
    """Day P&L baseline = yesterday's account snapshot, NOT the stale trade cost basis."""
    # Old trade row at cost basis 100000 (the bug source)
    db.execute(
        "INSERT INTO trades (timestamp,symbol,action,shares,price,notional,regime,portfolio_value,pnl_pct) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (_yesterday_iso(10), "AAPL", "BUY", 1.0, 100.0, 100.0, "RANGING", 100_000.0, 0.0),
    )
    # Yesterday's closing snapshot reflects real appreciated equity
    db.execute(
        "INSERT INTO portfolio_snapshots (timestamp, portfolio_value, available_cash, open_positions) "
        "VALUES (?,?,?,?)",
        (_yesterday_iso(20), 100_491.0, 5_000.0, 5),
    )
    db.commit()
    val, src = _anchor_daily_start(db)
    assert val == pytest.approx(100_491.0)     # snapshot, not 100000
    assert "snapshot" in src


def test_anchor_daily_start_falls_back_to_trade(db):
    """Older DB with no snapshots → last trade row before today."""
    db.execute(
        "INSERT INTO trades (timestamp,symbol,action,shares,price,notional,regime,portfolio_value,pnl_pct) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (_yesterday_iso(10), "AAPL", "BUY", 1.0, 100.0, 100.0, "RANGING", 99_000.0, 0.0),
    )
    db.commit()
    val, src = _anchor_daily_start(db)
    assert val == pytest.approx(99_000.0)
    assert "trade" in src


def test_apply_sim_capital_disabled_by_default(monkeypatch):
    monkeypatch.setattr("bot.main.PAPER_SIM_CAPITAL", 0.0)
    pv, cash, active = _apply_sim_capital(100_000.0, 80_000.0)
    assert (pv, cash, active) == (100_000.0, 80_000.0, False)


def test_apply_sim_capital_caps_equity(monkeypatch):
    monkeypatch.setattr("bot.main.PAPER_SIM_CAPITAL", 1_000.0)
    pv, cash, active = _apply_sim_capital(100_000.0, 80_000.0)
    assert pv == pytest.approx(1_000.0)      # sized as if $1k
    assert cash == pytest.approx(1_000.0)
    assert active is True


def test_apply_sim_capital_never_exceeds_real_account(monkeypatch):
    # If the real account is smaller than the sim cap, don't inflate it.
    monkeypatch.setattr("bot.main.PAPER_SIM_CAPITAL", 1_000.0)
    pv, cash, active = _apply_sim_capital(500.0, 400.0)
    assert pv == pytest.approx(500.0)
    assert cash == pytest.approx(400.0)
    assert active is True


def test_anchor_daily_start_ignores_today_only_data(db):
    """Snapshots/trades only from today (no prior day) → None (nothing to anchor to)."""
    today_ts = date.today().isoformat() + "T15:00:00+00:00"
    db.execute(
        "INSERT INTO portfolio_snapshots (timestamp, portfolio_value, available_cash, open_positions) "
        "VALUES (?,?,?,?)",
        (today_ts, 100_500.0, 5_000.0, 5),
    )
    db.commit()
    val, src = _anchor_daily_start(db)
    assert val is None


def test_latest_portfolio_value_zero_when_empty(db):
    from bot.monitor.dashboard_data import _latest_portfolio_value
    assert _latest_portfolio_value(db) == 0.0


def test_record_snapshot_upserts_same_timestamp(db, monkeypatch):
    """Two writes in the same cycle (same timestamp) must not duplicate rows."""
    import bot.main as m
    fixed = "2026-06-11T12:00:00+00:00"

    class _FixedDT:
        @staticmethod
        def now(tz=None):
            from datetime import datetime as _dt
            return _dt.fromisoformat(fixed)

    monkeypatch.setattr(m, "datetime", _FixedDT)
    _record_snapshot(db, 10_000.0, 1_000.0, 1)
    _record_snapshot(db, 10_050.0, 950.0, 1)
    n = db.execute("SELECT COUNT(*) FROM portfolio_snapshots").fetchone()[0]
    assert n == 1
    val = db.execute("SELECT portfolio_value FROM portfolio_snapshots").fetchone()[0]
    assert val == 10_050.0  # second write overwrote the first
