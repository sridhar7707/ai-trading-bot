from datetime import date, timedelta
import pytest
from bot.risk.risk_manager import RiskManager


@pytest.fixture
def risk():
    rm = RiskManager()
    rm.reset_daily(10_000.0)
    return rm


# --- approve_buy ---

def test_approve_buy_passes_normal(risk):
    assert risk.approve_buy("AAPL", 500, 10_000, 10_000, 2) is True


def test_approve_buy_blocked_when_halted(risk):
    risk.halted = True
    assert risk.approve_buy("AAPL", 500, 10_000, 10_000, 2) is False


def test_approve_buy_blocked_over_position_size(risk):
    # MAX_POSITION_PCT = 0.20; max notional = 0.20 * 10_000 = 2_000
    assert risk.approve_buy("AAPL", 2_500, 10_000, 10_000, 2) is False


def test_approve_buy_blocked_max_positions(risk):
    assert risk.approve_buy("AAPL", 500, 10_000, 10_000, 5) is False


def test_approve_buy_blocked_daily_loss(risk):
    # Current value is 6% below start (limit is 5%)
    current = 10_000 * (1 - 0.06)
    assert risk.approve_buy("AAPL", 500, 10_000, current, 2) is False


# --- check_daily_loss ---

def test_daily_loss_no_start_value_passes():
    rm = RiskManager()
    assert rm.check_daily_loss(9_000) is True


def test_daily_loss_zero_start_value_passes():
    rm = RiskManager()
    rm.daily_start_value = 0.0
    assert rm.check_daily_loss(0.0) is True


def test_daily_loss_within_limit(risk):
    current = 10_000 * (1 - 0.04)  # 4% loss, limit is 5%
    assert risk.check_daily_loss(current) is True


def test_daily_loss_at_limit_halts(risk):
    current = 10_000 * (1 - 0.05)  # exactly 5% → triggers (<=)
    assert risk.check_daily_loss(current) is False
    assert risk.halted is True


# --- check_stop_loss ---

def test_stop_loss_triggers(risk):
    assert risk.check_stop_loss("AAPL", -0.05) is True  # STOP_LOSS_PCT = 0.04


def test_stop_loss_within_tolerance(risk):
    assert risk.check_stop_loss("AAPL", -0.03) is False


def test_stop_loss_positive_pnl(risk):
    assert risk.check_stop_loss("AAPL", 0.10) is False


# --- PDT ---

def test_pdt_allows_when_no_trades(risk):
    assert risk.check_pdt(is_day_trade=True) is True


def test_pdt_blocks_at_limit(risk):
    for _ in range(3):  # PDT_MAX_DAY_TRADES = 3
        risk.record_day_trade()
    assert risk.check_pdt(is_day_trade=True) is False


def test_pdt_allows_non_day_trade_at_limit(risk):
    for _ in range(3):
        risk.record_day_trade()
    assert risk.check_pdt(is_day_trade=False) is True


def test_pdt_purges_old_trades(risk):
    # Inject a trade older than PDT_WINDOW_DAYS (5 days)
    old_date = date.today() - timedelta(days=6)
    risk.day_trade_log.append(old_date)
    risk.reset_daily(10_000)  # purge happens here
    assert risk.check_pdt(is_day_trade=True) is True


def test_record_day_trade_increments(risk):
    risk.record_day_trade()
    risk.record_day_trade()
    assert len(risk.day_trade_log) == 2
