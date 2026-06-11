from datetime import date, timedelta
import pytest
from bot.risk.risk_manager import RiskManager, _business_days_between
from config import DAILY_LOSS_LIMIT_PCT


@pytest.fixture
def risk():
    rm = RiskManager()
    rm.reset_daily(10_000.0)
    return rm


# --- approve_buy ---

def test_approve_buy_passes_normal(risk):
    assert risk.approve_buy("AAPL", 500, 10_000, 10_000, {}) is True


def test_approve_buy_blocked_when_halted(risk):
    risk.halted = True
    assert risk.approve_buy("AAPL", 500, 10_000, 10_000, {}) is False


def test_approve_buy_blocked_over_position_size(risk):
    # MAX_POSITION_PCT = 0.20; max notional = 0.20 * 10_000 = 2_000
    assert risk.approve_buy("AAPL", 2_500, 10_000, 10_000, {}) is False


def test_approve_buy_blocked_max_positions(risk):
    # MAX_POSITIONS = 5; dict must have 5 entries to hit the limit
    full = {"A": None, "B": None, "C": None, "D": None, "E": None}
    assert risk.approve_buy("AAPL", 500, 10_000, 10_000, full) is False


def test_approve_buy_blocked_daily_loss(risk):
    # Current value is 6% below start (limit is 5%)
    current = 10_000 * (1 - 0.06)
    assert risk.approve_buy("AAPL", 500, 10_000, current, {}) is False


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
    current = 10_000 * (1 - DAILY_LOSS_LIMIT_PCT)
    assert risk.check_daily_loss(current) is False
    assert risk.halted is True


# --- check_stop_loss ---
# New signature: check_stop_loss(symbol, current_price, entry_price, atr=None, pnl_pct=None)
# STOP_LOSS_PCT = 0.04; pnl_pct path used when atr=None

def test_stop_loss_triggers(risk):
    assert risk.check_stop_loss("AAPL", 95.0, 100.0, pnl_pct=-0.05) is True


def test_stop_loss_within_tolerance(risk):
    assert risk.check_stop_loss("AAPL", 97.0, 100.0, pnl_pct=-0.03) is False


def test_stop_loss_positive_pnl(risk):
    assert risk.check_stop_loss("AAPL", 110.0, 100.0, pnl_pct=0.10) is False


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


# --- check_weekly_loss ---

def test_weekly_loss_no_start_value_passes():
    rm = RiskManager()
    assert rm.check_weekly_loss(9_000) is True


def test_weekly_loss_within_limit(risk):
    risk.weekly_start_value = 10_000.0
    current = 10_000 * (1 - 0.09)  # 9% loss, limit is 10%
    assert risk.check_weekly_loss(current) is True


def test_weekly_loss_at_limit_blocks(risk):
    risk.weekly_start_value = 10_000.0
    current = 10_000 * (1 - 0.10)  # exactly at 10% limit
    assert risk.check_weekly_loss(current) is False


def test_approve_buy_blocked_weekly_loss(risk):
    risk.weekly_start_value = 10_000.0
    current = 10_000 * (1 - 0.11)  # 11% weekly loss
    assert risk.approve_buy("AAPL", 500, 10_000, current, {}) is False


# --- check_daily_loss_warning ---

def test_daily_loss_warning_in_zone(risk):
    # DAILY_LOSS_LIMIT_PCT=0.05, warning at 50% = 2.5%; put at 3% (in warning zone)
    current = 10_000 * (1 - 0.03)
    assert risk.check_daily_loss_warning(current) is True


def test_daily_loss_warning_not_sent_twice(risk):
    current = 10_000 * (1 - 0.03)
    risk.daily_warning_sent = True
    assert risk.check_daily_loss_warning(current) is False


def test_daily_loss_warning_outside_zone(risk):
    # 1% loss — below the 2.5% warning threshold
    current = 10_000 * (1 - 0.01)
    assert risk.check_daily_loss_warning(current) is False


# --- approve_sell ---

def test_approve_sell_always_returns_true(risk):
    # Sells must never be blocked — exits must always be possible.
    assert risk.approve_sell("AAPL", -0.10, 9_000) is True
    assert risk.approve_sell("AAPL",  0.05, 10_500) is True


# --- update_portfolio_high ---

def test_update_portfolio_high_sets_initial_value():
    rm = RiskManager()
    rm.update_portfolio_high(10_000.0)
    assert rm.portfolio_high == 10_000.0


def test_update_portfolio_high_tracks_peak():
    rm = RiskManager()
    rm.update_portfolio_high(10_000.0)
    rm.update_portfolio_high(12_000.0)
    rm.update_portfolio_high(11_000.0)  # retreat — peak should stay at 12k
    assert rm.portfolio_high == 12_000.0


def test_update_portfolio_high_initialised_from_constructor():
    rm = RiskManager(portfolio_high=15_000.0)
    rm.update_portfolio_high(14_000.0)  # below existing high
    assert rm.portfolio_high == 15_000.0


# --- check_portfolio_drawdown ---

def test_portfolio_drawdown_passes_within_limit():
    rm = RiskManager(portfolio_high=10_000.0)
    # 15% below peak — limit is 20%
    assert rm.check_portfolio_drawdown(8_500.0) is True


def test_portfolio_drawdown_blocks_at_limit():
    rm = RiskManager(portfolio_high=10_000.0)
    # Exactly 20% below peak
    assert rm.check_portfolio_drawdown(8_000.0) is False


def test_portfolio_drawdown_blocks_below_limit():
    rm = RiskManager(portfolio_high=10_000.0)
    # 25% below peak
    assert rm.check_portfolio_drawdown(7_500.0) is False


def test_portfolio_drawdown_passes_when_no_high():
    rm = RiskManager()
    assert rm.check_portfolio_drawdown(5_000.0) is True


def test_portfolio_drawdown_passes_when_high_is_zero():
    rm = RiskManager(portfolio_high=0.0)
    assert rm.check_portfolio_drawdown(5_000.0) is True


def test_approve_buy_blocked_by_portfolio_drawdown(risk):
    risk.portfolio_high = 10_000.0
    # 25% below all-time high — should block
    assert risk.approve_buy("AAPL", 500, 10_000, 7_500.0, {}) is False


def test_approve_buy_passes_with_acceptable_drawdown(risk):
    # Set high above daily start so a ~10% drawdown from peak stays within daily loss limit
    risk.portfolio_high = 11_000.0
    # current=9_900: 10% below peak (within 20% limit), 1% below daily_start (within 5% limit)
    assert risk.approve_buy("AAPL", 500, 10_000, 9_900.0, {}) is True


# --- _business_days_between (FINRA PDT business-day window) ---

def _weekday(offset: int) -> date:
    """Return a weekday date `offset` business days before today."""
    d, n = date.today(), 0
    while n < offset:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return d


def test_business_days_between_same_day():
    today = date.today()
    assert _business_days_between(today, today) == 0


def test_business_days_between_one_business_day():
    yesterday = _weekday(1)
    assert _business_days_between(yesterday, date.today()) == 1


def test_business_days_between_five_business_days():
    five_ago = _weekday(5)
    assert _business_days_between(five_ago, date.today()) == 5


def test_business_days_between_skips_weekends():
    # Friday to Monday = 1 business day (Saturday and Sunday skipped)
    # Find the most recent Monday
    today = date.today()
    days_since_mon = today.weekday()  # 0=Mon
    monday = today - timedelta(days=days_since_mon)
    friday = monday - timedelta(days=3)  # previous Friday
    result = _business_days_between(friday, monday)
    assert result == 1


def test_pdt_window_uses_business_days():
    # A trade 5 business days ago should NOT be counted (window is < 5)
    rm = RiskManager()
    rm.reset_daily(10_000.0)
    five_bdays_ago = _weekday(5)
    rm.day_trade_log.append(five_bdays_ago)
    # 5 business days ago: _business_days_between = 5, which is NOT < 5 → not counted
    today = date.today()
    from bot.risk.risk_manager import _business_days_between as bdb
    count = sum(1 for d in rm.day_trade_log if bdb(d, today) < 5)
    assert count == 0


def test_pdt_window_counts_recent_trades():
    rm = RiskManager()
    rm.reset_daily(10_000.0)
    four_bdays_ago = _weekday(4)
    rm.day_trade_log.append(four_bdays_ago)
    # 4 business days ago: < 5 → should be counted
    assert rm.check_pdt(is_day_trade=True) is True  # under limit (1 < 3)
    assert len([d for d in rm.day_trade_log
                if _business_days_between(d, date.today()) < 5]) == 1
