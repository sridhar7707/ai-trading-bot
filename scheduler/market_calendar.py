"""Single source of truth for NYSE market hours and holidays (ET timezone)."""
from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# NYSE 2026 holidays
_HOLIDAYS_2026: frozenset[datetime.date] = frozenset({
    datetime.date(2026, 1, 1),   # New Year's Day
    datetime.date(2026, 1, 19),  # MLK Day
    datetime.date(2026, 2, 16),  # Presidents Day
    datetime.date(2026, 4, 3),   # Good Friday
    datetime.date(2026, 5, 25),  # Memorial Day
    datetime.date(2026, 6, 19),  # Juneteenth
    datetime.date(2026, 7, 4),   # Independence Day
    datetime.date(2026, 9, 7),   # Labor Day
    datetime.date(2026, 11, 26), # Thanksgiving
    datetime.date(2026, 12, 25), # Christmas
})

# Early close days (13:00 ET): Jul 3, Nov 27
_EARLY_CLOSE_2026: frozenset[datetime.date] = frozenset({
    datetime.date(2026, 7, 3),
    datetime.date(2026, 11, 27),
})

_OPEN_TIME  = datetime.time(9, 30)
_CLOSE_TIME = datetime.time(16, 0)
_EARLY_CLOSE_TIME = datetime.time(13, 0)
_PREMARKET_START  = datetime.time(8, 0)


def is_trading_day(d: datetime.date) -> bool:
    """Return False for weekends and NYSE holidays."""
    if d.weekday() >= 5:
        return False
    return d not in _HOLIDAYS_2026


def is_early_close_day(d: datetime.date) -> bool:
    return d in _EARLY_CLOSE_2026


def get_close_time(d: datetime.date) -> datetime.time:
    return _EARLY_CLOSE_TIME if is_early_close_day(d) else _CLOSE_TIME


def get_market_state(dt_et: datetime.datetime) -> str:
    """Return PREMARKET / OPEN / CLOSED for a given ET datetime."""
    d = dt_et.date()
    if not is_trading_day(d):
        return "CLOSED"
    t = dt_et.time()
    close = get_close_time(d)
    if _OPEN_TIME <= t < close:
        return "OPEN"
    if _PREMARKET_START <= t < _OPEN_TIME:
        return "PREMARKET"
    return "CLOSED"


def now_et() -> datetime.datetime:
    return datetime.datetime.now(ET)


def minutes_until_open(dt_et: datetime.datetime) -> int:
    """Minutes until 09:30 ET; 0 if already open or past open."""
    open_dt = dt_et.replace(hour=9, minute=30, second=0, microsecond=0)
    if dt_et >= open_dt:
        return 0
    return max(0, int((open_dt - dt_et).total_seconds() // 60))


def minutes_until_close(dt_et: datetime.datetime) -> int:
    """Minutes until market close ET; 0 if already closed."""
    close = get_close_time(dt_et.date())
    close_dt = dt_et.replace(
        hour=close.hour, minute=close.minute, second=0, microsecond=0
    )
    if dt_et >= close_dt:
        return 0
    return max(0, int((close_dt - dt_et).total_seconds() // 60))
