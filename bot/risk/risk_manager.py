from __future__ import annotations
from datetime import date, timedelta
from collections import deque
from loguru import logger
from config import (
    MAX_POSITION_PCT, STOP_LOSS_PCT,
    DAILY_LOSS_LIMIT_PCT, DAILY_LOSS_WARNING_PCT, WEEKLY_LOSS_LIMIT_PCT,
    PORTFOLIO_DRAWDOWN_LIMIT_PCT,
    PDT_MAX_DAY_TRADES, PDT_WINDOW_DAYS,
    ATR_STOP_MULTIPLIER, ATR_TRAIL_MULTIPLIER,
    ATR_MIN_STOP_PCT, ATR_MAX_STOP_PCT,
    MAX_POSITIONS, MAX_SECTOR_POSITIONS, SECTOR_MAP,
)


def _business_days_between(d1: date, d2: date) -> int:
    """Count business days in [d1, d2) for FINRA's rolling 5-business-day PDT window.
    Calendar-day counting undercounts across weekends — a Monday trade expires from
    a 5-calendar-day window on Saturday but FINRA keeps it through the following Monday.
    """
    n, cur = 0, d1
    while cur < d2:
        if cur.weekday() < 5:
            n += 1
        cur += timedelta(days=1)
    return n


class RiskManager:
    """Hard-coded risk rules — the RL agent cannot bypass these."""

    def __init__(self,
                 daily_start_value: float | None = None,
                 day_trade_dates: list[str] | None = None,
                 weekly_start_value: float | None = None,
                 daily_warning_sent: bool = False,
                 weekly_halt_alerted: bool = False,
                 portfolio_high: float | None = None):
        self.day_trade_log: deque[date] = deque()
        if day_trade_dates:
            for ds in day_trade_dates:
                try:
                    self.day_trade_log.append(date.fromisoformat(ds))
                except ValueError:
                    pass
        self.daily_start_value   = daily_start_value
        self.weekly_start_value  = weekly_start_value
        self.daily_warning_sent  = daily_warning_sent
        self.weekly_halt_alerted = weekly_halt_alerted
        self.portfolio_high      = portfolio_high
        self.halted = False

    def reset_daily(self, portfolio_value: float):
        # Only record start-of-day value on the FIRST cycle of the day.
        # Subsequent cycles must NOT overwrite it — otherwise the daily loss
        # gate compares against "5 minutes ago" instead of true start-of-day.
        if self.daily_start_value is None:
            self.daily_start_value = portfolio_value
        # Weekly start is initialized to the first portfolio value seen this ISO week.
        # It persists across daily resets — cleared only when a new week begins.
        if self.weekly_start_value is None:
            self.weekly_start_value = portfolio_value
        # halted resets to False each cycle; protection relies on check_daily_loss()
        # recalculating from daily_start_value (persisted in DB) every approve_buy() call.
        self.halted = False
        today = date.today()
        self.day_trade_log = deque(
            d for d in self.day_trade_log
            if _business_days_between(d, today) < PDT_WINDOW_DAYS
        )
        logger.info(
            f"Risk state — daily_start=${self.daily_start_value:.2f}, "
            f"weekly_start=${self.weekly_start_value:.2f}, "
            f"current=${portfolio_value:.2f}, "
            f"day_trades_used={len(self.day_trade_log)}/{PDT_MAX_DAY_TRADES}"
        )

    # ── Daily loss gate ────────────────────────────────────────────────────────
    def check_daily_loss(self, current_value: float) -> bool:
        if self.daily_start_value is None or self.daily_start_value == 0.0:
            return True
        pnl_pct = (current_value - self.daily_start_value) / self.daily_start_value
        if pnl_pct <= -DAILY_LOSS_LIMIT_PCT:
            logger.warning(f"Daily loss limit hit ({pnl_pct:.1%}) — halting trading.")
            self.halted = True
            return False
        return True

    def check_daily_loss_warning(self, current_value: float) -> bool:
        """Returns True if portfolio is in the warning zone (50–100% of daily limit, unsent)."""
        if self.daily_warning_sent or self.daily_start_value is None or self.daily_start_value == 0.0:
            return False
        pnl_pct = (current_value - self.daily_start_value) / self.daily_start_value
        return -DAILY_LOSS_LIMIT_PCT < pnl_pct <= -DAILY_LOSS_WARNING_PCT

    # ── Weekly loss gate ───────────────────────────────────────────────────────
    def check_weekly_loss(self, current_value: float) -> bool:
        """Returns True (buy allowed) unless weekly drawdown circuit breaker is hit."""
        if self.weekly_start_value is None or self.weekly_start_value == 0.0:
            return True
        weekly_pnl = (current_value - self.weekly_start_value) / self.weekly_start_value
        if weekly_pnl <= -WEEKLY_LOSS_LIMIT_PCT:
            logger.warning(f"Weekly loss limit hit ({weekly_pnl:.1%}) — blocking new buys.")
            return False
        return True

    # ── All-time-high drawdown gate ───────────────────────────────────────────
    def update_portfolio_high(self, current_value: float):
        if self.portfolio_high is None or current_value > self.portfolio_high:
            self.portfolio_high = current_value

    def check_portfolio_drawdown(self, current_value: float) -> bool:
        """Returns False (block new buys) when portfolio is >= PORTFOLIO_DRAWDOWN_LIMIT_PCT
        below its all-time high — catches multi-week grinding losses the weekly gate misses."""
        if self.portfolio_high is None or self.portfolio_high == 0.0:
            return True
        dd = (self.portfolio_high - current_value) / self.portfolio_high
        if dd >= PORTFOLIO_DRAWDOWN_LIMIT_PCT:
            logger.warning(
                f"Portfolio drawdown limit: {dd:.1%} below all-time peak ${self.portfolio_high:.2f} "
                f"— blocking new buys until recovery."
            )
            return False
        return True

    # ── Stop-loss: ATR-aware with flat fallback ────────────────────────────────
    def check_stop_loss(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        atr: float | None = None,
        pnl_pct: float | None = None,
    ) -> bool:
        """
        Returns True if position should be stopped out.
        Prefers ATR-based stop (entry - 2×ATR) over flat percentage.
        Falls back to STOP_LOSS_PCT when ATR is unavailable.
        """
        if atr and atr > 0 and entry_price > 0:
            stop_pct = (ATR_STOP_MULTIPLIER * atr) / entry_price
            # Clamp to sane range so a tiny ATR doesn't stop a normal move
            stop_pct = max(ATR_MIN_STOP_PCT, min(ATR_MAX_STOP_PCT, stop_pct))
            stop_price = entry_price * (1 - stop_pct)
            if current_price <= stop_price:
                logger.warning(
                    f"ATR stop-loss triggered for {symbol}: "
                    f"price=${current_price:.2f} ≤ stop=${stop_price:.2f} "
                    f"(entry=${entry_price:.2f}, ATR=${atr:.2f}, threshold={stop_pct:.1%})"
                )
                return True
        elif pnl_pct is not None and pnl_pct <= -STOP_LOSS_PCT:
            # Flat-percentage fallback (no ATR available)
            logger.warning(
                f"Flat stop-loss triggered for {symbol}: pnl={pnl_pct:.1%} ≤ -{STOP_LOSS_PCT:.1%}"
            )
            return True
        return False

    # ── Trailing stop: lock in gains ──────────────────────────────────────────
    def check_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        high_water_mark: float,
        atr: float,
    ) -> bool:
        """
        Returns True if price has fallen 1.5×ATR below its high-water-mark.
        Only activates once price has risen meaningfully above entry.
        """
        if atr <= 0 or high_water_mark <= 0:
            return False
        trail_distance = ATR_TRAIL_MULTIPLIER * atr
        trail_price = high_water_mark - trail_distance
        if current_price <= trail_price:
            gain_from_hwm = (high_water_mark - current_price) / high_water_mark
            logger.warning(
                f"Trailing stop triggered for {symbol}: "
                f"price=${current_price:.2f} ≤ trail=${trail_price:.2f} "
                f"(hwm=${high_water_mark:.2f}, ATR=${atr:.2f}, fell {gain_from_hwm:.1%} from peak)"
            )
            return True
        return False

    # ── PDT compliance ────────────────────────────────────────────────────────
    def check_pdt(self, is_day_trade: bool = False) -> bool:
        today = date.today()
        recent = sum(1 for d in self.day_trade_log if _business_days_between(d, today) < PDT_WINDOW_DAYS)
        if is_day_trade and recent >= PDT_MAX_DAY_TRADES:
            logger.warning(f"PDT limit reached ({recent}/{PDT_MAX_DAY_TRADES}) — blocking day trade.")
            return False
        return True

    def record_day_trade(self):
        self.day_trade_log.append(date.today())

    # ── Sector concentration check ────────────────────────────────────────────
    def sector_check(self, symbol: str, open_positions: dict) -> bool:
        """
        Returns True (buy allowed) if the sector limit is not yet reached.
        ETFs in 'Broad_ETF' are always allowed (they're already diversified).
        """
        sector = SECTOR_MAP.get(symbol, "Unknown")
        if sector == "Broad_ETF":
            return True
        held_in_sector = sum(
            1 for sym in open_positions
            if SECTOR_MAP.get(sym, "Unknown") == sector
        )
        if held_in_sector >= MAX_SECTOR_POSITIONS:
            logger.warning(
                f"Sector limit: {symbol} ({sector}) blocked — "
                f"already holding {held_in_sector}/{MAX_SECTOR_POSITIONS} in that sector "
                f"({[s for s in open_positions if SECTOR_MAP.get(s) == sector]})"
            )
            return False
        return True

    # ── Buy approval gate ─────────────────────────────────────────────────────
    def approve_buy(
        self,
        symbol: str,
        notional: float,
        portfolio_value: float,
        current_value: float,
        open_positions: dict,        # full dict {symbol: position_obj} for sector check
    ) -> bool:
        if self.halted:
            logger.warning("Trading halted — buy blocked.")
            return False
        if not self.check_daily_loss(current_value):
            return False
        if not self.check_weekly_loss(current_value):
            return False
        if not self.check_portfolio_drawdown(current_value):
            return False
        max_notional = portfolio_value * MAX_POSITION_PCT
        if notional > max_notional:
            logger.warning(f"Position size ${notional:.2f} exceeds max ${max_notional:.2f}.")
            return False
        if len(open_positions) >= MAX_POSITIONS:
            logger.warning(f"Max {MAX_POSITIONS} open positions reached — buy blocked.")
            return False
        if not self.sector_check(symbol, open_positions):
            return False
        return True

    def approve_sell(self, symbol: str, pnl_pct: float, current_value: float) -> bool:
        # Sells are never blocked — exits must always be possible regardless of loss limits.
        return True
