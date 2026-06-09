from datetime import date
from collections import deque
from loguru import logger
from config import (
    MAX_POSITION_PCT, STOP_LOSS_PCT, DAILY_LOSS_LIMIT_PCT,
    PDT_MAX_DAY_TRADES, PDT_WINDOW_DAYS,
)


class RiskManager:
    """Hard-coded risk rules — the RL agent cannot bypass these."""

    def __init__(self):
        self.day_trade_log: deque[date] = deque()
        self.daily_start_value: float | None = None
        self.halted = False

    def reset_daily(self, portfolio_value: float):
        self.daily_start_value = portfolio_value
        self.halted = False
        today = date.today()
        # Purge day trades older than PDT_WINDOW_DAYS
        self.day_trade_log = deque(
            d for d in self.day_trade_log
            if (today - d).days < PDT_WINDOW_DAYS
        )
        logger.info(f"Daily risk reset — portfolio=${portfolio_value:.2f}, day_trades_used={len(self.day_trade_log)}/{PDT_MAX_DAY_TRADES}")

    def check_daily_loss(self, current_value: float) -> bool:
        if self.daily_start_value is None or self.daily_start_value == 0.0:
            return True
        pnl_pct = (current_value - self.daily_start_value) / self.daily_start_value
        if pnl_pct <= -DAILY_LOSS_LIMIT_PCT:
            logger.warning(f"Daily loss limit hit ({pnl_pct:.1%}) — halting trading.")
            self.halted = True
            return False
        return True

    def check_stop_loss(self, symbol: str, pnl_pct: float) -> bool:
        if pnl_pct <= -STOP_LOSS_PCT:
            logger.warning(f"Stop-loss triggered for {symbol} ({pnl_pct:.1%})")
            return True
        return False

    def check_pdt(self, is_day_trade: bool = False) -> bool:
        today = date.today()
        recent = sum(1 for d in self.day_trade_log if (today - d).days < PDT_WINDOW_DAYS)
        if is_day_trade and recent >= PDT_MAX_DAY_TRADES:
            logger.warning(f"PDT limit reached ({recent}/{PDT_MAX_DAY_TRADES}) — blocking day trade.")
            return False
        return True

    def record_day_trade(self):
        self.day_trade_log.append(date.today())

    def approve_buy(
        self,
        symbol: str,
        notional: float,
        portfolio_value: float,
        current_value: float,
        open_positions: int,
    ) -> bool:
        if self.halted:
            logger.warning("Trading halted — buy blocked.")
            return False
        if not self.check_daily_loss(current_value):
            return False
        max_notional = portfolio_value * MAX_POSITION_PCT
        if notional > max_notional:
            logger.warning(f"Position size ${notional:.2f} exceeds max ${max_notional:.2f} — capping.")
            return False
        if open_positions >= 5:
            logger.warning("Max 5 open positions reached — buy blocked.")
            return False
        return True

    def approve_sell(self, symbol: str, pnl_pct: float, current_value: float) -> bool:
        if not self.check_daily_loss(current_value):
            return True  # Force sell when daily limit hit
        return True  # Sells are generally always allowed
