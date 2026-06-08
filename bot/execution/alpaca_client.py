import alpaca_trade_api as tradeapi
import pandas as pd
from loguru import logger
from config import ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL, MAX_POSITION_PCT


class AlpacaClient:
    def __init__(self):
        self.api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL, api_version="v2")
        logger.info(f"Alpaca connected — mode: {'paper' if 'paper' in ALPACA_BASE_URL else 'live'}")

    def get_account(self):
        return self.api.get_account()

    def get_portfolio_value(self) -> float:
        return float(self.get_account().portfolio_value)

    def get_cash(self) -> float:
        return float(self.get_account().cash)

    def get_positions(self) -> dict:
        positions = self.api.list_positions()
        return {p.symbol: p for p in positions}

    def get_latest_price(self, symbol: str) -> float:
        bar = self.api.get_latest_bar(symbol)
        if bar is None:
            raise ValueError(f"No bar data returned for {symbol}")
        return bar.c

    def get_bars(self, symbol: str, timeframe: str = "5Min", limit: int = 100) -> pd.DataFrame:
        bars = self.api.get_bars(symbol, timeframe, limit=limit).df
        bars.index = pd.to_datetime(bars.index)
        return bars

    def buy(self, symbol: str, notional: float) -> dict | None:
        if notional < 1.0:
            logger.warning(f"BUY skipped {symbol} — notional ${notional:.2f} below $1 minimum")
            return None
        try:
            order = self.api.submit_order(
                symbol=symbol,
                notional=round(notional, 2),
                side="buy",
                type="market",
                time_in_force="day",
            )
            logger.info(f"BUY {symbol} notional=${notional:.2f} order_id={order.id}")
            return {"order_id": order.id, "symbol": symbol, "side": "buy", "notional": notional}
        except Exception as e:
            logger.error(f"BUY failed {symbol}: {e}")
            return None

    def sell(self, symbol: str) -> dict | None:
        try:
            positions = self.get_positions()
            if symbol not in positions:
                logger.warning(f"SELL skipped — no position in {symbol}")
                return None
            order = self.api.submit_order(
                symbol=symbol,
                qty=positions[symbol].qty,
                side="sell",
                type="market",
                time_in_force="day",
            )
            logger.info(f"SELL {symbol} qty={positions[symbol].qty} order_id={order.id}")
            return {"order_id": order.id, "symbol": symbol, "side": "sell"}
        except Exception as e:
            logger.error(f"SELL failed {symbol}: {e}")
            return None

    def get_position_value(self, symbol: str) -> float:
        positions = self.get_positions()
        if symbol not in positions:
            return 0.0
        return float(positions[symbol].market_value)

    def get_position_pnl_pct(self, symbol: str) -> float:
        positions = self.get_positions()
        if symbol not in positions:
            return 0.0
        p = positions[symbol]
        return float(p.unrealized_plpc)
