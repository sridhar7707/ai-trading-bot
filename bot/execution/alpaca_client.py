from __future__ import annotations
import time
import alpaca_trade_api as tradeapi
import pandas as pd
from loguru import logger
from config import ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL, MAX_POSITION_PCT

MIN_NOTIONAL = 1.0   # Alpaca minimum notional for fractional orders
LIMIT_BUF    = 0.001  # 0.1% aggressive-limit buffer — fills in normal liquid conditions


class AlpacaClient:
    def __init__(self):
        if not ALPACA_KEY or not ALPACA_SECRET:
            logger.error(
                "ALPACA_KEY/ALPACA_SECRET is EMPTY — Alpaca calls will fail and the account "
                "value will read $0.00. Set these as environment/Space secrets."
            )
        self.api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL, api_version="v2")
        # Inject a 30s timeout so a hung Alpaca connection fails fast instead
        # of waiting the OS-default ~2 min, which wastes an entire 5-min cycle.
        try:
            _orig = self.api._session.send
            def _send_timeout(*a, **kw):
                kw.setdefault("timeout", 30)
                return _orig(*a, **kw)
            self.api._session.send = _send_timeout
        except AttributeError:
            pass
        logger.info(
            f"Alpaca connected — mode: {'paper' if 'paper' in ALPACA_BASE_URL else 'live'}, "
            f"url={ALPACA_BASE_URL}, key=...{ALPACA_KEY[-4:] if ALPACA_KEY else 'MISSING'}"
        )

    def get_account(self):
        return self.api.get_account()

    def get_account_summary(self) -> tuple[float, float]:
        """Single API call returning (portfolio_value, available_cash)."""
        acct = self.get_account()
        return float(acct.portfolio_value), float(acct.cash)

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
            logger.debug(f"No bar data returned for {symbol}")
            raise ValueError("Price data unavailable")
        return bar.c

    def get_bars(self, symbol: str, timeframe: str = "5Min", limit: int = 100) -> pd.DataFrame:
        bars = self.api.get_bars(symbol, timeframe, limit=limit).df
        bars.index = pd.to_datetime(bars.index, utc=True)
        return bars

    def buy(self, symbol: str, notional: float, limit_price: float | None = None) -> dict | None:
        """
        Submit a buy order.
        If limit_price is given, uses a limit order at (limit_price × 1.001) — aggressive
        enough to fill on liquid names while avoiding the full bid-ask cost of a market order.
        Falls back to a market order when limit_price is None.
        """
        if notional < MIN_NOTIONAL:
            logger.warning(f"BUY skipped {symbol} — notional ${notional:.2f} below ${MIN_NOTIONAL} minimum")
            return None
        try:
            if limit_price is not None and limit_price > 0:
                effective_limit = round(limit_price * (1 + LIMIT_BUF), 2)
                qty = round(notional / effective_limit, 6)
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="buy",
                    type="limit",
                    time_in_force="day",
                    limit_price=effective_limit,
                )
                logger.info(f"BUY {symbol} qty={qty:.4f} limit=${effective_limit:.2f} order_id={order.id}")
            else:
                order = self.api.submit_order(
                    symbol=symbol,
                    notional=round(notional, 2),
                    side="buy",
                    type="market",
                    time_in_force="day",
                )
                logger.info(f"BUY {symbol} notional=${notional:.2f} (market) order_id={order.id}")
            return {"order_id": order.id, "symbol": symbol, "side": "buy", "notional": notional}
        except Exception as e:
            logger.error(f"BUY failed {symbol}: {e}")
            return None

    def sell(self, symbol: str, qty: float | None = None,
             limit_price: float | None = None) -> dict | None:
        """
        Submit a sell order.
        qty: pass the float quantity to avoid an extra get_positions() API call.
             If omitted, fetches positions internally (legacy path).
        limit_price: if given, uses a limit order at (limit_price × 0.999).
        """
        try:
            if qty is None:
                positions = self.get_positions()
                if symbol not in positions:
                    logger.warning(f"SELL skipped — no position in {symbol}")
                    return None
                qty = float(positions[symbol].qty)
            else:
                qty = float(qty)

            if qty <= 0:
                logger.warning(f"SELL skipped {symbol} — qty={qty}")
                return None

            if limit_price is not None and limit_price > 0:
                effective_limit = round(limit_price * (1 - LIMIT_BUF), 2)
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                    type="limit",
                    time_in_force="day",
                    limit_price=effective_limit,
                )
                logger.info(f"SELL {symbol} qty={qty:.4f} limit=${effective_limit:.2f} order_id={order.id}")
            else:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                    type="market",
                    time_in_force="day",
                )
                logger.info(f"SELL {symbol} qty={qty:.4f} (market) order_id={order.id}")
            return {"order_id": order.id, "symbol": symbol, "side": "sell", "qty": qty}
        except Exception as e:
            logger.error(f"SELL failed {symbol}: {e}")
            return None

    def wait_for_fill(self, order_id: str, timeout_secs: int = 15) -> bool:
        """Poll until filled, cancelled/rejected, or timeout.
        If timeout: cancels the order and returns False.
        Stop-loss callers should fall back to a market order when this returns False.
        """
        deadline = time.monotonic() + timeout_secs
        while time.monotonic() < deadline:
            try:
                order = self.api.get_order(order_id)
                if order.status == "filled":
                    return True
                if order.status in ("cancelled", "expired", "rejected", "done_for_day"):
                    logger.warning(f"Order {order_id} ended as {order.status} — not filled")
                    return False
            except Exception as e:
                logger.warning(f"Order status poll failed ({order_id}): {e}")
            time.sleep(2)
        # Timeout — cancel so it doesn't fill later at a stale price
        try:
            self.api.cancel_order(order_id)
            logger.warning(f"Order {order_id} timed out after {timeout_secs}s — cancelled")
        except Exception as e:
            logger.warning(f"Could not cancel order {order_id}: {e}")
        return False

    def sell_market(self, symbol: str, qty: float) -> dict | None:
        """Market sell — used as stop-loss escalation when a limit sell times out."""
        try:
            order = self.api.submit_order(
                symbol=symbol, qty=float(qty), side="sell",
                type="market", time_in_force="day",
            )
            logger.warning(f"SELL MARKET {symbol} qty={qty:.4f} (stop escalation) order_id={order.id}")
            return {"order_id": order.id, "symbol": symbol, "side": "sell", "qty": qty}
        except Exception as e:
            logger.error(f"Market sell escalation failed {symbol}: {e}")
            return None

    def get_open_order_symbols(self) -> tuple[set[str], set[str]]:
        """Return (buy_symbols, sell_symbols) with pending open orders.
        One API call serves both the buy-duplicate guard and the sell-duplicate guard.
        Separating sides prevents double-selling — submitting a second sell on a symbol
        that already has a pending sell order could create an unintended short position.
        """
        try:
            orders = self.api.list_orders(status="open")
            buy_syms  = {o.symbol for o in orders if o.side == "buy"}
            sell_syms = {o.symbol for o in orders if o.side == "sell"}
            return buy_syms, sell_syms
        except Exception as e:
            logger.warning(f"Could not fetch open orders: {e}")
            return set(), set()

    def get_fill_price(self, order_id: str) -> float | None:
        """Return the actual average fill price of a completed order.
        Records real execution price (not limit estimate) for accurate P&L and slippage tracking.
        Returns None if the order hasn't filled or the field is absent — caller falls back to estimate.
        """
        try:
            order = self.api.get_order(order_id)
            filled_avg = getattr(order, "filled_avg_price", None)
            if filled_avg:
                return float(filled_avg)
        except Exception as e:
            logger.debug(f"Could not get fill price for {order_id}: {e}")
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
        return float(positions[symbol].unrealized_plpc)
