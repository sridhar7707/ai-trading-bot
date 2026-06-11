from unittest.mock import MagicMock, patch
import pytest
from bot.execution.alpaca_client import AlpacaClient, MIN_NOTIONAL


@pytest.fixture
def client():
    with patch("bot.execution.alpaca_client.tradeapi.REST") as mock_rest:
        mock_rest.return_value = MagicMock()
        c = AlpacaClient()
        c.api = mock_rest.return_value
        return c


# --- get_latest_price ---

def test_get_latest_price_returns_close(client):
    bar = MagicMock()
    bar.c = 155.50
    client.api.get_latest_bar.return_value = bar
    assert client.get_latest_price("AAPL") == 155.50


def test_get_latest_price_raises_on_none(client):
    client.api.get_latest_bar.return_value = None
    with pytest.raises(ValueError, match="Price data unavailable"):
        client.get_latest_price("AAPL")


def test_get_latest_price_error_does_not_leak_symbol(client):
    client.api.get_latest_bar.return_value = None
    with pytest.raises(ValueError) as exc_info:
        client.get_latest_price("AAPL")
    assert "AAPL" not in str(exc_info.value)


# --- buy ---

def test_buy_skips_below_min_notional(client):
    result = client.buy("AAPL", MIN_NOTIONAL - 0.01)
    assert result is None
    client.api.submit_order.assert_not_called()


def test_buy_skips_at_zero(client):
    result = client.buy("AAPL", 0.0)
    assert result is None


def test_buy_succeeds_above_min_notional(client):
    order = MagicMock()
    order.id = "order-123"
    client.api.submit_order.return_value = order
    result = client.buy("AAPL", 500.0)
    assert result is not None
    assert result["symbol"] == "AAPL"
    assert result["order_id"] == "order-123"


def test_buy_returns_none_on_api_error(client):
    client.api.submit_order.side_effect = Exception("API error")
    result = client.buy("AAPL", 500.0)
    assert result is None


# --- sell ---

def test_sell_skips_when_no_position(client):
    client.api.list_positions.return_value = []
    result = client.sell("AAPL")
    assert result is None
    client.api.submit_order.assert_not_called()


def test_sell_submits_order(client):
    position = MagicMock()
    position.symbol = "AAPL"
    position.qty = "5"
    client.api.list_positions.return_value = [position]
    order = MagicMock()
    order.id = "sell-456"
    client.api.submit_order.return_value = order
    result = client.sell("AAPL")
    assert result["order_id"] == "sell-456"
    # Code converts position.qty (string from Alpaca API) to float before passing
    client.api.submit_order.assert_called_once_with(
        symbol="AAPL",
        qty=5.0,
        side="sell",
        type="market",
        time_in_force="day",
    )


def test_sell_returns_none_on_api_error(client):
    position = MagicMock()
    position.symbol = "AAPL"
    position.qty = "5"
    client.api.list_positions.return_value = [position]
    client.api.submit_order.side_effect = Exception("API error")
    assert client.sell("AAPL") is None


# --- get_position_pnl_pct ---

def test_get_position_pnl_pct_returns_float(client):
    pos = MagicMock()
    pos.symbol = "AAPL"
    pos.unrealized_plpc = "0.0523"
    client.api.list_positions.return_value = [pos]
    assert abs(client.get_position_pnl_pct("AAPL") - 0.0523) < 1e-6


def test_get_position_pnl_pct_no_position_returns_zero(client):
    client.api.list_positions.return_value = []
    assert client.get_position_pnl_pct("AAPL") == 0.0


# --- get_portfolio_value ---

def test_get_portfolio_value_returns_float(client):
    account = MagicMock()
    account.portfolio_value = "50000.00"
    client.api.get_account.return_value = account
    assert client.get_portfolio_value() == 50000.0


# --- get_open_order_symbols ---

def _make_order(symbol, side):
    o = MagicMock()
    o.symbol = symbol
    o.side = side
    return o


def test_get_open_order_symbols_returns_two_sets(client):
    client.api.list_orders.return_value = [
        _make_order("AAPL", "buy"),
        _make_order("MSFT", "sell"),
    ]
    buy_syms, sell_syms = client.get_open_order_symbols()
    assert "AAPL" in buy_syms
    assert "MSFT" in sell_syms
    assert "MSFT" not in buy_syms
    assert "AAPL" not in sell_syms


def test_get_open_order_symbols_empty_on_api_error(client):
    client.api.list_orders.side_effect = Exception("timeout")
    buy_syms, sell_syms = client.get_open_order_symbols()
    assert buy_syms == set()
    assert sell_syms == set()


def test_get_open_order_symbols_multiple_same_side(client):
    client.api.list_orders.return_value = [
        _make_order("AAPL", "buy"),
        _make_order("NVDA", "buy"),
        _make_order("MSFT", "sell"),
    ]
    buy_syms, sell_syms = client.get_open_order_symbols()
    assert buy_syms == {"AAPL", "NVDA"}
    assert sell_syms == {"MSFT"}


# --- get_fill_price ---

def test_get_fill_price_returns_float(client):
    order = MagicMock()
    order.filled_avg_price = "152.73"
    client.api.get_order.return_value = order
    assert client.get_fill_price("order-123") == pytest.approx(152.73)


def test_get_fill_price_returns_none_when_not_filled(client):
    order = MagicMock()
    order.filled_avg_price = None
    client.api.get_order.return_value = order
    assert client.get_fill_price("order-123") is None


def test_get_fill_price_returns_none_on_api_error(client):
    client.api.get_order.side_effect = Exception("not found")
    assert client.get_fill_price("order-123") is None


# --- sell_market ---

def test_sell_market_submits_market_order(client):
    order = MagicMock()
    order.id = "mkt-789"
    client.api.submit_order.return_value = order
    result = client.sell_market("AAPL", 5.0)
    assert result is not None
    assert result["order_id"] == "mkt-789"
    assert result["symbol"] == "AAPL"
    client.api.submit_order.assert_called_once_with(
        symbol="AAPL", qty=5.0, side="sell",
        type="market", time_in_force="day",
    )


def test_sell_market_returns_none_on_api_error(client):
    client.api.submit_order.side_effect = Exception("rejected")
    assert client.sell_market("AAPL", 5.0) is None


# --- wait_for_fill ---

def test_wait_for_fill_returns_true_when_filled(client):
    order = MagicMock()
    order.status = "filled"
    client.api.get_order.return_value = order
    assert client.wait_for_fill("order-123", timeout_secs=5) is True
    client.api.cancel_order.assert_not_called()


def test_wait_for_fill_returns_false_on_cancelled(client):
    order = MagicMock()
    order.status = "cancelled"
    client.api.get_order.return_value = order
    assert client.wait_for_fill("order-123", timeout_secs=5) is False


def test_wait_for_fill_returns_false_on_rejected(client):
    order = MagicMock()
    order.status = "rejected"
    client.api.get_order.return_value = order
    assert client.wait_for_fill("order-123", timeout_secs=5) is False


def test_wait_for_fill_returns_false_on_expired(client):
    order = MagicMock()
    order.status = "expired"
    client.api.get_order.return_value = order
    assert client.wait_for_fill("order-123", timeout_secs=5) is False


def test_wait_for_fill_cancels_and_returns_false_on_timeout(client):
    order = MagicMock()
    order.status = "pending_new"
    client.api.get_order.return_value = order
    result = client.wait_for_fill("order-abc", timeout_secs=0)
    assert result is False
    client.api.cancel_order.assert_called_once_with("order-abc")


def test_wait_for_fill_returns_false_when_poll_raises(client):
    # Poll throws — should not propagate, loop times out
    client.api.get_order.side_effect = Exception("network error")
    result = client.wait_for_fill("order-xyz", timeout_secs=0)
    assert result is False


# --- get_account_summary ---

def test_get_account_summary_returns_portfolio_and_cash(client):
    acct = MagicMock()
    acct.portfolio_value = "25000.00"
    acct.cash = "5000.00"
    client.api.get_account.return_value = acct
    pv, cash = client.get_account_summary()
    assert pv == pytest.approx(25_000.0)
    assert cash == pytest.approx(5_000.0)


def test_get_account_summary_makes_one_api_call(client):
    acct = MagicMock()
    acct.portfolio_value = "10000.00"
    acct.cash = "2000.00"
    client.api.get_account.return_value = acct
    client.get_account_summary()
    assert client.api.get_account.call_count == 1
