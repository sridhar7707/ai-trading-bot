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
