from unittest.mock import patch, MagicMock
import bot.monitor.telegram_bot as tg


@patch("bot.monitor.telegram_bot.TELEGRAM_TOKEN", "tok")
@patch("bot.monitor.telegram_bot.TELEGRAM_CHAT_ID", "123")
@patch("bot.monitor.telegram_bot.requests.post")
def test_send_posts_to_telegram(mock_post):
    tg.send("hello")
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["text"] == "hello"


@patch("bot.monitor.telegram_bot.TELEGRAM_TOKEN", "")
@patch("bot.monitor.telegram_bot.requests.post")
def test_send_skips_when_no_token(mock_post):
    tg.send("hello")
    mock_post.assert_not_called()


@patch("bot.monitor.telegram_bot._send")
def test_alert_bot_started_suppressed(mock_send):
    tg.alert_bot_started("paper", 10_000.0)
    mock_send.assert_not_called()


@patch("bot.monitor.telegram_bot._send")
def test_alert_buy_suppressed(mock_send):
    tg.alert_buy("AAPL", 10.0, 150.0, "BULL_TREND", 20_000.0, 0.01)
    mock_send.assert_not_called()


@patch("bot.monitor.telegram_bot._send")
def test_alert_sell_suppressed(mock_send):
    tg.alert_sell("MSFT", 5.0, 300.0, 0.05, reason="signal", notional=1500.0)
    mock_send.assert_not_called()


@patch("bot.monitor.telegram_bot._send")
def test_alert_daily_loss_limit_calls_send(mock_send):
    tg.alert_daily_loss_limit(9_500.0, -0.05)
    mock_send.assert_called_once()
    assert "DAILY LOSS" in mock_send.call_args[0][0]


@patch("bot.monitor.telegram_bot._send")
def test_alert_vix_halt_calls_send(mock_send):
    tg.alert_vix_halt()
    mock_send.assert_called_once()


@patch("bot.monitor.telegram_bot._send")
def test_alert_weekly_report_calls_send(mock_send):
    tg.alert_weekly_report(0.03, 0.01, 0.65, 1.2, 0.04)
    mock_send.assert_called_once()
