from unittest.mock import MagicMock, patch
import pytest
from bot.strategy.sentiment import get_news_headlines, batch_sentiment_scores


def _mock_resp(status_code: int, payload=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        resp.raise_for_status.return_value = None
    return resp


# --- get_news_headlines ---

def test_get_news_headlines_returns_titles_on_success():
    payload = {"articles": [{"title": "Stock surges"}, {"title": "Market gains"}]}
    with patch("bot.strategy.sentiment.requests.get", return_value=_mock_resp(200, payload)), \
         patch("bot.strategy.sentiment.NEWSAPI_KEY", "test-key"):
        result = get_news_headlines("AAPL")
    assert result == ["Stock surges", "Market gains"]


def test_get_news_headlines_returns_empty_on_426_quota_exhausted():
    with patch("bot.strategy.sentiment.requests.get", return_value=_mock_resp(426)), \
         patch("bot.strategy.sentiment.NEWSAPI_KEY", "test-key"):
        result = get_news_headlines("AAPL")
    assert result == []


def test_get_news_headlines_returns_empty_on_429_rate_limited():
    with patch("bot.strategy.sentiment.requests.get", return_value=_mock_resp(429)), \
         patch("bot.strategy.sentiment.NEWSAPI_KEY", "test-key"):
        result = get_news_headlines("AAPL")
    assert result == []


def test_get_news_headlines_returns_empty_when_no_key():
    with patch("bot.strategy.sentiment.NEWSAPI_KEY", ""):
        result = get_news_headlines("AAPL")
    assert result == []


def test_get_news_headlines_returns_empty_on_network_error():
    with patch("bot.strategy.sentiment.requests.get", side_effect=Exception("timeout")), \
         patch("bot.strategy.sentiment.NEWSAPI_KEY", "test-key"):
        result = get_news_headlines("AAPL")
    assert result == []


def test_get_news_headlines_skips_articles_with_no_title():
    payload = {"articles": [{"title": "Real headline"}, {"title": None}, {}]}
    with patch("bot.strategy.sentiment.requests.get", return_value=_mock_resp(200, payload)), \
         patch("bot.strategy.sentiment.NEWSAPI_KEY", "test-key"):
        result = get_news_headlines("AAPL")
    assert result == ["Real headline"]


def test_get_news_headlines_does_not_call_api_without_key():
    with patch("bot.strategy.sentiment.NEWSAPI_KEY", ""), \
         patch("bot.strategy.sentiment.requests.get") as mock_get:
        get_news_headlines("AAPL")
    mock_get.assert_not_called()


def test_get_news_headlines_raises_for_500_error():
    with patch("bot.strategy.sentiment.requests.get", return_value=_mock_resp(500)), \
         patch("bot.strategy.sentiment.NEWSAPI_KEY", "test-key"):
        result = get_news_headlines("AAPL")
    # 500 falls through raise_for_status → caught by outer except → returns []
    assert result == []


# --- batch_sentiment_scores ---

def test_batch_sentiment_scores_returns_neutral_when_finbert_unavailable():
    with patch("bot.strategy.sentiment._get_finbert", return_value=None):
        result = batch_sentiment_scores({"AAPL": ["headline"], "MSFT": []})
    assert result == {"AAPL": 0.0, "MSFT": 0.0}


def test_batch_sentiment_scores_empty_headlines_returns_neutral():
    with patch("bot.strategy.sentiment._get_finbert", return_value=None):
        result = batch_sentiment_scores({})
    assert result == {}


def test_batch_sentiment_scores_all_empty_texts():
    mock_pipe = MagicMock()
    with patch("bot.strategy.sentiment._get_finbert", return_value=mock_pipe):
        result = batch_sentiment_scores({"AAPL": [], "MSFT": []})
    assert result == {"AAPL": 0.0, "MSFT": 0.0}
    mock_pipe.assert_not_called()
