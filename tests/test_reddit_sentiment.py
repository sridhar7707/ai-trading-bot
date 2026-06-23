from unittest.mock import patch
from bot.strategy.reddit_sentiment import get_wsb_sentiment


def test_get_wsb_sentiment_returns_zeros_when_no_credentials():
    with patch("bot.strategy.reddit_sentiment.REDDIT_CLIENT_ID", ""):
        with patch("bot.strategy.reddit_sentiment.REDDIT_CLIENT_SECRET", ""):
            result = get_wsb_sentiment("AAPL")
    assert result == {"mentions": 0, "sentiment": 0.0}


def test_get_wsb_sentiment_returns_dict_shape():
    with patch("bot.strategy.reddit_sentiment.REDDIT_CLIENT_ID", ""):
        result = get_wsb_sentiment("TSLA")
    assert "mentions" in result
    assert "sentiment" in result
    assert isinstance(result["mentions"], int)
    assert isinstance(result["sentiment"], float)
