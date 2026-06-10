import os
from loguru import logger

REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "ai-trading-bot/1.0")

# Module-level singleton — one OAuth session per process lifetime, not per call.
# Creating a new praw.Reddit per call issues a fresh OAuth token every time
# (up to 1,950 requests/day in production), which can trigger Reddit rate limiting.
_reddit = None


def _get_reddit():
    global _reddit
    if _reddit is None and REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
        try:
            import praw
            _reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT,
            )
        except Exception as e:
            logger.warning(f"PRAW initialization failed: {e}")
    return _reddit


def get_wsb_sentiment(ticker: str) -> dict:
    """
    Returns WSB + investing subreddit mention count and FinBERT sentiment score.
    Returns {"mentions": 0, "sentiment": 0.0} if Reddit credentials are not configured.
    """
    reddit = _get_reddit()
    if reddit is None:
        return {"mentions": 0, "sentiment": 0.0}

    try:
        from bot.strategy.sentiment import _finbert_score
        subreddit = reddit.subreddit("wallstreetbets+investing+stocks")
        titles = []
        for post in subreddit.search(ticker, limit=50, time_filter="day"):
            titles.append(post.title)

        sentiment = _finbert_score(titles) if titles else 0.0
        logger.info(f"Reddit {ticker}: mentions={len(titles)}, sentiment={sentiment:.2f}")
        return {"mentions": len(titles), "sentiment": sentiment}

    except Exception as e:
        logger.warning(f"Reddit sentiment failed for {ticker}: {e}")
        return {"mentions": 0, "sentiment": 0.0}
