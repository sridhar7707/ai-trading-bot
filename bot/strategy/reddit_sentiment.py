import os
from loguru import logger

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "ai-trading-bot/1.0")


def get_wsb_sentiment(ticker: str) -> dict:
    """
    Returns WSB + investing subreddit mention count and FinBERT sentiment score.
    Returns {"mentions": 0, "sentiment": 0.0} if Reddit credentials are not configured.
    """
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        return {"mentions": 0, "sentiment": 0.0}

    try:
        import praw
        from bot.strategy.sentiment import _finbert_score

        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )

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
