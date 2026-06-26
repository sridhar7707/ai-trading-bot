import os
import threading
from loguru import logger

# Catch PRAW's rate-limit exception without requiring praw to be installed at import time
try:
    from praw.exceptions import RateLimitExceeded as _RedditRateLimit
except ImportError:
    _RedditRateLimit = None  # type: ignore[assignment,misc]

REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "ai-trading-bot/1.0")

# Module-level singleton — one OAuth session per process lifetime, not per call.
_reddit = None
# Circuit breaker: set True on first 401 to disable all subsequent calls this session.
# Lock ensures the flag is read/written atomically across parallel sentiment threads.
_reddit_auth_failed = False
_reddit_lock = threading.Lock()


def _get_reddit():
    with _reddit_lock:
        global _reddit, _reddit_auth_failed
        if _reddit_auth_failed:
            return None
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
    Returns {"mentions": 0, "sentiment": 0.0} if Reddit credentials are not configured
    or if credentials are invalid (401 circuit breaker is open).
    """
    reddit = _get_reddit()
    if reddit is None:
        return {"mentions": 0, "sentiment": 0.0}

    try:
        from bot.strategy.sentiment import _finbert_score
        subreddit = reddit.subreddit("wallstreetbets+investing+stocks")
        titles = [post.title for post in subreddit.search(ticker, limit=50, time_filter="day")]
        sentiment = _finbert_score(titles) if titles else 0.0
        logger.info(f"Reddit {ticker}: mentions={len(titles)}, sentiment={sentiment:.2f}")
        return {"mentions": len(titles), "sentiment": sentiment}

    except Exception as e:
        _emsg = str(e).lower()
        if _RedditRateLimit and isinstance(e, _RedditRateLimit):
            logger.warning(
                f"[API RATE LIMIT] Reddit rate-limited for {ticker} — "
                f"returning neutral sentiment (will recover next cycle): {e}"
            )
        elif "401" in _emsg or "unauthorized" in _emsg or "received 401" in _emsg:
            with _reddit_lock:
                global _reddit, _reddit_auth_failed
                if not _reddit_auth_failed:  # log exactly once, even under parallel calls
                    _reddit = None
                    _reddit_auth_failed = True
                    logger.warning(
                        "[API RATE LIMIT] Reddit credentials invalid (HTTP 401) — "
                        "disabling Reddit sentiment for this session. "
                        "Set REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET as repository secrets."
                    )
        else:
            logger.warning(f"Reddit sentiment failed for {ticker}: {e}")
        return {"mentions": 0, "sentiment": 0.0}
