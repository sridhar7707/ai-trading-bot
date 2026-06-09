import os
import requests
from loguru import logger

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
SEC_USER_AGENT = "ai-trading-bot ksri77@gmail.com"

_finbert_pipeline = None


def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline
            _finbert_pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
            )
            logger.info("FinBERT loaded.")
        except Exception as e:
            logger.warning(f"FinBERT unavailable: {e}")
    return _finbert_pipeline


def _finbert_score(texts: list[str]) -> float:
    """Score a list of texts with FinBERT. Returns mean score in [-1, +1]."""
    if not texts:
        return 0.0
    pipe = _get_finbert()
    if pipe is None:
        return 0.0
    scores = []
    for text in texts[:10]:
        try:
            result = pipe(text[:512])[0]
            s = result["score"]
            if result["label"] == "positive":
                scores.append(s)
            elif result["label"] == "negative":
                scores.append(-s)
            else:
                scores.append(0.0)
        except Exception as e:
            logger.warning(f"FinBERT scoring failed on text: {e}")
    return sum(scores) / len(scores) if scores else 0.0


def get_news_headlines(ticker: str) -> list[str]:
    if not NEWSAPI_KEY:
        return []
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": ticker,
                "sortBy": "publishedAt",
                "pageSize": 10,
                "apiKey": NEWSAPI_KEY,
                "language": "en",
            },
            timeout=5,
        )
        return [a["title"] for a in resp.json().get("articles", []) if a.get("title")]
    except Exception as e:
        logger.warning(f"NewsAPI failed for {ticker}: {e}")
        return []


def get_sec_headlines(ticker: str) -> list[str]:
    """Pull recent SEC filing descriptions from EDGAR full-text search (no key needed)."""
    try:
        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={"q": f'"{ticker}"', "forms": "8-K,10-Q", "dateRange": "custom", "startdt": "2024-01-01"},
            headers={"User-Agent": SEC_USER_AGENT},
            timeout=5,
        )
        hits = resp.json().get("hits", {}).get("hits", [])
        return [
            h["_source"].get("period_of_report", "") + " " + ", ".join(h["_source"].get("display_names", []))
            for h in hits[:5]
        ]
    except Exception as e:
        logger.warning(f"SEC EDGAR failed for {ticker}: {e}")
        return []


def get_sentiment_score(ticker: str) -> float:
    """Combined FinBERT sentiment score for a ticker in [-1, +1]. Returns 0.0 on failure."""
    headlines = get_news_headlines(ticker) + get_sec_headlines(ticker)
    if not headlines:
        return 0.0
    return _finbert_score(headlines)
