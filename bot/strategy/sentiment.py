from __future__ import annotations
import os
import requests
from datetime import date, timedelta
from loguru import logger

NEWSAPI_KEY    = os.getenv("NEWSAPI_KEY", "")
# SEC EDGAR requires a valid User-Agent with contact info per https://www.sec.gov/os/webmaster-faq
# Set SEC_USER_AGENT env var (e.g. "ai-trading-bot your@email.com") — do not hardcode in source.
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "ai-trading-bot contact@example.com")

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
        # Rolling 90-day window — hardcoded year would stop returning results as time passes
        startdt = (date.today() - timedelta(days=90)).isoformat()
        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={"q": f'"{ticker}"', "forms": "8-K,10-Q", "dateRange": "custom", "startdt": startdt},
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


def collect_headlines(ticker: str) -> list[str]:
    """Fetch headlines without running FinBERT — for use with batch_sentiment_scores."""
    return get_news_headlines(ticker) + get_sec_headlines(ticker)


def batch_sentiment_scores(symbol_headlines: dict[str, list[str]]) -> dict[str, float]:
    """Run FinBERT once across all symbols instead of once per symbol.

    Takes {symbol: [headlines]} and returns {symbol: score}.
    All texts go through a single pipeline call — ~12x faster than calling
    get_sentiment_score() in a loop.
    """
    pipe = _get_finbert()
    if pipe is None:
        return {sym: 0.0 for sym in symbol_headlines}

    # Flatten texts, tracking which symbol each belongs to
    order: list[tuple[str, int]] = []
    flat_texts: list[str] = []
    for sym, texts in symbol_headlines.items():
        batch = [t[:512] for t in texts[:10]]
        order.append((sym, len(batch)))
        flat_texts.extend(batch)

    if not flat_texts:
        return {sym: 0.0 for sym in symbol_headlines}

    try:
        results = pipe(flat_texts)
    except Exception as e:
        logger.warning(f"Batch FinBERT failed: {e}")
        return {sym: 0.0 for sym in symbol_headlines}

    scores: dict[str, float] = {}
    idx = 0
    for sym, count in order:
        sym_scores = []
        for r in results[idx: idx + count]:
            s = r["score"]
            sym_scores.append(s if r["label"] == "positive" else (-s if r["label"] == "negative" else 0.0))
        scores[sym] = sum(sym_scores) / len(sym_scores) if sym_scores else 0.0
        idx += count

    return scores
