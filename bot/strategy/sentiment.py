from __future__ import annotations
import json
import os
import sqlite3
import time
import requests
from datetime import date, timedelta
from pathlib import Path
from loguru import logger

NEWSAPI_KEY    = os.getenv("NEWSAPI_KEY", "")
# SEC EDGAR requires a valid User-Agent with contact info per https://www.sec.gov/os/webmaster-faq
# Set SEC_USER_AGENT env var (e.g. "ai-trading-bot your@email.com") — do not hardcode in source.
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "ai-trading-bot contact@example.com")

# L1 in-process cache — resets on process restart (GitHub Actions: new process each cycle).
# Keyed by "TICKER:YYYY-MM-DD".
_NEWS_DAY_CACHE: dict[str, list[str]] = {}

# L2 DB-backed cache — persists across process restarts so each symbol is fetched
# at most once per calendar day regardless of how many bot cycles run.
_NEWS_DB_PATH: str = os.getenv("TRADE_DB_PATH", "trades.db")

def _news_db_get(ticker: str, today: str) -> list[str] | None:
    """Return today's cached headlines from DB, or None if not cached yet."""
    try:
        db = Path(_NEWS_DB_PATH)
        if not db.exists():
            return None
        con = sqlite3.connect(str(db), check_same_thread=False, timeout=3)
        try:
            row = con.execute(
                "SELECT headlines_json FROM news_cache WHERE symbol=? AND fetch_date=?",
                (ticker, today),
            ).fetchone()
            return json.loads(row[0]) if row else None
        finally:
            con.close()
    except Exception:
        return None


def _news_db_set(ticker: str, today: str, headlines: list[str]) -> None:
    """Write today's headlines to DB cache. Creates the table on first use."""
    try:
        db = Path(_NEWS_DB_PATH)
        if not db.exists():
            return
        con = sqlite3.connect(str(db), check_same_thread=False, timeout=3)
        try:
            con.execute(
                "CREATE TABLE IF NOT EXISTS news_cache "
                "(symbol TEXT, fetch_date TEXT, headlines_json TEXT, cached_at TEXT, "
                "PRIMARY KEY (symbol, fetch_date))"
            )
            con.execute(
                "INSERT OR REPLACE INTO news_cache VALUES (?,?,?,datetime('now'))",
                (ticker, today, json.dumps(headlines)),
            )
            con.commit()
        finally:
            con.close()
    except Exception:
        pass


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
    today = date.today().isoformat()
    _cache_key = f"{ticker}:{today}"

    # L1: in-process memory cache
    if _cache_key in _NEWS_DAY_CACHE:
        logger.debug(f"NewsAPI L1 cache hit — {ticker} (quota preserved)")
        return _NEWS_DAY_CACHE[_cache_key]

    # L2: DB cache — survives process restarts on GitHub Actions
    db_cached = _news_db_get(ticker, today)
    if db_cached is not None:
        _NEWS_DAY_CACHE[_cache_key] = db_cached
        logger.debug(f"NewsAPI L2 DB cache hit — {ticker} (quota preserved)")
        return db_cached

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
        if resp.status_code == 426:
            logger.error(
                "[API RATE LIMIT] NewsAPI daily quota exhausted (HTTP 426) — "
                "all remaining symbols will use neutral sentiment. Upgrade plan or reduce universe."
            )
            return []
        if resp.status_code == 429:
            logger.warning(
                f"[API RATE LIMIT] NewsAPI rate-limited (HTTP 429) for {ticker} — "
                "returning empty headlines"
            )
            return []
        resp.raise_for_status()
        headlines = [a["title"] for a in resp.json().get("articles", []) if a.get("title")]
        _NEWS_DAY_CACHE[_cache_key] = headlines
        _news_db_set(ticker, today, headlines)
        return headlines
    except Exception as e:
        logger.warning(f"NewsAPI failed for {ticker}: {e}")
        return []


def get_sec_headlines(ticker: str) -> list[str]:
    """Pull recent SEC filing descriptions from EDGAR full-text search (no key needed).

    Throttled to ~8 req/s via a 0.13s sleep — SEC enforces a hard 10 req/s limit
    and will return HTTP 403 on violation.
    """
    try:
        startdt = (date.today() - timedelta(days=90)).isoformat()
        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={"q": f'"{ticker}"', "forms": "8-K,10-Q", "dateRange": "custom", "startdt": startdt},
            headers={"User-Agent": SEC_USER_AGENT},
            timeout=5,
        )
        if resp.status_code in (429, 403):
            logger.warning(
                f"[API RATE LIMIT] SEC EDGAR rate-limited (HTTP {resp.status_code}) for {ticker} — "
                "returning empty headlines"
            )
            return []
        hits = resp.json().get("hits", {}).get("hits", [])
        return [
            h["_source"].get("period_of_report", "") + " " + ", ".join(h["_source"].get("display_names", []))
            for h in hits[:5]
        ]
    except Exception as e:
        logger.warning(f"SEC EDGAR failed for {ticker}: {e}")
        return []
    finally:
        time.sleep(0.13)  # throttle to ~7 req/s — SEC hard limit is 10 req/s


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
