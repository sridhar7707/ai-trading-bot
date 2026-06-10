"""Pre-market sentiment prefetch — run once before the trading loop starts.
Collects headlines for all universe symbols, runs FinBERT batch inference,
and writes results to data/sentiment_today.json so the trading loop can
skip the 3-5 min in-cycle BERT pass.
"""
from __future__ import annotations
import sys, os, json
from pathlib import Path
from datetime import date, datetime, timezone
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from bot.strategy.sentiment import collect_headlines, batch_sentiment_scores

DATA_DIR = "data"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    from config import SYMBOLS
    symbols: list[str] = list(SYMBOLS)

    universe_path = f"{DATA_DIR}/universe_today.json"
    try:
        if os.path.exists(universe_path):
            with open(universe_path) as f:
                payload = json.load(f)
            if payload.get("date") == date.today().isoformat():
                syms = payload.get("symbols", [])
                if syms:
                    symbols = syms
    except Exception as e:
        logger.warning(f"Could not load screened universe — using config.SYMBOLS: {e}")

    logger.info(f"Prefetching sentiment for {len(symbols)} symbols: {symbols[:5]}...")

    symbol_headlines: dict[str, list[str]] = {}
    for sym in symbols:
        try:
            symbol_headlines[sym] = collect_headlines(sym)
        except Exception as e:
            logger.warning(f"Headline collection failed for {sym}: {e}")
            symbol_headlines[sym] = []

    scores = batch_sentiment_scores(symbol_headlines)

    output = {
        "date": date.today().isoformat(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "scores": scores,
    }
    out_path = f"{DATA_DIR}/sentiment_today.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Sentiment prefetch complete — {len(scores)} scores saved to {out_path}")


if __name__ == "__main__":
    main()
