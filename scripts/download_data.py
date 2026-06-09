"""Download historical OHLCV data from yfinance and save to data/raw/."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yfinance as yf
import pandas as pd  # noqa: E402 — needed after sys.path fix
from loguru import logger
from config import TRAINING_SYMBOLS, BENCHMARK

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import datetime
# 2015 captures: 2015-16 correction, 2018 rate-hike volatility,
# 2020 COVID crash+recovery, 2022 bear market — diverse regimes improve all models
START_DATE = "2015-01-01"
END_DATE = datetime.date.today().isoformat()


def download(symbol: str):
    logger.info(f"Downloading {symbol}...")
    df = yf.download(symbol, start=START_DATE, end=END_DATE, interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.to_csv(f"{OUTPUT_DIR}/{symbol}.csv")
    logger.info(f"Saved {symbol} — {len(df)} rows")


if __name__ == "__main__":
    all_symbols = list(dict.fromkeys(TRAINING_SYMBOLS + [BENCHMARK]))
    for sym in all_symbols:
        try:
            download(sym)
        except Exception as e:
            logger.error(f"Failed {sym}: {e}")
    logger.info(f"Download complete — {len(all_symbols)} symbols, {START_DATE} to {END_DATE}.")
