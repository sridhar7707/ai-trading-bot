"""
Nightly universe screener — ranks ~100 liquid candidates by momentum +
relative strength + volume surge, outputs top symbols to
data/universe_today.json.

Run once before market open (handled by GitHub Actions). If anything
fails, the bot falls back to config.SYMBOLS automatically.

Usage:
    python scripts/screen_universe.py
    python scripts/screen_universe.py --max 30 --adv 5000000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from config import SECTOR_MAP, SYMBOLS

OUTPUT_PATH = "data/universe_today.json"
DEFAULT_MAX   = 25
DEFAULT_MAX_SECTOR = 3
DEFAULT_MIN_PRICE  = 5.0
DEFAULT_MIN_ADV    = 2_000_000   # avg daily dollar volume (price × volume)

# ~110 liquid candidates across all 11 GICS sectors.
# Screener picks the best N of these each day — the bot only trades SYMBOLS
# unless overridden by the screener output.
CANDIDATE_UNIVERSE: list[str] = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AMD", "INTC", "QCOM", "TXN", "AVGO",
    "CRM", "ADBE", "ORCL", "CSCO", "NOW", "SNOW", "PANW", "MU",
    # Communication Services
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ",
    # Consumer Discretionary
    "AMZN", "TSLA", "NKE", "MCD", "SBUX", "HD", "LOW", "TGT", "BKNG",
    # Consumer Staples
    "WMT", "PG", "KO", "PEP", "COST", "CL", "MDLZ",
    # Energy
    "XOM", "CVX", "COP", "SLB", "OXY", "MPC", "PSX",
    # Financials
    "JPM", "BAC", "GS", "MS", "WFC", "V", "MA", "AXP", "BLK", "BRK-B",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "BMY", "AMGN", "GILD", "CVS",
    # Industrials
    "CAT", "HON", "DE", "UPS", "GE", "RTX", "BA", "MMM", "LMT",
    # Materials
    "LIN", "APD", "NEM", "FCX", "DD",
    # Real Estate
    "PLD", "AMT", "EQIX", "SPG",
    # Utilities
    "NEE", "DUK", "SO",
    # Broad ETFs
    "SPY", "QQQ", "VTI",
    # Sector ETFs
    "XLF", "XLV", "XLE", "XLK", "XLI", "XLY", "XLP", "XLC",
    # Macro ETFs
    "GLD", "TLT",
]

_EXTRA_SECTOR: dict[str, str] = {
    "AMD": "Technology",  "INTC": "Technology",  "QCOM": "Technology",
    "TXN": "Technology",  "AVGO": "Technology",  "CRM": "Technology",
    "ADBE": "Technology", "ORCL": "Technology",  "CSCO": "Technology",
    "NOW": "Technology",  "SNOW": "Technology",  "PANW": "Technology",
    "MU": "Technology",
    "NFLX": "Comm_Services",  "DIS": "Comm_Services",
    "CMCSA": "Comm_Services", "T": "Comm_Services", "VZ": "Comm_Services",
    "NKE": "Consumer_Disc",  "MCD": "Consumer_Disc",  "SBUX": "Consumer_Disc",
    "HD": "Consumer_Disc",   "LOW": "Consumer_Disc",  "TGT": "Consumer_Disc",
    "BKNG": "Consumer_Disc",
    "PG": "Consumer_Staples",   "KO": "Consumer_Staples",  "PEP": "Consumer_Staples",
    "COST": "Consumer_Staples", "CL": "Consumer_Staples",  "MDLZ": "Consumer_Staples",
    "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "OXY": "Energy", "MPC": "Energy", "PSX": "Energy",
    "BAC": "Financials", "GS": "Financials",  "MS": "Financials",
    "WFC": "Financials", "V": "Financials",   "MA": "Financials",
    "AXP": "Financials", "BLK": "Financials",
    "UNH": "Healthcare",  "PFE": "Healthcare",  "ABBV": "Healthcare",
    "MRK": "Healthcare",  "LLY": "Healthcare",  "BMY": "Healthcare",
    "AMGN": "Healthcare", "GILD": "Healthcare", "CVS": "Healthcare",
    "CAT": "Industrials", "HON": "Industrials", "DE": "Industrials",
    "UPS": "Industrials", "GE": "Industrials",  "RTX": "Industrials",
    "BA": "Industrials",  "MMM": "Industrials", "LMT": "Industrials",
    "LIN": "Materials", "APD": "Materials", "NEM": "Materials",
    "FCX": "Materials", "DD": "Materials",
    "PLD": "Real_Estate", "AMT": "Real_Estate",
    "EQIX": "Real_Estate", "SPG": "Real_Estate",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
}

FULL_SECTOR_MAP: dict[str, str] = {**_EXTRA_SECTOR, **SECTOR_MAP}


def _sector(sym: str) -> str:
    return FULL_SECTOR_MAP.get(sym, "Unknown")


def _rank_normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize after rank transform — robust to outliers."""
    ranked = series.rank(pct=True)
    return ranked


def screen(
    max_symbols: int = DEFAULT_MAX,
    max_per_sector: int = DEFAULT_MAX_SECTOR,
    min_price: float = DEFAULT_MIN_PRICE,
    min_adv: float = DEFAULT_MIN_ADV,
) -> list[str]:
    all_candidates = list(dict.fromkeys(CANDIDATE_UNIVERSE))
    logger.info(f"Downloading 35-day history for {len(all_candidates)} candidates...")

    raw = yf.download(
        tickers=all_candidates,
        period="35d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        logger.error("yfinance returned empty data — falling back to config.SYMBOLS")
        return list(SYMBOLS)

    # yfinance returns MultiIndex columns (field, ticker) for multi-ticker downloads
    close_df  = raw["Close"]  if "Close"  in raw.columns.get_level_values(0) else raw
    volume_df = raw["Volume"] if "Volume" in raw.columns.get_level_values(0) else None

    spy_close = close_df.get("SPY")

    scores: dict[str, dict] = {}
    for sym in all_candidates:
        if sym not in close_df.columns:
            continue
        closes = close_df[sym].dropna()
        if len(closes) < 6:
            continue

        last_price = float(closes.iloc[-1])
        if last_price < min_price:
            continue

        # ADV filter
        if volume_df is not None and sym in volume_df.columns:
            vol = volume_df[sym].dropna()
            adv = float((closes.reindex(vol.index) * vol).tail(20).mean())
            if adv < min_adv:
                continue

        # 20-day momentum
        mom_20 = float(closes.iloc[-1] / closes.iloc[max(-21, -len(closes))] - 1) if len(closes) >= 6 else 0.0

        # 5-day relative strength vs SPY
        rs_5 = 0.0
        if spy_close is not None and len(closes) >= 6 and len(spy_close.dropna()) >= 6:
            sym_5d  = float(closes.iloc[-1]                   / closes.iloc[max(-6, -len(closes))]   - 1)
            spy_5d  = float(spy_close.dropna().iloc[-1]        / spy_close.dropna().iloc[max(-6, -len(spy_close.dropna()))] - 1)
            rs_5 = sym_5d - spy_5d

        # Volume surge (5d avg / 20d avg)
        vol_surge = 0.0
        if volume_df is not None and sym in volume_df.columns:
            vols = volume_df[sym].dropna()
            if len(vols) >= 6:
                v5  = float(vols.tail(5).mean())
                v20 = float(vols.tail(20).mean())
                if v20 > 0:
                    vol_surge = np.clip(v5 / v20 - 1, -1.0, 3.0)

        scores[sym] = {
            "price": last_price,
            "mom_20": mom_20,
            "rs_5": rs_5,
            "vol_surge": vol_surge,
        }

    if not scores:
        logger.error("No symbols passed filters — falling back to config.SYMBOLS")
        return list(SYMBOLS)

    score_df = pd.DataFrame(scores).T
    # Normalize each factor to [0,1] via rank percentile, then blend
    score_df["score"] = (
        0.50 * _rank_normalize(score_df["mom_20"])
        + 0.30 * _rank_normalize(score_df["rs_5"])
        + 0.20 * _rank_normalize(score_df["vol_surge"])
    )
    score_df = score_df.sort_values("score", ascending=False)

    logger.info(f"Top 10 candidates: {score_df.head(10).index.tolist()}")

    # Apply sector cap: at most max_per_sector per sector
    selected: list[str] = []
    sector_counts: dict[str, int] = defaultdict(int)
    for sym in score_df.index:
        sec = _sector(sym)
        if sector_counts[sec] >= max_per_sector:
            continue
        selected.append(sym)
        sector_counts[sec] += 1
        if len(selected) >= max_symbols:
            break

    # Always keep SPY (used internally for relative-strength gate in main.py)
    if "SPY" not in selected:
        selected.append("SPY")

    return selected


def main(args: argparse.Namespace):
    os.makedirs("data", exist_ok=True)
    today = date.today().isoformat()

    try:
        symbols = screen(
            max_symbols=args.max,
            max_per_sector=args.max_per_sector,
            min_price=args.min_price,
            min_adv=args.min_adv,
        )
    except Exception as exc:
        logger.error(f"Screener failed: {exc} — falling back to config.SYMBOLS")
        symbols = list(SYMBOLS)

    payload = {
        "date": today,
        "screened_at": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "count": len(symbols),
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Universe written to {OUTPUT_PATH}: {len(symbols)} symbols")
    # Print sector breakdown
    from collections import Counter
    breakdown = Counter(_sector(s) for s in symbols)
    for sec, n in sorted(breakdown.items()):
        syms_in_sec = [s for s in symbols if _sector(s) == sec]
        logger.info(f"  {sec:<22} ({n}) {syms_in_sec}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-market universe screener")
    parser.add_argument("--max",          type=int,   default=DEFAULT_MAX,
                        help="Max symbols to select (default 25)")
    parser.add_argument("--max-per-sector", type=int, default=DEFAULT_MAX_SECTOR,
                        help="Max symbols per sector (default 3)")
    parser.add_argument("--min-price",    type=float, default=DEFAULT_MIN_PRICE,
                        help="Min stock price filter (default $5)")
    parser.add_argument("--min-adv",      type=float, default=DEFAULT_MIN_ADV,
                        help="Min avg daily dollar volume (default $2M)")
    main(parser.parse_args())
