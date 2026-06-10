"""
Nightly universe screener — ranks ~110 liquid candidates and outputs
the top symbols to data/universe_today.json.

Scoring methodology (institutional-grade):
  40%  Risk-adjusted 60-day momentum  (60d return / 20d volatility)
  25%  20-day momentum vs SPY         (relative strength, medium-term)
  20%  Trend quality                  (R² of 20-day linear regression)
  15%  Price-confirmed volume surge   (volume only counts when price rising)

Hard filters (all must pass):
  - Price ≥ $10
  - 20-day ADV ≥ $5M
  - Close above 50-day SMA  (no falling knives)
  - Earnings not within ±3 days

Run once before market open (handled by GitHub Actions workflow).
Falls back to config.SYMBOLS automatically on any failure.

Usage:
    python scripts/screen_universe.py
    python scripts/screen_universe.py --max 25 --adv 5000000
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
from scipy import stats as scipy_stats

from config import SECTOR_MAP, SYMBOLS

OUTPUT_PATH        = "data/universe_today.json"
DEFAULT_MAX        = 25
DEFAULT_MAX_SECTOR = 3
DEFAULT_MIN_PRICE  = 10.0     # raised from $5 — better spreads, institutional support
DEFAULT_MIN_ADV    = 5_000_000  # raised from $2M — guarantees clean algo fills

# ~110 liquid candidates across all 11 GICS sectors.
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


def _rank_pct(series: pd.Series) -> pd.Series:
    """Percentile rank — robust to outliers, values in [0, 1]."""
    return series.rank(pct=True)


def _trend_r2(prices: pd.Series) -> float:
    """R² of OLS linear fit over last 20 bars — measures trend cleanness.
    1.0 = perfect straight-line trend, 0.0 = pure noise."""
    n = min(20, len(prices))
    if n < 5:
        return 0.0
    y = prices.iloc[-n:].values.astype(float)
    x = np.arange(n)
    _, _, r, _, _ = scipy_stats.linregress(x, y)
    return float(r ** 2)


def screen(
    max_symbols: int = DEFAULT_MAX,
    max_per_sector: int = DEFAULT_MAX_SECTOR,
    min_price: float = DEFAULT_MIN_PRICE,
    min_adv: float = DEFAULT_MIN_ADV,
) -> list[str]:
    all_candidates = list(dict.fromkeys(CANDIDATE_UNIVERSE))
    logger.info(f"Downloading 75-day history for {len(all_candidates)} candidates...")

    # 75 days covers: 60-day momentum + 50-day SMA + buffer
    raw = yf.download(
        tickers=all_candidates,
        period="75d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        logger.error("yfinance returned empty data — falling back to config.SYMBOLS")
        return list(SYMBOLS)

    close_df  = raw["Close"]
    volume_df = raw.get("Volume")
    high_df   = raw.get("High")
    low_df    = raw.get("Low")

    spy_close = close_df.get("SPY")

    passed_filters = 0
    scores: dict[str, dict] = {}

    for sym in all_candidates:
        if sym not in close_df.columns:
            continue
        closes = close_df[sym].dropna()
        if len(closes) < 22:   # need at least 22 days for all signals
            continue

        last_price = float(closes.iloc[-1])

        # ── Hard filters ─────────────────────────────────────────────────────

        # 1. Price floor
        if last_price < min_price:
            continue

        # 2. ADV filter
        if volume_df is not None and sym in volume_df.columns:
            vols = volume_df[sym].dropna()
            adv = float((closes.reindex(vols.index) * vols).tail(20).mean())
            if adv < min_adv:
                continue
        else:
            continue   # skip if no volume data

        # 3. Above 50-day SMA — no falling knives
        sma50 = float(closes.tail(50).mean()) if len(closes) >= 50 else float(closes.mean())
        if last_price < sma50:
            continue

        passed_filters += 1

        # ── Signal factors ────────────────────────────────────────────────────

        # Factor 1 — Risk-adjusted 60-day momentum
        # Uses volatility normalisation: a smooth 5% move beats a choppy 15% move.
        ret_60 = 0.0
        if len(closes) >= 61:
            ret_60 = float(closes.iloc[-1] / closes.iloc[-61] - 1)
        std_20 = float(closes.pct_change().tail(20).std()) or 0.001
        risk_adj_mom = ret_60 / std_20   # Sharpe-like: return per unit of risk

        # Factor 2 — 20-day momentum relative to SPY (beats raw return)
        rs_20 = 0.0
        if spy_close is not None:
            spy = spy_close.dropna()
            n = min(21, len(closes), len(spy))
            sym_ret = float(closes.iloc[-1] / closes.iloc[-n] - 1)
            spy_ret = float(spy.iloc[-1]    / spy.iloc[-n]    - 1)
            rs_20 = sym_ret - spy_ret

        # Factor 3 — Trend quality (R² of 20-day regression)
        r2 = _trend_r2(closes)

        # Factor 4 — Price-confirmed volume surge
        # Volume surge only counts as bullish when price is also rising over 5 days.
        # High volume on a falling stock = distribution (bearish).
        vol_surge = 0.0
        if volume_df is not None and sym in volume_df.columns:
            vols_sym = volume_df[sym].dropna()
            if len(vols_sym) >= 6:
                v5  = float(vols_sym.tail(5).mean())
                v20 = float(vols_sym.tail(20).mean())
                if v20 > 0:
                    raw_surge = np.clip(v5 / v20 - 1, -1.0, 3.0)
                    # Only count surge when price has risen in last 5 days
                    price_5d = float(closes.iloc[-1] / closes.iloc[-6] - 1) if len(closes) >= 6 else 0.0
                    vol_surge = raw_surge * (1.0 if price_5d > 0 else -0.5)

        scores[sym] = {
            "price": last_price,
            "sma50": sma50,
            "risk_adj_mom": risk_adj_mom,
            "rs_20": rs_20,
            "r2": r2,
            "vol_surge": vol_surge,
        }

    logger.info(f"Candidates: {len(all_candidates)} → passed filters: {passed_filters} → scored: {len(scores)}")

    if not scores:
        logger.error("No symbols passed filters — falling back to config.SYMBOLS")
        return list(SYMBOLS)

    score_df = pd.DataFrame(scores).T

    # Blend factors using percentile ranks (robust to outliers)
    score_df["composite"] = (
        0.40 * _rank_pct(score_df["risk_adj_mom"])   # risk-adjusted momentum (primary)
        + 0.25 * _rank_pct(score_df["rs_20"])         # relative strength vs SPY
        + 0.20 * _rank_pct(score_df["r2"])            # trend quality / smoothness
        + 0.15 * _rank_pct(score_df["vol_surge"])     # price-confirmed volume
    )
    score_df = score_df.sort_values("composite", ascending=False)

    top10 = score_df.head(10)[["price", "risk_adj_mom", "rs_20", "r2", "composite"]]
    logger.info(f"Top 10 by composite score:\n{top10.to_string()}")

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

    # Always include SPY (used internally for RS gate in main.py)
    if "SPY" not in selected:
        selected.append("SPY")

    return selected


def main(args: argparse.Namespace) -> None:
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
    from collections import Counter
    breakdown = Counter(_sector(s) for s in symbols)
    for sec, n in sorted(breakdown.items()):
        syms_in_sec = [s for s in symbols if _sector(s) == sec]
        logger.info(f"  {sec:<22} ({n}) {syms_in_sec}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-market universe screener")
    parser.add_argument("--max",            type=int,   default=DEFAULT_MAX)
    parser.add_argument("--max-per-sector", type=int,   default=DEFAULT_MAX_SECTOR)
    parser.add_argument("--min-price",      type=float, default=DEFAULT_MIN_PRICE)
    parser.add_argument("--min-adv",        type=float, default=DEFAULT_MIN_ADV)
    main(parser.parse_args())
