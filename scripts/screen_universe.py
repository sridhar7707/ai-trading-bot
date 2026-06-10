"""
Pre-market universe screener — hedge fund multi-factor approach.

Two-stage pipeline:
  Stage 1 — Hard filters (price, ADV, above-50-SMA, beta range)
  Stage 2 — Factor scoring with regime-adaptive weights

Factors (all computed from 1-year daily price data, no extra API calls):
  • Risk-adjusted 60-day momentum  (return / 20d volatility)
  • 20-day relative strength vs SPY
  • Trend quality R²               (OLS fit over last 20 bars)
  • 52-week high proximity         (O'Neil breakout signal)
  • Price-confirmed volume surge   (volume only positive when price rising)
  • Market beta                    (filter: 0.6–2.0 for swing/day trading)

Regime-adaptive weights:
  Bull (SPY > 50-SMA): emphasise momentum
  Bear (SPY < 50-SMA): emphasise RS + trend quality + defensive sectors

Final pass — correlation deduplication:
  Replace any pair with pairwise correlation >0.85 with the next-best
  uncorrelated candidate (avoids owning 3 stocks that move identically).

Falls back to config.SYMBOLS on any failure.

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
DEFAULT_MIN_PRICE  = 10.0
DEFAULT_MIN_ADV    = 5_000_000
BETA_MIN           = 0.6    # too-slow stocks don't move enough for intraday trading
BETA_MAX           = 2.0    # too-wild stocks blow stops before the edge plays out
CORR_THRESHOLD     = 0.85   # pairwise correlation above which one position is redundant

DEFENSIVE_SECTORS = {"Consumer_Staples", "Healthcare", "Utilities", "Bonds", "Commodities"}

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
    # Macro
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
    return series.rank(pct=True)


def _trend_r2(prices: pd.Series, n: int = 20) -> float:
    """R² of OLS linear fit over last n bars — 1.0 = perfect trend, 0.0 = noise."""
    window = min(n, len(prices))
    if window < 5:
        return 0.0
    y = prices.iloc[-window:].values.astype(float)
    x = np.arange(window)
    _, _, r, _, _ = scipy_stats.linregress(x, y)
    return float(r ** 2)


def _compute_beta(sym_rets: pd.Series, spy_rets: pd.Series) -> float:
    """OLS beta of sym_rets ~ spy_rets over overlapping period."""
    aligned = pd.concat([sym_rets, spy_rets], axis=1).dropna()
    if len(aligned) < 20:
        return 1.0
    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    var = aligned.iloc[:, 1].var()
    return float(cov / var) if var > 0 else 1.0


def _detect_regime(spy_closes: pd.Series) -> str:
    """Bull if SPY above its 50-day SMA, else Bear."""
    if len(spy_closes) < 50:
        return "BULL"
    sma50 = float(spy_closes.tail(50).mean())
    return "BULL" if float(spy_closes.iloc[-1]) > sma50 else "BEAR"


def _factor_weights(regime: str) -> dict[str, float]:
    if regime == "BULL":
        return {
            "risk_adj_mom": 0.35,
            "rs_20":        0.25,
            "r2":           0.20,
            "proximity_hi": 0.10,
            "vol_surge":    0.10,
        }
    # BEAR — rotate toward quality + RS, de-emphasise raw momentum
    return {
        "risk_adj_mom": 0.15,
        "rs_20":        0.35,
        "r2":           0.25,
        "proximity_hi": 0.05,
        "vol_surge":    0.05,
        "defensive":    0.15,   # bonus for defensive sectors
    }


def _corr_dedup(
    ranked: list[str],
    close_df: pd.DataFrame,
    max_sym: int,
    threshold: float = CORR_THRESHOLD,
) -> list[str]:
    """
    Walk the ranked list in order, keep a symbol only if its pairwise
    correlation with every already-selected symbol is below threshold.
    Gives the same effect as portfolio diversification without an optimizer.
    """
    selected: list[str] = []
    returns_cache: dict[str, pd.Series] = {}

    for sym in ranked:
        if len(selected) >= max_sym:
            break
        if sym not in close_df.columns:
            selected.append(sym)
            continue
        rets = close_df[sym].pct_change().dropna()
        returns_cache[sym] = rets

        too_corr = False
        for held in selected:
            if held not in returns_cache:
                continue
            aligned = pd.concat([rets, returns_cache[held]], axis=1).dropna()
            if len(aligned) < 10:
                continue
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            if corr > threshold:
                too_corr = True
                logger.debug(f"  {sym} ↔ {held} corr={corr:.2f} > {threshold} — skipped")
                break

        if not too_corr:
            selected.append(sym)

    return selected


def screen(
    max_symbols: int = DEFAULT_MAX,
    max_per_sector: int = DEFAULT_MAX_SECTOR,
    min_price: float = DEFAULT_MIN_PRICE,
    min_adv: float = DEFAULT_MIN_ADV,
) -> list[str]:
    candidates = list(dict.fromkeys(CANDIDATE_UNIVERSE))
    logger.info(f"Downloading 1-year history for {len(candidates)} candidates...")

    raw = yf.download(
        tickers=candidates,
        period="1y",
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

    spy_closes = close_df.get("SPY", pd.Series(dtype=float)).dropna()
    regime     = _detect_regime(spy_closes)
    weights    = _factor_weights(regime)
    logger.info(f"Market regime: {regime} | weights: {weights}")

    spy_rets = spy_closes.pct_change().dropna()

    # ── Stage 1: hard filters + factor computation ────────────────────────────
    scores: dict[str, dict] = {}
    n_filtered = 0

    for sym in candidates:
        if sym not in close_df.columns:
            continue
        closes = close_df[sym].dropna()
        if len(closes) < 60:
            continue

        last_price = float(closes.iloc[-1])
        if last_price < min_price:
            continue

        # ADV filter
        if volume_df is None or sym not in volume_df.columns:
            continue
        vols = volume_df[sym].dropna()
        adv  = float((closes.reindex(vols.index) * vols).tail(20).mean())
        if adv < min_adv:
            continue

        # Above 50-SMA — no falling knives
        sma50 = float(closes.tail(50).mean())
        if last_price < sma50:
            continue

        # Beta filter — too slow or too wild
        sym_rets = closes.pct_change().dropna()
        beta = _compute_beta(sym_rets, spy_rets)
        if beta < BETA_MIN or beta > BETA_MAX:
            continue

        n_filtered += 1

        # ── Factor 1: risk-adjusted 60-day momentum ───────────────────────────
        ret_60    = float(closes.iloc[-1] / closes.iloc[-61] - 1) if len(closes) >= 61 else 0.0
        std_20    = float(sym_rets.tail(20).std()) or 0.001
        risk_adj_mom = ret_60 / std_20

        # ── Factor 2: 20-day RS vs SPY ────────────────────────────────────────
        rs_20 = 0.0
        if len(spy_closes) >= 21 and len(closes) >= 21:
            sym_20d = float(closes.iloc[-1] / closes.iloc[-21] - 1)
            spy_20d = float(spy_closes.iloc[-1] / spy_closes.iloc[-21] - 1)
            rs_20 = sym_20d - spy_20d

        # ── Factor 3: trend quality R² ────────────────────────────────────────
        r2 = _trend_r2(closes)

        # ── Factor 4: 52-week high proximity (O'Neil breakout signal) ─────────
        hi_52wk      = float(closes.tail(252).max())
        proximity_hi = float(closes.iloc[-1] / hi_52wk) if hi_52wk > 0 else 0.5

        # ── Factor 5: price-confirmed volume surge ────────────────────────────
        vol_surge = 0.0
        if len(vols) >= 6:
            v5, v20 = float(vols.tail(5).mean()), float(vols.tail(20).mean())
            if v20 > 0:
                raw_surge = np.clip(v5 / v20 - 1, -1.0, 3.0)
                price_5d  = float(closes.iloc[-1] / closes.iloc[-6] - 1) if len(closes) >= 6 else 0.0
                # Only count surge as bullish when price is also rising
                vol_surge = raw_surge * (1.0 if price_5d > 0 else -0.5)

        # ── Factor 6: defensive sector bonus (only active in BEAR regime) ─────
        defensive = 1.0 if _sector(sym) in DEFENSIVE_SECTORS else 0.0

        scores[sym] = {
            "beta": beta, "price": last_price,
            "risk_adj_mom": risk_adj_mom, "rs_20": rs_20,
            "r2": r2, "proximity_hi": proximity_hi,
            "vol_surge": vol_surge, "defensive": defensive,
        }

    logger.info(f"Candidates: {len(candidates)} → passed filters: {n_filtered} → scored: {len(scores)}")

    if not scores:
        logger.error("No symbols passed all filters — falling back to config.SYMBOLS")
        return list(SYMBOLS)

    # ── Stage 2: composite score with regime-aware weights ────────────────────
    score_df = pd.DataFrame(scores).T
    composite = pd.Series(0.0, index=score_df.index)
    for factor, w in weights.items():
        if factor in score_df.columns:
            composite += w * _rank_pct(score_df[factor])
    score_df["composite"] = composite
    score_df = score_df.sort_values("composite", ascending=False)

    top10 = score_df.head(10)[["beta", "price", "risk_adj_mom", "rs_20", "r2", "proximity_hi", "composite"]]
    logger.info(f"Top 10 candidates ({regime} regime):\n{top10.round(3).to_string()}")

    # ── Stage 3: sector cap, then correlation deduplication ───────────────────
    # First apply sector cap on the ranked list
    sector_capped: list[str] = []
    sector_counts: dict[str, int] = defaultdict(int)
    for sym in score_df.index:
        sec = _sector(sym)
        if sector_counts[sec] < max_per_sector:
            sector_capped.append(sym)
            sector_counts[sec] += 1

    # Then deduplicate by pairwise correlation
    selected = _corr_dedup(sector_capped, close_df, max_symbols)

    # SPY always included — used for RS gate inside main.py
    if "SPY" not in selected:
        selected.append("SPY")

    return selected


def main(args: argparse.Namespace) -> None:
    os.makedirs("data", exist_ok=True)

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
        "date": date.today().isoformat(),
        "screened_at": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "count": len(symbols),
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Universe → {OUTPUT_PATH}: {len(symbols)} symbols")
    from collections import Counter
    for sec, n in sorted(Counter(_sector(s) for s in symbols).items()):
        logger.info(f"  {sec:<22} ({n}) {[s for s in symbols if _sector(s)==sec]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-market universe screener")
    parser.add_argument("--max",            type=int,   default=DEFAULT_MAX)
    parser.add_argument("--max-per-sector", type=int,   default=DEFAULT_MAX_SECTOR)
    parser.add_argument("--min-price",      type=float, default=DEFAULT_MIN_PRICE)
    parser.add_argument("--min-adv",        type=float, default=DEFAULT_MIN_ADV)
    main(parser.parse_args())
