"""
Pre-market universe screener — hedge fund multi-factor approach.

Two-stage pipeline:
  Stage 1 — Hard filters (price, ADV, above-50-SMA, beta range,
             earnings blackout ±2 days, overnight gap history)
  Stage 2 — Factor scoring with regime-adaptive weights + analyst signal

Factors (computed from 1-year daily price data + optional Finnhub):
  • Risk-adjusted 60-day momentum  (return / 20d volatility)
  • 20-day relative strength vs SPY
  • Trend quality R²               (OLS fit over last 20 bars)
  • 52-week high proximity         (O'Neil breakout signal)
  • Price-confirmed volume surge   (volume only positive when price rising)
  • Sector ETF momentum            (is the stock's sector ETF above its 20-SMA?)
  • Analyst signal                 (Finnhub upgrade/downgrade, top-50 only)
  • Market beta                    (filter: 0.6–2.0 for swing/day trading)

Regime-adaptive weights:
  Bull (SPY > 50-SMA): emphasise momentum
  Bear (SPY < 50-SMA): emphasise RS + trend quality + defensive sectors

Final pass — correlation deduplication:
  Replace any pair with pairwise correlation >CORRELATION_THRESHOLD with the next-best
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
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from config import SYMBOLS

OUTPUT_PATH          = "data/universe_today.json"
DEFAULT_MAX          = 25
DEFAULT_MAX_SECTOR   = 3
DEFAULT_MIN_PRICE    = 10.0
DEFAULT_MIN_ADV      = 5_000_000
DEFAULT_MIN_HISTORY  = 245   # ~1 year minus holiday variance — filters IPOs where LSTM has no sequence
BETA_MIN             = 0.6   # too-slow stocks don't move enough for intraday trading
BETA_MAX             = 2.0   # too-wild stocks blow stops before the edge plays out
MAX_AVG_OVERNIGHT_GAP = 0.03 # skip chronic gappers — limit orders fill far from signal price

DEFENSIVE_SECTORS = {"Consumer_Staples", "Healthcare", "Utilities", "Bonds", "Commodities"}

from scripts._screener_helpers import (  # noqa: E402
    FULL_SECTOR_MAP,
    _sector, _rank_pct, _trend_r2, _compute_beta, _detect_regime,
    _factor_weights, _corr_dedup, _earnings_blackout_set,
    _avg_overnight_gap, _analyst_signal, _sector_etf_momentum,
    _pead_score, _eps_quality_score,
)

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



def screen(
    max_symbols: int = DEFAULT_MAX,
    max_per_sector: int = DEFAULT_MAX_SECTOR,
    min_price: float = DEFAULT_MIN_PRICE,
    min_adv: float = DEFAULT_MIN_ADV,
    min_history: int = DEFAULT_MIN_HISTORY,
) -> tuple[list[str], pd.DataFrame, str]:
    finnhub_token = os.environ.get("FINNHUB_API_KEY", "")

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
        return list(SYMBOLS), pd.DataFrame(), "UNKNOWN"

    close_df  = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    try:
        open_df = raw["Open"]
    except (KeyError, Exception):
        open_df = None
    try:
        volume_df = raw["Volume"]
    except (KeyError, Exception):
        volume_df = None

    logger.info(f"yfinance raw columns (first 5): {list(raw.columns[:5])} | close_df type: {type(close_df).__name__} | close_df shape: {close_df.shape if hasattr(close_df, 'shape') else 'N/A'} | volume_df: {'OK' if volume_df is not None else 'MISSING'}")

    spy_closes = close_df.get("SPY", pd.Series(dtype=float)).dropna()
    regime     = _detect_regime(spy_closes)
    weights    = _factor_weights(regime)
    logger.info(f"Market regime: {regime} | weights: {weights}")

    spy_rets = spy_closes.pct_change().dropna()

    # Pre-compute earnings blackout (one API round-trip per symbol — do it in
    # bulk before the main loop so we only call yfinance.Ticker once per symbol)
    logger.info("Checking earnings calendar for blackout dates...")
    earnings_blocked = _earnings_blackout_set(candidates)
    if earnings_blocked:
        logger.info(f"Earnings blackout: {sorted(earnings_blocked)}")

    # ── Stage 1: hard filters + factor computation ────────────────────────────
    scores: dict[str, dict] = {}
    n_filtered = 0
    _skip = {"no_data": 0, "history": 0, "price": 0, "earnings": 0,
             "volume": 0, "adv": 0, "sma50": 0, "gap": 0, "beta": 0}

    for sym in candidates:
        if sym not in close_df.columns:
            _skip["no_data"] += 1
            continue
        closes = close_df[sym].dropna()

        # IPO age / history requirement — LSTM needs 252 bars for a full year sequence
        if len(closes) < min_history:
            _skip["history"] += 1
            continue

        last_price = float(closes.iloc[-1])
        if last_price < min_price:
            _skip["price"] += 1
            continue

        # Earnings blackout ±2 days
        if sym in earnings_blocked:
            _skip["earnings"] += 1
            continue

        # ADV filter (20-day average dollar volume)
        if volume_df is None or sym not in volume_df.columns:
            _skip["volume"] += 1
            continue
        vols = volume_df[sym].dropna()
        adv  = float((closes.reindex(vols.index) * vols).tail(20).mean())
        if adv < min_adv:
            _skip["adv"] += 1
            continue

        # Above 50-SMA — no falling knives
        sma50 = float(closes.tail(50).mean())
        if last_price < sma50:
            _skip["sma50"] += 1
            continue

        # Overnight gap filter — skip chronic gappers (limit orders fill far from signal)
        if open_df is not None and sym in open_df.columns:
            opens = open_df[sym].dropna()
            avg_gap = _avg_overnight_gap(closes.reindex(opens.index), opens)
            if avg_gap > MAX_AVG_OVERNIGHT_GAP:
                logger.debug(f"{sym} skipped — avg overnight gap {avg_gap:.1%} > {MAX_AVG_OVERNIGHT_GAP:.0%}")
                _skip["gap"] += 1
                continue

        # Beta filter — too slow or too wild
        sym_rets = closes.pct_change().dropna()
        beta = _compute_beta(sym_rets, spy_rets)
        if beta < BETA_MIN or beta > BETA_MAX:
            _skip["beta"] += 1
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

        # ── Factor 6: sector ETF momentum alignment ───────────────────────────
        sec = _sector(sym)
        etf_momentum = _sector_etf_momentum(sec, close_df)

        # ── Factor 7: defensive sector bonus (only active in BEAR regime) ─────
        defensive = 1.0 if sec in DEFENSIVE_SECTORS else 0.0

        # ── Factor 8: post-earnings drift (PEAD) ─────────────────────────────
        # Scores 0.0–1.0 when stock beat EPS estimates 2–10 days ago.
        # Throttled inside _pead_score; ETFs always return 0.0.
        pead = _pead_score(sym)
        if pead > 0:
            logger.info(f"PEAD signal: {sym} earnings beat score={pead:.2f}")

        eps_quality = _eps_quality_score(sym)
        if eps_quality < 0.4:
            logger.debug(f"EPS quality low: {sym} score={eps_quality:.2f} (declining earnings)")

        scores[sym] = {
            "beta": beta, "price": last_price,
            "risk_adj_mom": risk_adj_mom, "rs_20": rs_20,
            "r2": r2, "proximity_hi": proximity_hi,
            "vol_surge": vol_surge, "etf_momentum": etf_momentum,
            "defensive": defensive, "analyst_signal": 0.0,  # filled in Stage 2b
            "pead": pead, "eps_quality": eps_quality,
        }

    logger.info(
        f"Candidates: {len(candidates)} → passed filters: {n_filtered} → scored: {len(scores)} | "
        f"skipped: no_data={_skip['no_data']} history={_skip['history']} price={_skip['price']} "
        f"earnings={_skip['earnings']} volume={_skip['volume']} adv={_skip['adv']} "
        f"sma50={_skip['sma50']} gap={_skip['gap']} beta={_skip['beta']}"
    )

    if not scores:
        logger.error("No symbols passed all filters — falling back to config.SYMBOLS")
        return list(SYMBOLS), pd.DataFrame(), "UNKNOWN"

    # ── Stage 2a: initial rank without analyst signal ─────────────────────────
    score_df  = pd.DataFrame(scores).T
    composite = pd.Series(0.0, index=score_df.index)
    base_weights = {k: v for k, v in weights.items() if k != "analyst_signal"}
    for factor, w in base_weights.items():
        if factor in score_df.columns:
            composite += w * _rank_pct(score_df[factor])
    score_df["composite"] = composite
    score_df = score_df.sort_values("composite", ascending=False)

    # ── Stage 2b: enrich top-50 with Finnhub analyst signal ──────────────────
    # Only the top-50 so Finnhub calls stay under ~50 (avoids rate-limit issues).
    top50 = list(score_df.head(50).index)
    if finnhub_token and top50:
        logger.info(f"Fetching Finnhub analyst ratings for top {len(top50)} candidates...")
        for sym in top50:
            sig = _analyst_signal(sym, finnhub_token)
            scores[sym]["analyst_signal"] = sig
            time.sleep(0.05)  # ~20 req/s — comfortably under free-tier 30/s limit

        # Re-score with analyst signal included
        score_df = pd.DataFrame(scores).T
        composite = pd.Series(0.0, index=score_df.index)
        full_weights = dict(weights)
        # Redistribute 10% weight to analyst signal from momentum + rs_20
        if "analyst_signal" not in full_weights:
            full_weights["analyst_signal"] = 0.10
            for adj in ("risk_adj_mom", "rs_20"):
                if adj in full_weights:
                    full_weights[adj] = max(0, full_weights[adj] - 0.05)
        for factor, w in full_weights.items():
            if factor in score_df.columns:
                composite += w * _rank_pct(score_df[factor])
        score_df["composite"] = composite
        score_df = score_df.sort_values("composite", ascending=False)
        logger.info("Analyst signal incorporated into final ranking.")
    else:
        if not finnhub_token:
            logger.info("FINNHUB_API_KEY not set — analyst signal skipped (set secret to enable)")

    display_cols = ["beta", "price", "risk_adj_mom", "rs_20", "r2",
                    "proximity_hi", "etf_momentum", "pead", "eps_quality", "analyst_signal", "composite"]
    top10_cols = [c for c in display_cols if c in score_df.columns]
    logger.info(f"Top 10 candidates ({regime} regime):\n"
                f"{score_df.head(10)[top10_cols].round(3).to_string()}")

    # ── Stage 3: sector cap, then correlation deduplication ───────────────────
    sector_capped: list[str] = []
    sector_counts: dict[str, int] = defaultdict(int)
    for sym in score_df.index:
        sec = _sector(sym)
        if sector_counts[sec] < max_per_sector:
            sector_capped.append(sym)
            sector_counts[sec] += 1

    selected = _corr_dedup(sector_capped, close_df, max_symbols)

    # SPY always included — used for RS gate inside main.py
    if "SPY" not in selected:
        selected.append("SPY")

    return selected, score_df, regime


_TRADE_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trades.db")


def _save_screener_to_db(
    selected: list[str],
    score_df: pd.DataFrame,
    regime: str,
    screened_at: str,
) -> None:
    """Persist screener picks + factor scores to trades.db for dashboard display."""
    try:
        con = sqlite3.connect(_TRADE_DB)
        con.execute("""
            CREATE TABLE IF NOT EXISTS screener_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                screened_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                rank INTEGER,
                composite_score REAL,
                analyst_signal REAL,
                etf_momentum REAL,
                regime TEXT,
                sector TEXT
            )
        """)
        # Prune runs older than 7 days
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        con.execute("DELETE FROM screener_log WHERE screened_at < ?", (cutoff,))
        # Write selected symbols (excluding always-added SPY) in rank order
        rows = []
        rank = 1
        for sym in selected:
            if sym == "SPY":
                continue
            row = score_df.loc[sym] if sym in score_df.index else None
            rows.append((
                screened_at, sym, rank,
                round(float(row["composite"]), 4) if row is not None and "composite" in score_df.columns else None,
                round(float(row["analyst_signal"]), 4) if row is not None and "analyst_signal" in score_df.columns else 0.0,
                round(float(row["etf_momentum"]), 4) if row is not None and "etf_momentum" in score_df.columns else None,
                regime,
                _sector(sym),
            ))
            rank += 1
        con.executemany(
            "INSERT INTO screener_log (screened_at,symbol,rank,composite_score,analyst_signal,etf_momentum,regime,sector) "
            "VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        con.commit()
        con.close()
        logger.info(f"screener_log: saved {len(rows)} symbols to trades.db")
    except Exception as exc:
        logger.warning(f"screener_log write failed (non-fatal): {exc}")


def main(args: argparse.Namespace) -> None:
    os.makedirs("data", exist_ok=True)
    screened_at = datetime.now(timezone.utc).isoformat()

    try:
        symbols, score_df, regime = screen(
            max_symbols=args.max,
            max_per_sector=args.max_per_sector,
            min_price=args.min_price,
            min_adv=args.min_adv,
            min_history=args.min_history,
        )
    except Exception as exc:
        logger.error(f"Screener failed: {exc} — falling back to config.SYMBOLS")
        symbols, score_df, regime = list(SYMBOLS), pd.DataFrame(), "BULL"

    _save_screener_to_db(symbols, score_df, regime, screened_at)

    # Build per-symbol pick records so the trading bot can import them into its
    # own screener_log table.  universe_today.json crosses the GitHub Actions job
    # boundary via cache; the premarket runner's trades.db does NOT — so this is
    # the only path for screener data to reach the bot's DB and the dashboard.
    picks: list[dict] = []
    rank = 1
    for sym in symbols:
        if sym == "SPY":
            continue
        row = score_df.loc[sym] if (not score_df.empty and sym in score_df.index) else None
        picks.append({
            "symbol":          sym,
            "rank":            rank,
            "composite_score": round(float(row["composite"]), 4)      if row is not None and "composite"      in score_df.columns else None,
            "analyst_signal":  round(float(row["analyst_signal"]), 4) if row is not None and "analyst_signal" in score_df.columns else 0.0,
            "etf_momentum":    round(float(row["etf_momentum"]), 4)   if row is not None and "etf_momentum"   in score_df.columns else None,
            "regime":          regime,
            "sector":          _sector(sym),
        })
        rank += 1

    payload = {
        "date":        date.today().isoformat(),
        "screened_at": screened_at,
        "regime":      regime,
        "symbols":     symbols,
        "count":       len(symbols),
        "picks":       picks,      # full factor data — imported by bot into screener_log
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
    parser.add_argument("--min-history",    type=int,   default=DEFAULT_MIN_HISTORY)
    main(parser.parse_args())
