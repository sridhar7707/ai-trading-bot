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
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from loguru import logger
from scipy import stats as scipy_stats

from config import SECTOR_MAP, SYMBOLS

OUTPUT_PATH          = "data/universe_today.json"
DEFAULT_MAX          = 25
DEFAULT_MAX_SECTOR   = 3
DEFAULT_MIN_PRICE    = 10.0
DEFAULT_MIN_ADV      = 5_000_000
DEFAULT_MIN_HISTORY  = 252   # require 1 full year — filters IPOs where LSTM has no sequence
BETA_MIN             = 0.6   # too-slow stocks don't move enough for intraday trading
BETA_MAX             = 2.0   # too-wild stocks blow stops before the edge plays out
CORR_THRESHOLD       = 0.85  # pairwise correlation above which one position is redundant
MAX_AVG_OVERNIGHT_GAP = 0.03 # skip chronic gappers — limit orders fill far from signal price
ANALYST_LOOKBACK_DAYS = 5    # only count upgrades/downgrades from the last N days
_FINNHUB_BASE         = "https://finnhub.io/api/v1"

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
            "risk_adj_mom": 0.30,
            "rs_20":        0.20,
            "r2":           0.15,
            "proximity_hi": 0.10,
            "vol_surge":    0.10,
            "etf_momentum": 0.15,  # sector tailwind — avoid picking winners in losing sectors
        }
    # BEAR — rotate toward quality + RS, de-emphasise raw momentum
    return {
        "risk_adj_mom": 0.10,
        "rs_20":        0.25,
        "r2":           0.20,
        "proximity_hi": 0.05,
        "vol_surge":    0.05,
        "etf_momentum": 0.15,  # sector tailwind matters even more in bear market
        "defensive":    0.20,  # bonus for defensive sectors
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


def _earnings_blackout_set(symbols: list[str], window_days: int = 2) -> set[str]:
    """Return symbols whose next earnings date is within ±window_days of today.

    Uses yfinance calendar.  Silently skips symbols where data is unavailable
    (better to include than to exclude incorrectly).
    """
    today = date.today()
    blocked: set[str] = set()
    for sym in symbols:
        try:
            cal = yf.Ticker(sym).calendar
            if cal is None or cal.empty:
                continue
            # calendar index may be dates or strings depending on yfinance version
            if "Earnings Date" in cal.index:
                ed = cal.loc["Earnings Date"]
            elif "Earnings Date" in cal.columns:
                ed = cal["Earnings Date"].iloc[0]
            else:
                continue
            # ed may be a Timestamp, list, or scalar
            if hasattr(ed, "__iter__") and not isinstance(ed, str):
                dates = [pd.to_datetime(d).date() for d in ed]
            else:
                dates = [pd.to_datetime(ed).date()]
            for d in dates:
                if abs((d - today).days) <= window_days:
                    logger.info(f"Earnings blackout: {sym} (earnings {d})")
                    blocked.add(sym)
                    break
        except Exception:
            pass
    return blocked


def _avg_overnight_gap(closes: pd.Series, opens: pd.Series) -> float:
    """Average absolute overnight gap as a fraction of the prior close.

    A stock with avg gap > 3% regularly opens far from the prior bar's close,
    making limit orders unreliable — better to skip it.
    """
    aligned = pd.concat([closes.shift(1), opens], axis=1).dropna()
    aligned.columns = ["prev_close", "open"]
    aligned = aligned[aligned["prev_close"] > 0]
    if len(aligned) < 20:
        return 0.0
    gaps = ((aligned["open"] - aligned["prev_close"]) / aligned["prev_close"]).abs()
    return float(gaps.tail(20).mean())


def _analyst_signal(symbol: str, token: str, lookback_days: int = ANALYST_LOOKBACK_DAYS) -> float:
    """Return a signal in [-1, +1] based on recent analyst upgrade/downgrade activity.

    +1  = strong upgrade (buy/outperform initiation or upgrade from neutral)
     0  = no recent action or conflicting signals
    -1  = downgrade (sell/underperform or reduction from buy)

    Only checks activity from the last `lookback_days` days to stay timely.
    Silently returns 0 on any API failure.
    """
    if not token:
        return 0.0
    try:
        r = requests.get(
            f"{_FINNHUB_BASE}/stock/upgrade-downgrade",
            params={"symbol": symbol, "token": token},
            timeout=5,
        )
        if r.status_code != 200:
            return 0.0
        items = r.json()
        if not items:
            return 0.0
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp()
        recent = [i for i in items if i.get("gradeTime", 0) >= cutoff]
        if not recent:
            return 0.0
        _POSITIVE = {"buy", "strong buy", "outperform", "overweight", "accumulate", "positive"}
        _NEGATIVE = {"sell", "strong sell", "underperform", "underweight", "reduce", "negative"}
        score = 0.0
        for item in recent:
            action = (item.get("action") or "").lower()
            to_g   = (item.get("toGrade") or "").lower()
            if action in ("up", "init", "reit") and to_g in _POSITIVE:
                score += 1.0
            elif action == "down" or to_g in _NEGATIVE:
                score -= 1.0
        return float(np.clip(score / max(len(recent), 1), -1.0, 1.0))
    except Exception:
        return 0.0


def _sector_etf_momentum(sector: str, close_df: pd.DataFrame) -> float:
    """Return 1.0 if the symbol's sector ETF is above its 20-day SMA, else 0.0.

    Uses already-downloaded close_df so no extra API calls are needed.
    Falls back to 0.5 (neutral) when the sector ETF isn't in the download.
    """
    _SECTOR_ETF: dict[str, str] = {
        "Technology":        "XLK",
        "Comm_Services":     "XLC",
        "Consumer_Disc":     "XLY",
        "Consumer_Staples":  "XLP",
        "Financials":        "XLF",
        "Healthcare":        "XLV",
        "Energy":            "XLE",
        "Industrials":       "XLI",
        "Materials":         "XLB",
        "Utilities":         "XLU",
        "Real_Estate":       "XLRE",
        "Broad_ETF":         "SPY",
        "Commodities":       "GLD",
        "Bonds":             "TLT",
    }
    etf = _SECTOR_ETF.get(sector)
    if not etf or etf not in close_df.columns:
        return 0.5
    closes = close_df[etf].dropna()
    if len(closes) < 20:
        return 0.5
    sma20 = float(closes.tail(20).mean())
    return 1.0 if float(closes.iloc[-1]) > sma20 else 0.0


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
        return list(SYMBOLS)

    close_df  = raw["Close"]
    open_df   = raw.get("Open")
    volume_df = raw.get("Volume")

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

    for sym in candidates:
        if sym not in close_df.columns:
            continue
        closes = close_df[sym].dropna()

        # IPO age / history requirement — LSTM needs 252 bars for a full year sequence
        if len(closes) < min_history:
            continue

        last_price = float(closes.iloc[-1])
        if last_price < min_price:
            continue

        # Earnings blackout ±2 days
        if sym in earnings_blocked:
            continue

        # ADV filter (20-day average dollar volume)
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

        # Overnight gap filter — skip chronic gappers (limit orders fill far from signal)
        if open_df is not None and sym in open_df.columns:
            opens = open_df[sym].dropna()
            avg_gap = _avg_overnight_gap(closes.reindex(opens.index), opens)
            if avg_gap > MAX_AVG_OVERNIGHT_GAP:
                logger.debug(f"{sym} skipped — avg overnight gap {avg_gap:.1%} > {MAX_AVG_OVERNIGHT_GAP:.0%}")
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

        # ── Factor 6: sector ETF momentum alignment ───────────────────────────
        sec = _sector(sym)
        etf_momentum = _sector_etf_momentum(sec, close_df)

        # ── Factor 7: defensive sector bonus (only active in BEAR regime) ─────
        defensive = 1.0 if sec in DEFENSIVE_SECTORS else 0.0

        scores[sym] = {
            "beta": beta, "price": last_price,
            "risk_adj_mom": risk_adj_mom, "rs_20": rs_20,
            "r2": r2, "proximity_hi": proximity_hi,
            "vol_surge": vol_surge, "etf_momentum": etf_momentum,
            "defensive": defensive, "analyst_signal": 0.0,  # filled in Stage 2b
        }

    logger.info(f"Candidates: {len(candidates)} → passed filters: {n_filtered} → scored: {len(scores)}")

    if not scores:
        logger.error("No symbols passed all filters — falling back to config.SYMBOLS")
        return list(SYMBOLS)

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
                    "proximity_hi", "etf_momentum", "analyst_signal", "composite"]
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

    payload = {
        "date": date.today().isoformat(),
        "screened_at": screened_at,
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
    parser.add_argument("--min-history",    type=int,   default=DEFAULT_MIN_HISTORY)
    main(parser.parse_args())
