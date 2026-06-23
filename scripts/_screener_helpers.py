"""Screener factor-computation helpers, extracted from screen_universe.py."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from loguru import logger
from scipy import stats as scipy_stats

from config import SECTOR_MAP

CORR_THRESHOLD        = 0.85
ANALYST_LOOKBACK_DAYS = 5
_FINNHUB_BASE         = "https://finnhub.io/api/v1"

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
            "etf_momentum": 0.15,
        }
    # BEAR — rotate toward quality + RS, de-emphasise raw momentum
    return {
        "risk_adj_mom": 0.10,
        "rs_20":        0.25,
        "r2":           0.20,
        "proximity_hi": 0.05,
        "vol_surge":    0.05,
        "etf_momentum": 0.15,
        "defensive":    0.20,
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
    """Return symbols whose next earnings date is within ±window_days of today."""
    from datetime import date
    today = date.today()
    blocked: set[str] = set()
    for sym in symbols:
        try:
            cal = yf.Ticker(sym).calendar
            if cal is None or cal.empty:
                continue
            if "Earnings Date" in cal.index:
                ed = cal.loc["Earnings Date"]
            elif "Earnings Date" in cal.columns:
                ed = cal["Earnings Date"].iloc[0]
            else:
                continue
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
    """Average absolute overnight gap as a fraction of the prior close."""
    aligned = pd.concat([closes.shift(1), opens], axis=1).dropna()
    aligned.columns = ["prev_close", "open"]
    aligned = aligned[aligned["prev_close"] > 0]
    if len(aligned) < 20:
        return 0.0
    gaps = ((aligned["open"] - aligned["prev_close"]) / aligned["prev_close"]).abs()
    return float(gaps.tail(20).mean())


def _analyst_signal(symbol: str, token: str, lookback_days: int = ANALYST_LOOKBACK_DAYS) -> float:
    """Return a signal in [-1, +1] based on recent analyst upgrade/downgrade activity."""
    if not token:
        return 0.0
    try:
        import scripts.screen_universe as _parent
        r = _parent.requests.get(
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
    """Return 1.0 if the symbol's sector ETF is above its 20-day SMA, else 0.0."""
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
