import os
import math
import time
import threading
from loguru import logger


def _sigmoid(x: float, center: float, scale: float) -> float:
    """Logistic sigmoid centered at `center`, transitions over ±`scale`."""
    return 1.0 / (1.0 + math.exp(-(x - center) / scale))

FRED_API_KEY = os.getenv("FRED_API_KEY", "")

_CACHE_LOCK = threading.Lock()
_MACRO_CACHE: dict = {}      # {"score": float, "cap": float}
_MACRO_TS: float   = 0.0
_MACRO_TTL: float  = 4 * 3600  # FRED data is daily — re-fetch every 4 hours


def _fetch_macro_raw() -> dict:
    """Fetch yield curve, VIX, and fed rate from FRED. Returns raw values dict."""
    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)
    return {
        "yield_curve": float(fred.get_series("T10Y2Y").dropna().iloc[-1]),
        "vix":         float(fred.get_series("VIXCLS").dropna().iloc[-1]),
        "fed_rate":    float(fred.get_series("FEDFUNDS").dropna().iloc[-1]),
    }


def _compute_from_raw(raw: dict) -> dict:
    yield_curve = raw["yield_curve"]
    vix         = raw["vix"]
    fed_rate    = raw["fed_rate"]

    # Continuous sigmoid scoring removes hard-threshold discontinuities.
    # VIX: low fear → bullish. Center=20, scale=5 gives smooth 15→25 transition.
    vix_score  = 1.0 - _sigmoid(vix, center=20.0, scale=5.0)
    # Yield curve: positive (normal) → bullish, negative (inverted) → bearish.
    yc_score   = _sigmoid(yield_curve, center=0.0, scale=0.5)
    # Fed rate: high rates pressure multiples. Center=3.5%, scale=1.5.
    rate_score = 1.0 - _sigmoid(fed_rate, center=3.5, scale=1.5)

    score = max(0.0, min(1.0, 0.50 * vix_score + 0.30 * yc_score + 0.20 * rate_score))

    # Cap position sizing when macro stress is elevated (same logic, same threshold)
    cap = 0.5 if (yield_curve < 0 or vix > 30) else 1.0

    logger.info(
        f"Macro: yield_curve={yield_curve:.2f}, vix={vix:.1f}, "
        f"fed_rate={fed_rate:.2f} → score={score:.2f}, cap={cap:.1f}"
    )
    return {"score": score, "cap": cap}


def _get_cached() -> dict:
    """Returns cached macro dict, refreshing if older than TTL."""
    global _MACRO_CACHE, _MACRO_TS
    now = time.time()
    with _CACHE_LOCK:
        if _MACRO_CACHE and (now - _MACRO_TS) < _MACRO_TTL:
            return _MACRO_CACHE
        try:
            raw = _fetch_macro_raw()
            _MACRO_CACHE = _compute_from_raw(raw)
        except Exception as e:
            logger.warning(f"FRED macro fetch failed: {e}")
            if not _MACRO_CACHE:
                _MACRO_CACHE = {"score": 0.5, "cap": 1.0}
        _MACRO_TS = now
        return _MACRO_CACHE


def get_macro_signal() -> float:
    """
    Returns a macro market score in [0, 1].
    0 = very bearish, 1 = very bullish, 0.5 = neutral.
    Cached for 4 hours — FRED data is daily.
    """
    if not FRED_API_KEY:
        return 0.5
    return _get_cached()["score"]


def get_macro_position_cap() -> float:
    """
    Returns a position size multiplier: 1.0 normal, 0.5 when macro risk is elevated.
    Shares the same 4-hour cache as get_macro_signal() — no extra FRED calls.
    """
    if not FRED_API_KEY:
        return 1.0
    return _get_cached()["cap"]
