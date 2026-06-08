import os
from loguru import logger

FRED_API_KEY = os.getenv("FRED_API_KEY", "")


def get_macro_signal() -> float:
    """
    Returns a macro market score in [0, 1].
    0 = very bearish macro conditions, 1 = very bullish, 0.5 = neutral.
    Returns 0.5 if FRED_API_KEY is not set.
    """
    if not FRED_API_KEY:
        return 0.5

    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)

        yield_curve = float(fred.get_series("T10Y2Y").dropna().iloc[-1])
        vix = float(fred.get_series("VIXCLS").dropna().iloc[-1])
        fed_rate = float(fred.get_series("FEDFUNDS").dropna().iloc[-1])

        score = 0.5

        # Inverted yield curve signals recession
        if yield_curve < 0:
            score -= 0.15
        elif yield_curve > 1.5:
            score += 0.10

        # High VIX = fear = reduce exposure
        if vix > 30:
            score -= 0.20
        elif vix < 15:
            score += 0.10

        # Very high fed rate = headwind for growth stocks
        if fed_rate > 5.0:
            score -= 0.10
        elif fed_rate < 2.0:
            score += 0.05

        score = max(0.0, min(1.0, score))
        logger.info(f"Macro: yield_curve={yield_curve:.2f}, vix={vix:.1f}, fed_rate={fed_rate:.2f} → score={score:.2f}")
        return score

    except Exception as e:
        logger.warning(f"FRED macro signal failed: {e}")
        return 0.5


def get_macro_position_cap() -> float:
    """
    Returns a position size multiplier based on macro risk.
    1.0 = normal, 0.5 = halve all position sizes in risky conditions.
    """
    if not FRED_API_KEY:
        return 1.0

    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        yield_curve = float(fred.get_series("T10Y2Y").dropna().iloc[-1])
        vix = float(fred.get_series("VIXCLS").dropna().iloc[-1])
        if yield_curve < 0 or vix > 30:
            logger.warning("Macro risk elevated — capping positions at 50%.")
            return 0.5
        return 1.0
    except Exception:
        return 1.0
