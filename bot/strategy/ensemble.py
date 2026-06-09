import math
from loguru import logger

# Fix #10: macro_score added as a 5th signal.
# Weights redistributed so they still sum to 1.0.
WEIGHTS = {
    "xgb":       0.25,
    "lstm":      0.25,
    "sentiment": 0.15,
    "regime":    0.20,
    "macro":     0.15,
}

STRONG_BUY_THRESHOLD  = 0.70
BUY_THRESHOLD         = 0.60
SELL_THRESHOLD        = 0.40
STRONG_SELL_THRESHOLD = 0.30
STRONG_BUY_FRACTION   = 0.20
BUY_FRACTION          = 0.12
SELL_FRACTION         = 0.00
STRONG_SELL_FRACTION  = 0.00

REGIME_SCORES = {
    "TRENDING_UP":    1.0,
    "RANGING":        0.5,
    "HIGH_VOLATILITY":0.2,
    "TRENDING_DOWN":  0.0,
}


def ensemble_signal(
    xgb_prob: float,
    lstm_prob: float,
    sentiment_score: float,
    regime: str,
    macro_score: float = 0.5,
) -> tuple[str, float]:
    """
    Combine model signals into a final action and position fraction.

    Args:
        macro_score: macro environment score in [0, 1] — 0 bearish, 1 bullish, 0.5 neutral.
                     Defaults to 0.5 (neutral) for backtest / when FRED is unavailable.

    Returns:
        (action, position_fraction)
        action: STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL
        position_fraction: fraction of portfolio to allocate (0.0 for HOLD/SELL)
    """
    regime_score   = REGIME_SCORES.get(regime, 0.5)
    sentiment_norm = (sentiment_score + 1.0) / 2.0  # [-1, +1] → [0, 1]

    score = (
        WEIGHTS["xgb"]       * xgb_prob +
        WEIGHTS["lstm"]      * lstm_prob +
        WEIGHTS["sentiment"] * sentiment_norm +
        WEIGHTS["regime"]    * regime_score +
        WEIGHTS["macro"]     * macro_score
    )

    if math.isnan(score):
        logger.warning(
            f"Ensemble score is NaN — inputs: xgb={xgb_prob}, lstm={lstm_prob}, "
            f"sentiment={sentiment_score}, regime={regime}, macro={macro_score}. Defaulting to HOLD."
        )
        return "HOLD", 0.00

    logger.debug(
        f"Ensemble score={score:.3f} "
        f"(xgb={xgb_prob:.2f}, lstm={lstm_prob:.2f}, "
        f"sentiment={sentiment_score:.2f}, regime={regime}, macro={macro_score:.2f})"
    )

    if score > STRONG_BUY_THRESHOLD:
        return "STRONG_BUY",  STRONG_BUY_FRACTION
    elif score > BUY_THRESHOLD:
        return "BUY",         BUY_FRACTION
    elif score < STRONG_SELL_THRESHOLD:
        return "STRONG_SELL", STRONG_SELL_FRACTION
    elif score < SELL_THRESHOLD:
        return "SELL",        SELL_FRACTION
    else:
        return "HOLD",        0.00


def action_to_int(action: str) -> int:
    """Convert ensemble action to RL-compatible int: 0=Hold, 1=Buy, 2=Sell."""
    if "BUY" in action:
        return 1
    if "SELL" in action:
        return 2
    return 0
