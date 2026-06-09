import math
from loguru import logger

WEIGHTS = {
    "xgb":       0.30,
    "lstm":      0.30,
    "sentiment": 0.20,
    "regime":    0.20,
}

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
) -> tuple[str, float]:
    """
    Combine model signals into a final action and position fraction.

    Returns:
        (action, position_fraction)
        action: STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL
        position_fraction: fraction of portfolio to allocate (0.0 for HOLD/SELL)
    """
    regime_score = REGIME_SCORES.get(regime, 0.5)
    sentiment_norm = (sentiment_score + 1.0) / 2.0  # [-1, +1] → [0, 1]

    score = (
        WEIGHTS["xgb"]       * xgb_prob +
        WEIGHTS["lstm"]      * lstm_prob +
        WEIGHTS["sentiment"] * sentiment_norm +
        WEIGHTS["regime"]    * regime_score
    )

    if math.isnan(score):
        logger.warning(
            f"Ensemble score is NaN — inputs: xgb={xgb_prob}, lstm={lstm_prob}, "
            f"sentiment={sentiment_score}, regime={regime}. Defaulting to HOLD."
        )
        return "HOLD", 0.00

    logger.debug(
        f"Ensemble score={score:.3f} "
        f"(xgb={xgb_prob:.2f}, lstm={lstm_prob:.2f}, "
        f"sentiment={sentiment_score:.2f}, regime={regime})"
    )

    if score > 0.70:
        return "STRONG_BUY",  0.20
    elif score > 0.60:
        return "BUY",         0.12
    elif score < 0.30:
        return "STRONG_SELL", 0.00
    elif score < 0.40:
        return "SELL",        0.00
    else:
        return "HOLD",        0.00


def action_to_int(action: str) -> int:
    """Convert ensemble action to RL-compatible int: 0=Hold, 1=Buy, 2=Sell."""
    if "BUY" in action:
        return 1
    if "SELL" in action:
        return 2
    return 0
