from __future__ import annotations
import math
from loguru import logger

# Fix #10: macro_score added as a 5th signal.
# Weights redistributed so they still sum to 1.0.
# Regime is intentionally excluded from the score weights —
# it acts as a hard entry gate in main.py (ENTRY_REGIMES). Including it here
# would double-count it: once as a score component and again as a binary block.
WEIGHTS = {
    "xgb":       0.55,
    "lstm":      0.15,
    "sentiment": 0.15,
    "macro":     0.15,
}

STRONG_BUY_THRESHOLD  = 0.65
BUY_THRESHOLD         = 0.55
SELL_THRESHOLD        = 0.40
STRONG_SELL_THRESHOLD = 0.30
STRONG_BUY_FRACTION   = 0.10   # reduced from 0.20 — smaller positions enable more parallel trades
BUY_FRACTION          = 0.08   # reduced from 0.12 — tighter sizing lowers per-trade drawdown at higher frequency
SELL_FRACTION         = 0.00
STRONG_SELL_FRACTION  = 0.00

# LSTM outputs in this band carry no directional information — model is indeterminate.
# When LSTM is indeterminate, we reweight the ensemble so XGB carries the signal alone
# rather than letting a flat-lining model permanently veto every trade.
_LSTM_INDETERMINATE_LO = 0.45
_LSTM_INDETERMINATE_HI = 0.55


def _lstm_is_indeterminate(lstm_prob: float) -> bool:
    return _LSTM_INDETERMINATE_LO <= lstm_prob <= _LSTM_INDETERMINATE_HI


def _reweight_score(
    xgb_prob: float, lstm_prob: float,
    sentiment_norm: float, macro_score: float,
    lstm_indeterminate: bool,
) -> float:
    """Compute ensemble score, redistributing LSTM weight to XGB when LSTM is flat."""
    if lstm_indeterminate:
        # LSTM has no conviction — transfer its 0.35 weight to XGB so the signal
        # comes from the model that is actually producing a directional output.
        w_xgb  = WEIGHTS["xgb"] + WEIGHTS["lstm"]  # 0.70
        w_lstm = 0.0
    else:
        w_xgb  = WEIGHTS["xgb"]
        w_lstm = WEIGHTS["lstm"]
    return (
        w_xgb       * xgb_prob +
        w_lstm      * lstm_prob +
        WEIGHTS["sentiment"] * sentiment_norm +
        WEIGHTS["macro"]     * macro_score
    )


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
    sentiment_norm     = (sentiment_score + 1.0) / 2.0  # [-1, +1] → [0, 1]
    lstm_indeterminate = _lstm_is_indeterminate(lstm_prob)

    score = _reweight_score(xgb_prob, lstm_prob, sentiment_norm, macro_score, lstm_indeterminate)

    if math.isnan(score):
        logger.warning(
            f"Ensemble score is NaN — inputs: xgb={xgb_prob}, lstm={lstm_prob}, "
            f"sentiment={sentiment_score}, regime={regime}, macro={macro_score}. Defaulting to HOLD."
        )
        return "HOLD", 0.00

    if lstm_indeterminate:
        logger.debug(
            f"Ensemble score={score:.3f} (xgb={xgb_prob:.2f}, lstm={lstm_prob:.2f} [indeterminate — "
            f"weight transferred to xgb], sentiment={sentiment_score:.2f}, regime={regime}, macro={macro_score:.2f})"
        )
    else:
        logger.debug(
            f"Ensemble score={score:.3f} "
            f"(xgb={xgb_prob:.2f}, lstm={lstm_prob:.2f}, "
            f"sentiment={sentiment_score:.2f}, regime={regime}, macro={macro_score:.2f})"
        )

    if score > STRONG_BUY_THRESHOLD:
        # ML agreement gate: only blocks when LSTM has a clear directional opinion
        # that disagrees with XGB. A flat-lining LSTM (indeterminate) is not a veto.
        if not lstm_indeterminate and lstm_prob < 0.50:
            logger.debug(
                f"Ensemble: STRONG_BUY suppressed — LSTM disagrees "
                f"(xgb={xgb_prob:.3f}, lstm={lstm_prob:.3f})"
            )
            return "HOLD", 0.00
        if xgb_prob < 0.50:
            logger.debug(
                f"Ensemble: STRONG_BUY suppressed — XGB below threshold "
                f"(xgb={xgb_prob:.3f})"
            )
            return "HOLD", 0.00
        return "STRONG_BUY",  STRONG_BUY_FRACTION
    elif score > BUY_THRESHOLD:
        if not lstm_indeterminate and lstm_prob < 0.50:
            logger.debug(
                f"Ensemble: BUY suppressed — LSTM disagrees "
                f"(xgb={xgb_prob:.3f}, lstm={lstm_prob:.3f})"
            )
            return "HOLD", 0.00
        if xgb_prob < 0.50:
            logger.debug(
                f"Ensemble: BUY suppressed — XGB below threshold "
                f"(xgb={xgb_prob:.3f})"
            )
            return "HOLD", 0.00
        return "BUY",         BUY_FRACTION
    elif score < STRONG_SELL_THRESHOLD:
        return "STRONG_SELL", STRONG_SELL_FRACTION
    elif score < SELL_THRESHOLD:
        return "SELL",        SELL_FRACTION
    else:
        return "HOLD",        0.00


def action_to_int(action: str) -> int:
    """Convert ensemble action string to int: 0=Hold, 1=Buy, 2=Sell."""
    if "BUY" in action:
        return 1
    if "SELL" in action:
        return 2
    return 0
