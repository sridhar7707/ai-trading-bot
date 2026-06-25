"""High-confidence 7-layer gate for user-facing signals.

The bot's internal BUY threshold (ensemble score > 0.55) is permissive enough
to generate trades while learning. User-facing signals need a higher bar:
all 7 checks must pass independently before a signal is published.
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

# Thresholds — stricter than the bot's internal BUY gates
_HC_XGB_MIN     = 0.65   # vs bot's 0.50 secondary gate
_HC_LSTM_MIN    = 0.55   # LSTM must have directional conviction
_HC_VOL_RATIO   = 1.5    # volume 1.5× 20-day average
_HC_MACRO_MIN   = 0.50   # macro at least neutral
_STOP_PCT       = 0.04   # 4% stop distance (matches bot's STOP_LOSS_PCT)
_RR_MIN         = 2.0    # minimum risk/reward ratio
_BREAKOUT_ZONE  = 0.05   # within 5% of 52-week high = breakout candidate
_PULLBACK_ZONE  = 0.03   # within 3% above 20-day MA = pullback candidate


def check_signal_gate(
    symbol: str,
    xgb_prob: float,
    lstm_prob: float,
    macro_score: float,
    bars_daily: pd.DataFrame,
    volume_ratio: float,
    spy_today_pct: float,
) -> tuple[bool, dict]:
    """
    7-layer high-confidence gate for user-facing signals.

    All 7 checks must pass. Returns (passed, meta) where meta is always
    populated — use it for logging and DB recording regardless of outcome.
    """
    if bars_daily is None or bars_daily.empty or len(bars_daily) < 30:
        return False, {"reason": "insufficient bar history"}

    close = float(bars_daily["close"].iloc[-1])
    entry = close
    stop  = round(entry * (1 - _STOP_PCT), 2)

    reasons: list[str] = []

    # ① XGB must show strong conviction (raises bar from ensemble's 0.50 gate)
    if xgb_prob < _HC_XGB_MIN:
        reasons.append(f"XGB {xgb_prob:.2f} < {_HC_XGB_MIN}")

    # ② LSTM must have directional conviction — not in the flat [0.45–0.55] band
    lstm_indeterminate = 0.45 <= lstm_prob <= 0.55
    if lstm_indeterminate or lstm_prob < _HC_LSTM_MIN:
        reasons.append(f"LSTM {lstm_prob:.2f} indeterminate or below {_HC_LSTM_MIN}")

    # ③ Volume elevated — confirms institutional participation, not retail noise
    if volume_ratio < _HC_VOL_RATIO:
        reasons.append(f"volume ratio {volume_ratio:.2f} < {_HC_VOL_RATIO}")

    # ④ Technical setup: breaking out near 52-week high OR pulling back to 20-day MA
    high_52w = float(bars_daily["high"].max())
    sma_20   = float(bars_daily["sma_20"].iloc[-1]) if "sma_20" in bars_daily.columns else close
    pct_from_52wh = (high_52w - close) / high_52w if high_52w > 0 else 1.0
    pct_from_sma  = (close - sma_20) / sma_20 if sma_20 > 0 else 1.0

    is_breakout = pct_from_52wh <= _BREAKOUT_ZONE
    is_pullback = 0 < pct_from_sma <= _PULLBACK_ZONE
    setup_type  = "breakout" if is_breakout else ("pullback" if is_pullback else None)

    if setup_type is None:
        reasons.append(
            f"no technical setup: {pct_from_52wh:.1%} from 52wk-high, "
            f"{pct_from_sma:+.1%} vs sma20"
        )

    # ⑤ SPY must be positive today — don't fight the market
    if spy_today_pct <= 0:
        reasons.append(f"SPY negative today ({spy_today_pct:+.2%})")

    # ⑥ Risk/reward: target must be at least 2× the stop distance away
    # Target = 2× stop distance above entry (gives R:R = 2.0 by construction),
    # then cap at 52-week high × 1.01 so we never target beyond proven resistance.
    target_base = round(entry * (1 + 2 * _STOP_PCT), 2)
    if high_52w > entry:
        target = round(min(high_52w * 1.01, target_base), 2)
    else:
        target = target_base
    rr_ratio = round((target - entry) / (entry - stop), 2) if entry > stop else 0.0
    if rr_ratio < _RR_MIN:
        reasons.append(f"R:R {rr_ratio:.2f} < {_RR_MIN} (entry={entry:.2f} stop={stop:.2f} target={target:.2f})")

    # ⑦ Macro must be at least neutral
    if macro_score < _HC_MACRO_MIN:
        reasons.append(f"macro {macro_score:.2f} < {_HC_MACRO_MIN}")

    passed = len(reasons) == 0
    meta = {
        "entry_price":   round(entry, 2),
        "stop_price":    stop,
        "target_price":  target,
        "rr_ratio":      rr_ratio,
        "setup_type":    setup_type or "none",
        "high_52w":      round(high_52w, 2),
        "volume_ratio":  round(volume_ratio, 2),
        "spy_today_pct": round(spy_today_pct, 4),
        "reason":        "; ".join(reasons) if reasons else "all gates passed",
    }

    if not passed:
        logger.debug(f"[SIGNAL GATE] {symbol} blocked: {meta['reason']}")
    else:
        logger.info(
            f"[SIGNAL GATE] {symbol} PASSED — entry={entry:.2f} stop={stop:.2f} "
            f"target={target:.2f} R:R={rr_ratio:.2f} setup={setup_type}"
        )

    return passed, meta
