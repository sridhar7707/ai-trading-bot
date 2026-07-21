"""Component ablation engine.

Simulates the effect of disabling individual investment engine components.

Two ablation modes
------------------
Filter ablation (signal components — XGB, LSTM, FinBERT, Macro):
  Re-compute the ensemble score without the disabled component (its weight
  redistributed proportionally to the remaining active components). Trades
  whose recomputed score falls below BUY_THRESHOLD are excluded from the
  scenario, showing how many trades *depended* on that component to fire.

Gate ablation (regime gate, LSTM veto):
  The regime gate and LSTM veto block entries before the ensemble score is
  checked, so their counterfactual trades are in signal_log (not in trades).
  For regime gate: the scenario filters existing trades by regime membership
  to show which regime the component is protecting against.
  For LSTM veto: if LSTM is disabled, the veto is also lifted.

Position sizing ablation (Kelly → flat):
  Since pnl_pct is percentage-based, win rate and avg return are unaffected.
  Dollar P&L is rescaled: flat_pnl = realized_pnl * (BUY_FRACTION / kelly_f).
  This lets you compare total realized P&L with Kelly vs. flat sizing.
"""
from __future__ import annotations

import pandas as pd

from bot.strategy.ensemble import (
    WEIGHTS, BUY_THRESHOLD, BUY_FRACTION,
    _LSTM_INDETERMINATE_LO, _LSTM_INDETERMINATE_HI,
)
from config import ENTRY_REGIMES

# Component key → column name in the trades DataFrame
_COL = {
    "xgb":       "xgb_prob",
    "lstm":      "lstm_prob",
    "sentiment": "sentiment_score",
    "macro":     "macro_score",
}

_NEUTRAL_RAW = {
    "xgb":       0.50,
    "lstm":      0.50,
    "sentiment": 0.00,  # raw [-1,+1]; normalises to 0.5
    "macro":     0.50,
}


def _sent_norm(raw: float) -> float:
    return (float(raw) + 1.0) / 2.0


def _ensemble_score(row: pd.Series, disabled: set[str]) -> float:
    """Recompute ensemble score with disabled components replaced by neutrals.

    Follows the same LSTM-indeterminate redistribution as ensemble.py:
      if LSTM is indeterminate (0.45–0.55) its weight transfers to XGB.
    When LSTM is *disabled* the weight transfers to the remaining active set.
    """
    active_keys = [k for k in WEIGHTS if k not in disabled]
    total_w     = sum(WEIGHTS[k] for k in active_keys)
    if total_w == 0:
        return 0.5

    # Raw component values (neutral when disabled)
    vals = {
        k: (float(row[_COL[k]]) if k not in disabled else _NEUTRAL_RAW[k])
        for k in WEIGHTS
    }
    vals_norm = dict(vals)
    vals_norm["sentiment"] = _sent_norm(vals["sentiment"])

    # LSTM indeterminate redistribution
    lstm_val   = vals["lstm"]
    lstm_indet = _LSTM_INDETERMINATE_LO <= lstm_val <= _LSTM_INDETERMINATE_HI

    # Build effective weights
    w = {k: WEIGHTS[k] for k in WEIGHTS}
    if "lstm" in disabled:
        # disabled: redistribute lstm weight proportionally to remaining active
        if total_w > 0:
            for k in active_keys:
                w[k] = WEIGHTS[k] / total_w
        w["lstm"] = 0.0
    elif lstm_indet:
        # indeterminate: transfer lstm weight to xgb (live logic)
        w["xgb"]  = WEIGHTS["xgb"] + WEIGHTS["lstm"]
        w["lstm"] = 0.0

    return sum(w[k] * vals_norm[k] for k in WEIGHTS)


def _lstm_veto_active(row: pd.Series) -> bool:
    """Return True if the live LSTM veto would block this trade."""
    lstm = float(row.get("lstm_prob", 0.5))
    indet = _LSTM_INDETERMINATE_LO <= lstm <= _LSTM_INDETERMINATE_HI
    return (not indet) and lstm < 0.50


def simulate(
    trades: pd.DataFrame,
    disabled: set[str] | None = None,
    disable_regime_gate: bool = False,
    flat_sizing: bool = False,
) -> pd.DataFrame:
    """Return the subset of trades that would have occurred under the scenario.

    Args:
        trades:              Completed BUY→SELL pairs from loader.load_completed_trades().
        disabled:            Set of component keys to disable — any of
                             {"xgb", "lstm", "sentiment", "macro"}.
        disable_regime_gate: If True, skip ENTRY_REGIMES filter. Existing trades
                             already passed the gate, so this only documents
                             which trades are in non-standard regimes.
        flat_sizing:         If True, add 'flat_realized_pnl' column showing
                             what realized P&L would have been with flat
                             BUY_FRACTION sizing instead of Kelly. pnl_pct
                             is unchanged (percentage returns don't vary with size).

    Returns:
        Filtered DataFrame (same schema as input) with an optional
        'ablated_score' column when signal components are disabled.
    """
    if trades.empty:
        return trades.copy()

    disabled = disabled or set()
    df = trades.copy()

    # ── Regime gate ──────────────────────────────────────────────────────────
    if not disable_regime_gate and "regime" in df.columns:
        df = df[df["regime"].isin(ENTRY_REGIMES)].copy()

    # ── Signal component ablation ─────────────────────────────────────────────
    if disabled:
        scores = df.apply(lambda row: _ensemble_score(row, disabled), axis=1)
        df = df[scores >= BUY_THRESHOLD].copy()
        df["ablated_score"] = scores[df.index]

        # Lift LSTM veto when LSTM is disabled
        if "lstm" not in disabled:
            veto = df.apply(_lstm_veto_active, axis=1)
            df = df[~veto].copy()

    # ── Flat sizing column ────────────────────────────────────────────────────
    if flat_sizing and "realized_pnl" in df.columns and "notional" in df.columns and "portfolio_value" in df.columns:
        def _flat_pnl(row: pd.Series) -> float:
            pv = float(row["portfolio_value"]) if row["portfolio_value"] > 0 else 1.0
            flat_notional = BUY_FRACTION * pv
            kelly_notional = float(row["notional"]) if row["notional"] > 0 else flat_notional
            if kelly_notional == 0:
                return 0.0
            return float(row["realized_pnl"]) * (flat_notional / kelly_notional)
        df["flat_realized_pnl"] = df.apply(_flat_pnl, axis=1)

    return df.reset_index(drop=True)


# Named scenarios used by evaluate_engine.py
SCENARIOS: dict[str, dict] = {
    "baseline":            {"disabled": set(),             "regime": False, "flat": False},
    "disable_xgb":         {"disabled": {"xgb"},           "regime": False, "flat": False},
    "disable_lstm":        {"disabled": {"lstm"},           "regime": False, "flat": False},
    "disable_finbert":     {"disabled": {"sentiment"},      "regime": False, "flat": False},
    "disable_macro":       {"disabled": {"macro"},          "regime": False, "flat": False},
    "disable_regime_gate": {"disabled": set(),              "regime": True,  "flat": False},
    "disable_kelly_sizing":{"disabled": set(),              "regime": False, "flat": True},
}


def gate_analysis(signal_log: pd.DataFrame) -> dict:
    """Summarise how many BUY-intent signals were blocked by hard gates.

    Uses signal_log (all evaluated cycles, including non-trades) to estimate
    the counterfactual size of each hard gate.

    Returns a dict with keys:
      total_buy_signals:       signals where ensemble_action in ('BUY','STRONG_BUY')
      blocked_by_regime:       buy signals in non-entry regimes
      pct_blocked_by_regime:   fraction of buy signals blocked by regime gate
    """
    if signal_log.empty:
        return {}

    buy_mask  = signal_log["ensemble_action"].isin(["BUY", "STRONG_BUY"])
    buy_sigs  = signal_log[buy_mask]
    total     = len(buy_sigs)

    regime_blocked = 0
    if "regime" in buy_sigs.columns:
        regime_blocked = int((~buy_sigs["regime"].isin(ENTRY_REGIMES)).sum())

    return {
        "total_buy_signals":       total,
        "blocked_by_regime":       regime_blocked,
        "pct_blocked_by_regime":   round(regime_blocked / total, 3) if total else 0.0,
    }
