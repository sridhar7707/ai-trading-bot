"""Walk-forward cross-validation: compare FEATURE_COLS_V3 vs FEATURE_COLS_V4.

Trains XGBoost on 5 expanding windows and reports AUC, simulated win rate,
and simulated average return per window. Use this to confirm V4 features
improve medium-term prediction before committing FEATURE_COLS = FEATURE_COLS_V4.

Usage
-----
    python scripts/walkforward_validate.py [--symbols N] [--verbose]

Output
------
Per-window table (V3 vs V4), then aggregate mean AUC and simulated Sharpe.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from bot.strategy.features import FEATURE_COLS_V3, FEATURE_COLS_V4, compute_features
from bot.strategy.xgb_predictor import FORWARD_PERIODS, MIN_MOVE_PCT
from config import TRAINING_SYMBOLS

DATA_DIR = "data/raw"

# 5 expanding windows — each trains on all data before the test year.
# Most recent 5 calendar years give the best signal for current market dynamics.
WINDOWS = [
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-01-01"),
]

BUY_THRESHOLD = 0.55   # matches ensemble.py BUY_THRESHOLD


class WindowResult(NamedTuple):
    test_start: str
    feature_set: str
    n_train: int
    n_test: int
    auc: float
    win_rate: float
    avg_return: float
    n_signals: int


def _load_spy_close(cutoff: str | None = None) -> pd.Series | None:
    path = f"{DATA_DIR}/SPY.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if cutoff:
        df = df[df.index < cutoff]
    return df["close"]


def _load_symbol(sym: str, spy_close: pd.Series | None) -> pd.DataFrame | None:
    path = f"{DATA_DIR}/{sym}.csv"
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = compute_features(df, spy_close=spy_close)
        df["symbol"] = sym
        return df
    except Exception as e:
        logger.debug(f"Skipping {sym}: {e}")
        return None


def _build_dataset(
    cutoff: str | None,
    feature_cols: list[str],
    symbols: list[str],
    spy_close: pd.Series | None,
) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        path = f"{DATA_DIR}/{sym}.csv"
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if cutoff:
                df = df[df.index < cutoff]
            df = compute_features(df, spy_close=spy_close)
            future_ret = (df["close"].shift(-FORWARD_PERIODS) - df["close"]) / df["close"]
            df["target"] = (future_ret > MIN_MOVE_PCT).astype(int)
            df = df.dropna(subset=feature_cols + ["target"])
            df["symbol"] = sym
            frames.append(df[feature_cols + ["target", "close", "symbol"]])
        except Exception as e:
            logger.debug(f"Skipping {sym}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def _train_and_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[float, float, float, int]:
    """Returns (auc, win_rate, avg_return, n_signals)."""
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test  = test_df[feature_cols]
    y_test  = test_df["target"]

    if len(X_train) < 100 or y_train.nunique() < 2:
        return 0.0, 0.0, 0.0, 0

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42,
    )
    # Internal 10% val split for early stopping only — NOT the hold-out window
    split = int(len(X_train) * 0.9)
    model.fit(
        X_train.iloc[:split], y_train.iloc[:split],
        eval_set=[(X_train.iloc[split:], y_train.iloc[split:])],
        verbose=False,
    )

    if len(X_test) == 0 or y_test.nunique() < 2:
        return 0.0, 0.0, 0.0, 0

    proba   = model.predict_proba(X_test)[:, 1]
    auc     = float(roc_auc_score(y_test, proba))

    # Simulate: "enter" when predict_proba > BUY_THRESHOLD, measure forward return
    signal_mask  = proba > BUY_THRESHOLD
    n_signals    = int(signal_mask.sum())
    if n_signals == 0:
        return auc, 0.0, 0.0, 0

    future_ret   = (test_df["close"].shift(-FORWARD_PERIODS) / test_df["close"] - 1)
    trade_rets   = future_ret.values[signal_mask]
    trade_rets   = trade_rets[~np.isnan(trade_rets)]
    if len(trade_rets) == 0:
        return auc, 0.0, 0.0, 0

    win_rate   = float((trade_rets > 0).mean())
    avg_return = float(trade_rets.mean())
    return auc, win_rate, avg_return, len(trade_rets)


def run_walkforward(symbols: list[str], verbose: bool) -> list[WindowResult]:
    results: list[WindowResult] = []

    for test_start, test_end in WINDOWS:
        if verbose:
            logger.info(f"Window: test {test_start} to {test_end}")

        spy_train = _load_spy_close(cutoff=test_start)
        spy_full  = _load_spy_close(cutoff=test_end)

        for feat_name, feat_cols in [("V3", FEATURE_COLS_V3), ("V4", FEATURE_COLS_V4)]:
            train_df = _build_dataset(test_start, feat_cols, symbols, spy_train)
            full_df  = _build_dataset(test_end,   feat_cols, symbols, spy_full)

            if train_df.empty or full_df.empty:
                continue

            # Test set = rows in full_df that are NOT in training period
            test_df = full_df[full_df.index >= test_start]
            if test_df.empty:
                continue

            auc, wr, avg_ret, n_sig = _train_and_eval(train_df, test_df, feat_cols)
            results.append(WindowResult(
                test_start  = test_start,
                feature_set = feat_name,
                n_train     = len(train_df),
                n_test      = len(test_df),
                auc         = auc,
                win_rate    = wr,
                avg_return  = avg_ret,
                n_signals   = n_sig,
            ))
            if verbose:
                logger.info(
                    f"  {feat_name}: n_train={len(train_df):,}  n_test={len(test_df):,}  "
                    f"auc={auc:.4f}  wr={wr*100:.1f}%  avg_ret={avg_ret*100:+.2f}%  "
                    f"signals={n_sig}"
                )

    return results


def _print_report(results: list[WindowResult]) -> None:
    W = 80
    print(f"\n{'=' * W}")
    print("  WALK-FORWARD VALIDATION  —  V3 (current) vs V4 (medium-term features)")
    print(f"  FORWARD_PERIODS={FORWARD_PERIODS}  BUY_THRESHOLD={BUY_THRESHOLD}")
    print(f"{'=' * W}")

    print(
        f"\n  {'Test year':<12}  {'Set':<4}  {'n_train':>8}  {'n_test':>7}"
        f"  {'AUC':>6}  {'Win%':>6}  {'AvgRet':>8}  {'Signals':>8}"
    )
    print(f"  {'-' * 72}")

    for r in results:
        year = r.test_start[:4]
        print(
            f"  {year:<12}  {r.feature_set:<4}  {r.n_train:>8,}  {r.n_test:>7,}"
            f"  {r.auc:>6.4f}  {r.win_rate*100:>5.1f}%  {r.avg_return*100:>+7.2f}%  {r.n_signals:>8}"
        )

    # Aggregate by feature set
    print(f"\n{'=' * W}")
    print("  AGGREGATE (mean across windows)")
    print(f"  {'-' * 72}")

    for feat in ("V3", "V4"):
        subset = [r for r in results if r.feature_set == feat]
        if not subset:
            continue
        mean_auc = np.mean([r.auc for r in subset])
        mean_wr  = np.mean([r.win_rate for r in subset])
        mean_ret = np.mean([r.avg_return for r in subset])
        total_sig = sum(r.n_signals for r in subset)
        print(
            f"  {feat:<4}  mean_auc={mean_auc:.4f}"
            f"  mean_win={mean_wr*100:.1f}%"
            f"  mean_ret={mean_ret*100:+.2f}%"
            f"  total_signals={total_sig}"
        )

    # Delta
    v3 = [r for r in results if r.feature_set == "V3"]
    v4 = [r for r in results if r.feature_set == "V4"]
    if v3 and v4 and len(v3) == len(v4):
        d_auc = np.mean([r.auc for r in v4]) - np.mean([r.auc for r in v3])
        d_wr  = np.mean([r.win_rate for r in v4]) - np.mean([r.win_rate for r in v3])
        d_ret = np.mean([r.avg_return for r in v4]) - np.mean([r.avg_return for r in v3])
        verdict = "V4 WINS" if d_auc > 0 and d_ret > 0 else ("V3 WINS" if d_auc < 0 else "MIXED")
        print(f"\n  Delta V4 - V3:  auc={d_auc:+.4f}  win={d_wr*100:+.1f}pp  ret={d_ret*100:+.2f}pp")
        print(f"  Verdict: {verdict}")
        if verdict == "V4 WINS":
            print("  -> FEATURE_COLS = FEATURE_COLS_V4 is confirmed. Retrain with scripts/train_model.py.")
        elif verdict == "V3 WINS":
            print("  -> Keep FEATURE_COLS_V3. Investigate V4 feature quality before retrain.")
        else:
            print("  -> Mixed result. Review per-window breakdown before deciding.")

    print(f"\n{'=' * W}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward validation: V3 vs V4 feature sets"
    )
    parser.add_argument(
        "--symbols", type=int, default=len(TRAINING_SYMBOLS),
        help=f"Number of symbols to use (default: all {len(TRAINING_SYMBOLS)})",
    )
    parser.add_argument("--verbose", action="store_true", help="Log per-window progress")
    args = parser.parse_args()

    symbols = TRAINING_SYMBOLS[: args.symbols]
    print(f"\nWalk-forward validation on {len(symbols)} symbols, {len(WINDOWS)} windows...")
    print(f"V3 features: {len(FEATURE_COLS_V3)}  |  V4 features: {len(FEATURE_COLS_V4)}")

    results = run_walkforward(symbols, verbose=args.verbose)

    if not results:
        print("No results — check that data/raw/ has CSV files. Run scripts/download_data.py first.")
        sys.exit(1)

    _print_report(results)


if __name__ == "__main__":
    main()
