"""Offline training script — trains all models and saves validation artefacts.
Run locally or on HuggingFace ZeroGPU.

Outputs after a successful run:
  models/saved/                   — trained model artefacts (pkl / pt)
  models/validation_report.json   — XGB val AUC, LSTM val loss, training metadata
  models/feature_importance.json  — XGBoost feature importances (used by dashboard chart)

Walk-forward split: train on data before TRAIN_CUTOFF (2007–2025), validate on 2026-present.
Re-run scripts/download_data.py first if raw data is missing or stale.
"""
from __future__ import annotations

import json, os, sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from loguru import logger
from bot.strategy.features import FEATURE_COLS, compute_features
from bot.strategy.regime_classifier import RegimeClassifier, label_regime
from bot.strategy.xgb_predictor import XGBPredictor
from bot.strategy.lstm_predictor import LSTMPredictor
from config import TRAINING_SYMBOLS, INITIAL_CAPITAL

DATA_DIR = "data/raw"

# Walk-forward split: train on 2007–2025, validate/test on 2026-present
TRAIN_CUTOFF = "2026-01-01"


def load_combined(cutoff: str | None = None) -> pd.DataFrame:
    frames = []
    for sym in TRAINING_SYMBOLS:
        path = f"{DATA_DIR}/{sym}.csv"
        if not os.path.exists(path):
            logger.warning(f"Missing data for {sym} — skipping.")
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if cutoff:
            df = df[df.index < cutoff]
        df = compute_features(df)
        df["regime"] = label_regime(df)
        df["symbol"] = sym
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No data found. Run scripts/download_data.py first.")
    return pd.concat(frames).sort_index()


def main():
    logger.info(f"Loading training data (cutoff={TRAIN_CUTOFF}) — {len(TRAINING_SYMBOLS)} symbols...")
    df = load_combined(cutoff=TRAIN_CUTOFF)
    date_min = df.index.min().date().isoformat()
    date_max = df.index.max().date().isoformat()
    logger.info(f"Training rows: {len(df):,} | date range: {date_min} → {date_max}")

    logger.info("Training regime classifier...")
    regime_clf = RegimeClassifier()
    regime_clf.train(df)

    # Use combined data for XGBoost (more samples = better generalisation)
    logger.info("Training XGBoost predictor...")
    xgb = XGBPredictor()
    xgb.train(df.drop(columns=["symbol", "regime"], errors="ignore"))

    # symbol column is kept so _make_sequences() can group per-symbol,
    # preventing 60-bar windows that mix data from different companies.
    logger.info("Training LSTM predictor on full multi-symbol dataset...")
    lstm = LSTMPredictor()
    lstm.train(df.drop(columns=["regime"], errors="ignore"))

    os.makedirs("models", exist_ok=True)

    # ── Validation report ─────────────────────────────────────────────────────
    # Captures out-of-sample quality signal from each model's internal 80/20 split.
    # Pushed to HF alongside model weights so the dashboard can display it without
    # having to load the model files.
    report = {
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "train_cutoff":     TRAIN_CUTOFF,
        "training_symbols": len(TRAINING_SYMBOLS),
        "training_rows":    len(df),
        "date_range":       {"from": date_min, "to": date_max},
        "xgb_val_auc":      round(xgb.val_auc, 4),
        "lstm_val_loss":    round(lstm.val_loss, 4),
    }
    with open("models/validation_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    logger.info(
        f"Validation report → models/validation_report.json "
        f"(xgb_auc={report['xgb_val_auc']}, lstm_loss={report['lstm_val_loss']})"
    )

    # ── Feature importance ────────────────────────────────────────────────────
    # XGBClassifier.feature_importances_ = normalised gain — same scale across runs.
    # Saved separately so the dashboard can render the explainability chart without
    # loading the full 10 MB model file.
    if xgb.model is not None:
        importances = dict(zip(FEATURE_COLS, xgb.model.feature_importances_.tolist()))
        with open("models/feature_importance.json", "w") as fh:
            json.dump(importances, fh, indent=2)
        logger.info("Feature importance → models/feature_importance.json")

    logger.info("All models trained successfully.")


if __name__ == "__main__":
    main()
