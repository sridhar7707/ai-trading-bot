from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from loguru import logger
from bot.strategy.features import FEATURE_COLS

XGB_MODEL_PATH  = Path("models/saved/xgb_predictor.pkl")
FORWARD_PERIODS = 21   # 1-month target — aligned with MAX_HOLD_DAYS=45 medium-term horizon
MIN_MOVE_PCT    = 0.003  # require ≥0.3% move to label as "up" — filters 5-min microstructure noise


class XGBPredictor:
    def __init__(self):
        self.model = None
        self.val_auc: float = 0.0  # populated by train(); read by train_model.py for report
        self._load()

    def _load(self):
        if XGB_MODEL_PATH.exists():
            try:
                self.model = joblib.load(XGB_MODEL_PATH)
                # Detect feature set mismatch: model trained on a different FEATURE_COLS
                if hasattr(self.model, "feature_names_in_"):
                    trained = list(self.model.feature_names_in_)
                    if trained != FEATURE_COLS:
                        logger.warning(
                            f"XGBoost model trained on {len(trained)} features but "
                            f"FEATURE_COLS now has {len(FEATURE_COLS)} — model is stale. "
                            "Run scripts/train_model.py to retrain."
                        )
                        self.model = None
                        return
                logger.info("XGBoost model loaded.")
            except Exception as e:
                logger.warning(f"Failed to load XGBoost model: {e}")
        else:
            logger.warning("No XGBoost model found — will need training.")

    def train(self, df: pd.DataFrame, save: bool = True) -> None:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.error("xgboost not installed. Run: pip install xgboost")
            return

        df = df.copy()
        future_return = (df["close"].shift(-FORWARD_PERIODS) - df["close"]) / df["close"]
        df["target"] = (future_return > MIN_MOVE_PCT).astype(int)
        df.dropna(inplace=True)

        # Temporal split — train only on the first 80% so the model never sees
        # future prices during training (prevents lookahead bias / overfitting).
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        mask = train_df[FEATURE_COLS].notna().all(axis=1)
        X = train_df.loc[mask, FEATURE_COLS]
        y = train_df.loc[mask, "target"]

        val_mask = val_df[FEATURE_COLS].notna().all(axis=1)
        X_val = val_df.loc[val_mask, FEATURE_COLS]
        y_val = val_df.loc[val_mask, "target"]

        self.model = XGBClassifier(
            n_estimators=1000,         # high ceiling — early stopping finds the right count
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            early_stopping_rounds=50,  # XGBoost 2.x: moved from fit() to constructor
            random_state=42,
        )
        self.model.fit(
            X, y,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best_trees = self.model.best_iteration
        logger.info(f"XGBoost early stopped at tree {best_trees}")

        if len(X_val) > 0:
            val_proba = self.model.predict_proba(X_val)[:, 1]
            val_auc   = float(roc_auc_score(y_val, val_proba))
            self.val_auc = val_auc
            if val_auc < 0.52:
                logger.warning(
                    f"XGBoost val AUC-ROC {val_auc:.3f} is near random — "
                    "model may have degraded. Review training data before deploying."
                )
            else:
                logger.info(f"XGBoost val AUC-ROC (holdout 20%): {val_auc:.3f}")

        if save:
            XGB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, XGB_MODEL_PATH)
            logger.info(f"XGBoost trained and saved to {XGB_MODEL_PATH}")

    def predict_proba(self, row: pd.Series) -> float:
        """Return probability (0-1) that price will be higher in {FORWARD_PERIODS} candles."""
        if self.model is None:
            return 0.5
        try:
            features = pd.DataFrame([row[FEATURE_COLS]])
            return float(self.model.predict_proba(features)[0, 1])
        except Exception as e:
            logger.error(f"XGBoost predict failed: {e}")
            return 0.5

    def explain(self, row: pd.Series) -> list:
        """Return top-3 [(feature_name, shap_value), ...] driving this prediction.

        Uses XGBoost's built-in tree SHAP — no extra library required.
        Positive shap_value = pushed toward BUY; negative = pushed away.
        Returns [] if model is unavailable or SHAP computation fails.
        """
        if self.model is None:
            return []
        try:
            from xgboost import DMatrix
            X = row[FEATURE_COLS].values.reshape(1, -1).astype(float)
            dmat = DMatrix(X, feature_names=FEATURE_COLS)
            # pred_contribs shape: (1, n_features+1) — last column is the bias term
            contribs = self.model.get_booster().predict(dmat, pred_contribs=True)[0][:-1]
            idx = np.argsort(np.abs(contribs))[::-1][:3]
            return [(FEATURE_COLS[i], round(float(contribs[i]), 4)) for i in idx]
        except Exception as exc:
            logger.warning(f"XGBoost explain failed: {exc}")
            return []
