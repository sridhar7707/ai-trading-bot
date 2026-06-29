from __future__ import annotations
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from loguru import logger
from bot.strategy.features import FEATURE_COLS

LSTM_MODEL_PATH  = Path("models/saved/lstm_predictor.pt")
SCALER_PATH      = Path("models/saved/lstm_scaler.pkl")
SEQ_LEN          = 20   # 4-week lookback (was 60); matches 1-week prediction target, faster retraining
FORWARD_PERIODS  = 5   # 1-week target — matches XGBPredictor short-term horizon (retrain required)
MIN_MOVE_PCT     = 0.003   # must match XGBPredictor — both models predict the same target
PATIENCE         = 7       # stop if val_loss doesn't improve for this many epochs
LR_PATIENCE      = 3       # halve LR after this many epochs without improvement
LR_MIN           = 1e-5    # stop reducing LR below this value


class _LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            # No Sigmoid here — BCEWithLogitsLoss includes it during training;
            # torch.sigmoid() is applied explicitly in predict_proba().
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# Val-loss above this threshold means the LSTM has converged to outputting logit≈0
# (sigmoid(0)=0.5) and is providing no directional signal.
LSTM_DEGRADED_VAL_LOSS = 0.58

_VALIDATION_REPORT_PATH = Path("models/validation_report.json")


class LSTMPredictor:
    def __init__(self):
        self.model:  _LSTMModel | None   = None
        self.scaler: StandardScaler | None = None
        self.val_loss: float = 1.0  # populated by train(); read by train_model.py for report
        self.is_degraded: bool = False  # True when loaded model val_loss > LSTM_DEGRADED_VAL_LOSS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self):
        if LSTM_MODEL_PATH.exists():
            try:
                self.model = _LSTMModel(input_size=len(FEATURE_COLS)).to(self.device)
                self.model.load_state_dict(
                    torch.load(LSTM_MODEL_PATH, map_location=self.device)
                )
                self.model.eval()
                logger.info("LSTM model loaded.")
            except Exception as e:
                logger.warning(f"Failed to load LSTM model: {e}")
                self.model = None
        else:
            logger.warning("No LSTM model found — will need training.")

        if SCALER_PATH.exists():
            try:
                self.scaler = joblib.load(SCALER_PATH)
                logger.info("LSTM feature scaler loaded.")
            except Exception as e:
                logger.warning(f"Failed to load LSTM scaler: {e}")
                self.scaler = None

        self._check_degradation()

    def _check_degradation(self) -> None:
        """Read validation_report.json and flag the model as degraded if val_loss is near-random."""
        if not _VALIDATION_REPORT_PATH.exists():
            return
        try:
            import json
            report = json.loads(_VALIDATION_REPORT_PATH.read_text())
            saved_val_loss = float(report.get("lstm_val_loss", 0.0))
            if saved_val_loss > LSTM_DEGRADED_VAL_LOSS:
                self.is_degraded = True
                self.val_loss = saved_val_loss
                logger.warning(
                    f"⚠ LSTM model is DEGRADED — val_loss={saved_val_loss:.4f} "
                    f"(threshold={LSTM_DEGRADED_VAL_LOSS}). "
                    f"Model outputs will be treated as indeterminate. "
                    f"Run: python scripts/train_model.py"
                )
        except Exception as exc:
            logger.debug(f"LSTM degradation check skipped: {exc}")

    def _make_sequences(
        self, df: pd.DataFrame, symbol_col: pd.Series | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build (X, y) sequences, grouped by symbol to prevent cross-symbol windows."""
        X_all: list[np.ndarray] = []
        y_all: list[float]      = []

        def _seq_for(sub_df: pd.DataFrame):
            data         = sub_df[FEATURE_COLS].values.astype(np.float32)
            future_ret   = (sub_df["close"].shift(-FORWARD_PERIODS) - sub_df["close"]) / sub_df["close"]
            labels       = (future_ret > MIN_MOVE_PCT).astype(float).values
            for i in range(SEQ_LEN, len(data) - FORWARD_PERIODS):
                X_all.append(data[i - SEQ_LEN: i])
                y_all.append(labels[i])

        if symbol_col is not None:
            for sym in symbol_col.unique():
                mask = symbol_col == sym
                _seq_for(df[mask.values])
        else:
            _seq_for(df)

        return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.float32)

    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 64) -> None:
        symbol_col = df["symbol"] if "symbol" in df.columns else None
        df_feat    = df.drop(columns=["symbol", "regime"], errors="ignore")

        X, y = self._make_sequences(df_feat, symbol_col=symbol_col)
        if len(X) == 0:
            logger.error("Not enough data to train LSTM (need at least 65 rows per symbol).")
            return

        # Temporal 80/20 split — no shuffle across the boundary
        split   = int(len(X) * 0.8)
        X_tr_r, X_va_r = X[:split], X[split:]
        y_train, y_val  = y[:split], y[split:]

        # Fit StandardScaler on training timesteps only (flattened to 2D)
        n_tr, seq_l, n_f = X_tr_r.shape
        self.scaler = StandardScaler()
        self.scaler.fit(X_tr_r.reshape(-1, n_f))
        SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, SCALER_PATH)

        X_train = self.scaler.transform(X_tr_r.reshape(-1, n_f)).reshape(n_tr, seq_l, n_f)
        n_va    = X_va_r.shape[0]
        X_val   = self.scaler.transform(X_va_r.reshape(-1, n_f)).reshape(n_va, seq_l, n_f)

        X_train_t = torch.tensor(X_train).to(self.device)
        y_train_t = torch.tensor(y_train).unsqueeze(1).to(self.device)
        X_val_t   = torch.tensor(X_val).to(self.device)
        y_val_t   = torch.tensor(y_val).unsqueeze(1).to(self.device)

        # Class-balanced loss — corrects for imbalance in >0.3% up-move label
        pos_count = float(y_train.sum())
        neg_count = float(len(y_train) - pos_count)
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info(f"LSTM class balance: pos={pos_count:.0f} neg={neg_count:.0f} "
                    f"pos_weight={pos_weight.item():.2f}")

        self.model = _LSTMModel(input_size=len(FEATURE_COLS)).to(self.device)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=LR_PATIENCE,
            min_lr=LR_MIN,
        )
        dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        wait          = 0
        LSTM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(loader)

            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(X_val_t), y_val_t).item()

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"LSTM epoch {epoch + 1}/{epochs} — "
                f"train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.2e}"
            )

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                torch.save(self.model.state_dict(), LSTM_MODEL_PATH)
            else:
                wait += 1
                if wait >= PATIENCE or current_lr <= LR_MIN:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(best val={best_val_loss:.4f}, wait={wait}, lr={current_lr:.2e})"
                    )
                    break

        self.val_loss = best_val_loss
        self.model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=self.device))
        self.model.eval()
        logger.info(f"LSTM training complete — best val_loss={best_val_loss:.4f}")

    def predict_proba(self, df: pd.DataFrame) -> float:
        """Return probability (0–1) that price will be ≥0.3% higher in {FORWARD_PERIODS} bars."""
        if self.model is None or len(df) < SEQ_LEN:
            return 0.5
        try:
            seq = df[FEATURE_COLS].values[-SEQ_LEN:].astype(np.float32)
            if np.isnan(seq).any():
                logger.warning("LSTM predict: NaN in feature sequence — returning 0.5")
                return 0.5
            if self.scaler is not None:
                seq = self.scaler.transform(seq).astype(np.float32)
            x = torch.tensor(seq).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logit = self.model(x).item()
            result = float(torch.sigmoid(torch.tensor(logit)).item())
            if math.isnan(result):
                logger.warning("LSTM predict: output is NaN — returning 0.5")
                return 0.5
            return result
        except Exception as e:
            logger.error(f"LSTM predict failed: {e}")
            return 0.5
