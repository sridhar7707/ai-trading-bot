from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from bot.strategy.features import FEATURE_COLS

LSTM_MODEL_PATH = Path("models/saved/lstm_predictor.pt")
SEQ_LEN         = 60
FORWARD_PERIODS = 5
PATIENCE        = 3   # early stopping: halt if val loss doesn't improve for this many epochs


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
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class LSTMPredictor:
    def __init__(self):
        self.model: _LSTMModel | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self):
        if LSTM_MODEL_PATH.exists():
            try:
                self.model = _LSTMModel(input_size=len(FEATURE_COLS)).to(self.device)
                self.model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=self.device))
                self.model.eval()
                logger.info("LSTM model loaded.")
            except Exception as e:
                logger.warning(f"Failed to load LSTM model: {e}")
                self.model = None
        else:
            logger.warning("No LSTM model found — will need training.")

    def _make_sequences(self, df: pd.DataFrame):
        data  = df[FEATURE_COLS].values.astype(np.float32)
        future_close = df["close"].shift(-FORWARD_PERIODS)
        labels = (future_close > df["close"]).astype(float).values

        X, y = [], []
        for i in range(SEQ_LEN, len(data) - FORWARD_PERIODS):
            X.append(data[i - SEQ_LEN:i])
            y.append(labels[i])
        return np.array(X), np.array(y, dtype=np.float32)

    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 64):
        X, y = self._make_sequences(df)
        if len(X) == 0:
            logger.error("Not enough data to train LSTM (need at least 65 rows).")
            return

        # Temporal 80/20 split — no shuffle across the split boundary to prevent lookahead
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        X_train_t = torch.tensor(X_train).to(self.device)
        y_train_t = torch.tensor(y_train).unsqueeze(1).to(self.device)
        X_val_t   = torch.tensor(X_val).to(self.device)
        y_val_t   = torch.tensor(y_val).unsqueeze(1).to(self.device)

        self.model = _LSTMModel(input_size=len(FEATURE_COLS)).to(self.device)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion  = nn.BCELoss()
        dataset    = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        loader     = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

            logger.info(
                f"LSTM epoch {epoch + 1}/{epochs} — "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                torch.save(self.model.state_dict(), LSTM_MODEL_PATH)
            else:
                wait += 1
                if wait >= PATIENCE:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(best val_loss={best_val_loss:.4f})"
                    )
                    break

        # Restore best weights (saved when val_loss was lowest)
        self.model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=self.device))
        self.model.eval()
        logger.info(f"LSTM training complete — best val_loss={best_val_loss:.4f}")

    def predict_proba(self, df: pd.DataFrame) -> float:
        """Return probability (0-1) that price will be higher. Needs full bars DataFrame."""
        if self.model is None or len(df) < SEQ_LEN:
            return 0.5
        try:
            seq = df[FEATURE_COLS].values[-SEQ_LEN:].astype(np.float32)
            if np.isnan(seq).any():
                logger.warning("LSTM predict: NaN in feature sequence — returning 0.5")
                return 0.5
            x = torch.tensor(seq).unsqueeze(0).to(self.device)
            with torch.no_grad():
                result = float(self.model(x).item())
            if math.isnan(result):
                logger.warning("LSTM predict: model output is NaN — returning 0.5")
                return 0.5
            return result
        except Exception as e:
            logger.error(f"LSTM predict failed: {e}")
            return 0.5
