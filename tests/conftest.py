import os
import sys
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Prevent bot/main.py from adding the trading.log file sink during test collection.
# bot/main.py guards logger.add() with this env var; setting it here before any
# bot import ensures tests never write to logs/trading.log.
os.environ.setdefault("_BOT_LOG_HANDLER_ADDED", "1")

# alpaca_trade_api → aiohttp has a TypedDict bug on Python 3.9.
# Mock the package early so test_alpaca_client and test_main can be collected
# on any Python version without a real Alpaca installation.
if "alpaca_trade_api" not in sys.modules:
    _alpaca_mock = MagicMock()
    for _mod in ("alpaca_trade_api", "alpaca_trade_api.rest",
                 "alpaca_trade_api.rest_async", "alpaca_trade_api.stream"):
        sys.modules[_mod] = _alpaca_mock


def make_ohlcv(n: int = 270, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with enough rows for all indicators including 252-bar momentum."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.clip(close, 1, None)
    high = close + np.abs(rng.normal(0, 0.3, n))
    low = close - np.abs(rng.normal(0, 0.3, n))
    low = np.clip(low, 0.01, None)
    open_ = close + rng.normal(0, 0.2, n)
    volume = np.abs(rng.normal(10000, 1000, n)) + 100
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


@pytest.fixture
def ohlcv():
    return make_ohlcv()
