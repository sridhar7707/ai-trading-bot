import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with enough rows for all indicators."""
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
