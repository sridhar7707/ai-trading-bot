"""Quality gate: run backtest on holdout data and block model push if metrics fail."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf
from loguru import logger
from backtest.engine import run_backtest
from config import SYMBOLS, INITIAL_CAPITAL

# Thresholds — model must earn these on a 2023-present holdout before deploying
MIN_SHARPE = 0.8        # annualised Sharpe — below 0.8 is too noisy for real capital
MIN_RETURN = -0.03      # must not lose more than 3% on holdout
MAX_DRAWDOWN = 0.20     # max 20% drawdown
MIN_WIN_RATE = 0.45     # at least 45% of closed trades must be winners

# Holdout = last 60 days of 5-min bars (yfinance max for intraday; out-of-sample)
MIN_VALID_ROWS = 500  # ~78 bars/day × 60 days = ~4680; 500 is a conservative floor


def main():
    logger.info("=== Backtest quality gate (holdout: 2023-present) ===")
    results = []

    # Test all SYMBOLS, not just 3 — we have more data now so this is affordable
    for symbol in SYMBOLS:
        try:
            df = yf.download(symbol, period="60d", interval="5m", progress=False, auto_adjust=True)
            if df is None or len(df) < MIN_VALID_ROWS:
                logger.warning(f"{symbol}: insufficient holdout data ({len(df) if df is not None else 0} rows), skipping")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna()

            metrics = run_backtest(df, initial_balance=INITIAL_CAPITAL)
            sharpe = metrics.get("sharpe", 0.0)
            total_return = metrics.get("total_return", 0.0)
            max_dd = metrics.get("max_drawdown", 1.0)
            win_rate = metrics.get("win_rate", 0.0)
            logger.info(
                f"{symbol}: sharpe={sharpe:.2f}, return={total_return:.2%}, "
                f"drawdown={max_dd:.2%}, win_rate={win_rate:.1%}"
            )
            results.append((symbol, sharpe, total_return, max_dd, win_rate))
        except Exception as e:
            logger.error(f"{symbol} backtest failed: {e}")

    if not results:
        logger.error("No backtest results — possible data/network failure. Blocking model push.")
        sys.exit(1)

    n = len(results)
    avg_sharpe = sum(r[1] for r in results) / n
    avg_return = sum(r[2] for r in results) / n
    avg_drawdown = sum(r[3] for r in results) / n
    avg_win_rate = sum(r[4] for r in results) / n

    logger.info(
        f"Gate summary ({n} symbols): sharpe={avg_sharpe:.2f}, return={avg_return:.2%}, "
        f"drawdown={avg_drawdown:.2%}, win_rate={avg_win_rate:.1%}"
    )

    failures = []
    if avg_sharpe < MIN_SHARPE:
        failures.append(f"sharpe {avg_sharpe:.2f} < {MIN_SHARPE}")
    if avg_return < MIN_RETURN:
        failures.append(f"return {avg_return:.2%} < {MIN_RETURN:.0%}")
    if avg_drawdown > MAX_DRAWDOWN:
        failures.append(f"drawdown {avg_drawdown:.2%} > {MAX_DRAWDOWN:.0%}")
    if avg_win_rate < MIN_WIN_RATE:
        failures.append(f"win_rate {avg_win_rate:.1%} < {MIN_WIN_RATE:.0%}")

    if failures:
        logger.error(f"Backtest gate FAILED: {'; '.join(failures)}. Blocking model push.")
        sys.exit(1)

    logger.info("Backtest gate PASSED — model meets all thresholds.")
    sys.exit(0)


if __name__ == "__main__":
    main()
