"""Quality gate: run backtest on a holdout set and block model push if metrics fail."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
from loguru import logger
from backtest.engine import run_backtest
from config import SYMBOLS, INITIAL_CAPITAL

MIN_SHARPE = 0.0        # must be positive
MIN_RETURN = -0.05      # must not lose more than 5% on holdout

HOLDOUT_DAYS = 150      # ~90 trading days — enough headroom for indicator warmup (RSI-14, SMA-20, etc.)
MIN_VALID_ROWS = 50     # minimum rows required after feature warmup


def main():
    logger.info("=== Backtest quality gate ===")
    results = []

    for symbol in SYMBOLS[:3]:  # spot-check 3 symbols to keep CI fast
        try:
            df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df is None or len(df) < 100:
                logger.warning(f"{symbol}: insufficient data, skipping")
                continue

            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna()

            holdout = df.iloc[-HOLDOUT_DAYS:]
            if len(holdout) < MIN_VALID_ROWS:
                logger.warning(f"{symbol}: only {len(holdout)} rows in holdout (need {MIN_VALID_ROWS}), skipping")
                continue
            metrics = run_backtest(holdout, initial_balance=INITIAL_CAPITAL)
            sharpe = metrics.get("sharpe", 0.0)
            total_return = metrics.get("total_return", 0.0)
            logger.info(f"{symbol}: sharpe={sharpe:.2f}, return={total_return:.2%}")
            results.append((symbol, sharpe, total_return))
        except Exception as e:
            logger.error(f"{symbol} backtest failed: {e}")

    if not results:
        logger.error("No backtest results collected — possible data/network failure. Blocking model push.")
        sys.exit(1)

    avg_sharpe = sum(r[1] for r in results) / len(results)
    avg_return = sum(r[2] for r in results) / len(results)

    passed = avg_sharpe >= MIN_SHARPE and avg_return >= MIN_RETURN
    logger.info(f"Gate result: avg_sharpe={avg_sharpe:.2f}, avg_return={avg_return:.2%} — {'PASS' if passed else 'FAIL'}")

    if not passed:
        logger.error(
            f"Backtest gate FAILED (sharpe={avg_sharpe:.2f} < {MIN_SHARPE}, "
            f"return={avg_return:.2%} < {MIN_RETURN:.0%}). Blocking model push."
        )
        sys.exit(1)

    logger.info("Backtest gate PASSED — proceeding with model push.")
    sys.exit(0)


if __name__ == "__main__":
    main()
