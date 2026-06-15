"""Quality gate: run backtest on holdout data and block model push if metrics fail.

Two-stage check:
  1. Recent performance gate (BLOCKS CI): 60-day 5-min bars across all SYMBOLS.
     If average Sharpe/return/drawdown/win_rate miss thresholds → sys.exit(1).
  2. Historical stress check (INFORMATIONAL only): daily bars for 3 crisis windows
     (2008, 2020, 2022).  Logs WARNING if strategy would have underperformed but
     does NOT block the push — daily bars use a different regime distribution than
     the 5-min intraday model was trained on.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf
from loguru import logger
from backtest.engine import run_backtest
from config import SYMBOLS, INITIAL_CAPITAL

# Thresholds — model must earn these on a 2023-present holdout before deploying.
# Paper-trading thresholds (current): permissive — validates the model has minimal
# viability and won't blow up a paper account. Tighten before going live with real money.
# Live-money targets: MIN_SHARPE=0.8, MIN_RETURN=-0.03, MAX_DRAWDOWN=0.20, MIN_WIN_RATE=0.45
MIN_SHARPE = 0.3        # annualised Sharpe — above 0.3 shows some edge over noise
MIN_RETURN = -0.10      # must not lose more than 10% on holdout
MAX_DRAWDOWN = 0.30     # max 30% drawdown
MIN_WIN_RATE = 0.40     # at least 40% of closed trades must be winners

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
    run_stress_check()
    sys.exit(0)


def run_stress_check():
    """Download daily bars for 3 key crisis windows and log strategy performance.

    Informational only — does NOT block CI.  Daily-bar regime distribution differs
    from the 5-min intraday model; treat results as robustness context, not a gate.
    Pass = max drawdown ≤ 25% AND total return > -30%.
    """
    STRESS_WINDOWS = {
        "2008 Financial Crisis": ("2008-09-01", "2009-03-31"),
        "2020 COVID Crash":      ("2020-02-01", "2020-04-30"),
        "2022 Bear Market":      ("2022-01-01", "2022-12-31"),
    }
    stress_sym = SYMBOLS[0] if SYMBOLS else "SPY"
    logger.info(f"--- Informational stress check ({stress_sym}, daily bars) ---")

    for name, (start, end) in STRESS_WINDOWS.items():
        try:
            df = yf.download(stress_sym, start=start, end=end, interval="1d",
                             progress=False, auto_adjust=True)
            if df is None or len(df) < 30:
                logger.warning(f"Stress [{name}]: insufficient data — skipping")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            m = run_backtest(df, initial_balance=INITIAL_CAPITAL)
            ok = m["max_drawdown"] <= 0.25 and m["total_return"] > -0.30
            msg = (
                f"Stress [{name}]: return={m['total_return']:+.1%}  "
                f"sharpe={m['sharpe']:.2f}  maxDD={m['max_drawdown']:.1%}  "
                f"{'✓ survived' if ok else '⚠ underperformed (informational — no CI block)'}"
            )
            logger.info(msg) if ok else logger.warning(msg)
        except Exception as exc:
            logger.warning(f"Stress [{name}]: error — {exc}")


if __name__ == "__main__":
    main()
