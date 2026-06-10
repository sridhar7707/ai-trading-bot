"""
Walk-forward out-of-sample validation.

Splits data into a training window (2015–TRAIN_CUTOFF) and a held-out test window
(TRAIN_CUTOFF–present), then runs the full ensemble backtest on the test period.
Reports comprehensive metrics so you know actual out-of-sample performance before
risking real capital.

Usage:
    python scripts/walk_forward.py
    python scripts/walk_forward.py --symbol SPY
    python scripts/walk_forward.py --rolling --train-months 12 --test-months 3
"""
from __future__ import annotations
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from loguru import logger
from backtest.engine  import run_backtest
from backtest.metrics import compute_metrics
from config import TRAINING_SYMBOLS, INITIAL_CAPITAL

DATA_DIR    = "data/raw"
TRAIN_CUTOFF = "2023-01-01"


def _load_symbol(sym: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    path = f"{DATA_DIR}/{sym}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data for {sym} — run scripts/download_data.py first.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index < end]
    return df


def _print_metrics(label: str, m: dict):
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  Total Return:    {m['total_return']:>+8.2%}")
    print(f"  Ann. Return:     {m['ann_return']:>+8.2%}")
    print(f"  Sharpe Ratio:    {m['sharpe']:>8.2f}")
    print(f"  Calmar Ratio:    {m['calmar']:>8.2f}")
    print(f"  Max Drawdown:    {m['max_drawdown']:>8.2%}")
    print(f"  Profit Factor:   {m['profit_factor']:>8.2f}")
    print(f"  Win Rate:        {m['win_rate']:>8.1%}")
    print(f"  Expectancy:      {m['expectancy']:>+8.3%}")
    print(f"  Avg Win:         {m['avg_win']:>+8.2%}")
    print(f"  Avg Loss:        {m['avg_loss']:>+8.2%}")
    print(f"  Num Trades:      {m['num_trades']:>8d}")
    print(f"  Final Value:     ${m['final_value']:>10,.2f}")


def run_simple(symbol: str):
    """Single-symbol in-sample vs out-of-sample comparison."""
    logger.info(f"Loading {symbol} data...")
    df_train = _load_symbol(symbol, end=TRAIN_CUTOFF)
    df_test  = _load_symbol(symbol, start=TRAIN_CUTOFF)
    if df_test.empty:
        logger.error(f"No test data for {symbol} after {TRAIN_CUTOFF}.")
        return

    print(f"\n{'═'*55}")
    print(f"  Walk-Forward Validation — {symbol}")
    print(f"  Train: up to {TRAIN_CUTOFF}  |  Test: {TRAIN_CUTOFF}+")
    print(f"{'═'*55}")

    logger.info("Running in-sample backtest...")
    m_train = run_backtest(df_train, initial_balance=INITIAL_CAPITAL)
    _print_metrics(f"IN-SAMPLE  (train period)", m_train)

    logger.info("Running out-of-sample backtest...")
    m_test = run_backtest(df_test, initial_balance=INITIAL_CAPITAL)
    _print_metrics(f"OUT-OF-SAMPLE  (test period, {TRAIN_CUTOFF}+)", m_test)

    # Degradation check
    sr_delta  = m_test["sharpe"]  - m_train["sharpe"]
    ret_delta = m_test["ann_return"] - m_train["ann_return"]
    print(f"\n  Sharpe degradation:  {sr_delta:>+.2f}")
    print(f"  Return degradation:  {ret_delta:>+.2%}")
    if m_test["sharpe"] < 0.5:
        print("\n  ⚠  Out-of-sample Sharpe < 0.5 — consider retraining or reviewing signals.")
    elif m_test["sharpe"] > 1.0:
        print("\n  ✓  Out-of-sample Sharpe > 1.0 — strategy looks robust.")


def run_rolling(symbol: str, train_months: int = 12, test_months: int = 3):
    """Rolling walk-forward: train on train_months, test on next test_months, slide forward."""
    df_all = _load_symbol(symbol)
    if df_all.empty:
        return

    df_all.index = pd.to_datetime(df_all.index)
    start = df_all.index.min()
    end   = df_all.index.max()

    print(f"\n{'═'*55}")
    print(f"  Rolling Walk-Forward — {symbol}")
    print(f"  Train: {train_months}mo  |  Test: {test_months}mo  |  Step: {test_months}mo")
    print(f"{'═'*55}")

    results = []
    cur = start + pd.DateOffset(months=train_months)
    while cur + pd.DateOffset(months=test_months) <= end:
        train_start = cur - pd.DateOffset(months=train_months)
        test_end    = cur + pd.DateOffset(months=test_months)
        df_tr = df_all.loc[train_start:cur]
        df_te = df_all.loc[cur:test_end]
        if len(df_tr) < 100 or len(df_te) < 10:
            cur += pd.DateOffset(months=test_months)
            continue
        try:
            m = run_backtest(df_te, initial_balance=INITIAL_CAPITAL)
            results.append({
                "period": f"{cur.strftime('%Y-%m')} → {test_end.strftime('%Y-%m')}",
                **m
            })
            print(
                f"  {results[-1]['period']}  "
                f"ret={m['total_return']:>+.1%}  "
                f"sharpe={m['sharpe']:>5.2f}  "
                f"dd={m['max_drawdown']:>.1%}  "
                f"trades={m['num_trades']}"
            )
        except Exception as e:
            logger.warning(f"Window {cur} failed: {e}")
        cur += pd.DateOffset(months=test_months)

    if results:
        avg_ret    = sum(r["total_return"]  for r in results) / len(results)
        avg_sharpe = sum(r["sharpe"]        for r in results) / len(results)
        avg_dd     = sum(r["max_drawdown"]  for r in results) / len(results)
        win_windows = sum(1 for r in results if r["total_return"] > 0)
        print(f"\n  {'─'*51}")
        print(f"  Windows: {len(results)}   Profitable: {win_windows}/{len(results)}")
        print(f"  Avg Return: {avg_ret:>+.2%}   Avg Sharpe: {avg_sharpe:.2f}   Avg DD: {avg_dd:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward backtest validation")
    parser.add_argument("--symbol", default="SPY", help="Symbol to validate (default: SPY)")
    parser.add_argument("--rolling",       action="store_true",
                        help="Rolling walk-forward instead of single train/test split")
    parser.add_argument("--train-months",  type=int, default=12)
    parser.add_argument("--test-months",   type=int, default=3)
    args = parser.parse_args()

    if args.rolling:
        run_rolling(args.symbol, args.train_months, args.test_months)
    else:
        run_simple(args.symbol)
