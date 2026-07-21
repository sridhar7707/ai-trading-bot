#!/usr/bin/env python3
"""Investment engine evaluation framework.

Computes per-component precision/recall/calibration, portfolio-level Sharpe/
Sortino/drawdown/alpha, and ablation analysis showing the effect of disabling
each component individually.

Usage
-----
    python scripts/evaluate_engine.py [--days 180] [--db PATH]

The report is printed to stdout; redirect to a file for archival.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure box-drawing characters render on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd

from bot.eval.loader import (
    load_completed_trades,
    load_equity_curve,
    load_signal_log,
    fetch_spy_daily,
)
from bot.eval.metrics import (
    alpha_vs_spy,
    calibration_buckets,
    precision_recall,
    summary,
)
from bot.eval.ablation import gate_analysis, simulate, SCENARIOS
from bot.strategy.ensemble import WEIGHTS, BUY_THRESHOLD
from config import ENTRY_REGIMES, TRADE_DB_PATH


# ── Formatting ────────────────────────────────────────────────────────────────

W = 60


def _sep(char: str = "─") -> None:
    print(f"  {char * (W - 2)}")


def _section(title: str) -> None:
    print(f"\n{'═' * W}")
    print(f"  {title}")
    print(f"{'═' * W}")


def _sub(title: str) -> None:
    print(f"\n  ▸ {title}")
    _sep()


def _row(label: str, value: str, width: int = 30) -> None:
    print(f"    {label:<{width}}{value}")


def _pct(v: float, sign: bool = False) -> str:
    fmt = f"{v * 100:+.1f}%" if sign else f"{v * 100:.1f}%"
    return fmt


def _print_metrics_row(label: str, m: dict, alpha: float = 0.0) -> None:
    n = m["n_trades"]
    if n == 0:
        print(f"  {label:<30}  (no trades)")
        return
    sortino_str = f"{m['sortino']:.2f}" if m["sortino"] != float("inf") else "inf "
    print(
        f"  {label:<30}"
        f"  n={n:>3}"
        f"  win={_pct(m['win_rate']):>6}"
        f"  avg={_pct(m['avg_return'], sign=True):>7}"
        f"  dd={_pct(m['max_drawdown'], sign=True):>7}"
        f"  shp={m['sharpe']:>5.2f}"
        f"  srt={sortino_str:>5}"
        f"  pf={m['profit_factor']:>5.2f}"
        f"  hold={m['avg_holding_days']:>4.1f}d"
        f"  α={_pct(alpha / 252 if alpha else 0.0, sign=True):>7}"
    )


# ── Report sections ───────────────────────────────────────────────────────────

def _section_baseline(trades: pd.DataFrame, equity: pd.Series, spy: pd.Series) -> dict:
    _section("1 · BASELINE PORTFOLIO METRICS")
    pnl  = trades["pnl_pct"]
    hold = trades["holding_days"]
    m    = summary(pnl, hold, equity)
    al   = alpha_vs_spy(trades["sell_ts"], pnl, spy)

    _row("Completed trades",    str(m["n_trades"]))
    _row("Win rate",            _pct(m["win_rate"]))
    _row("Avg return / trade",  _pct(m["avg_return"], sign=True))
    _row("Max drawdown",        _pct(m["max_drawdown"], sign=True))
    _row("Sharpe ratio",        f"{m['sharpe']:.3f}")
    _row("Sortino ratio",       f"{m['sortino']:.3f}" if m["sortino"] != float("inf") else "inf")
    _row("Profit factor",       f"{m['profit_factor']:.3f}")
    _row("Avg holding period",  f"{m['avg_holding_days']:.1f} days")
    _row("Alpha vs SPY (ann.)", _pct(al, sign=True))

    if "regime" in trades.columns:
        print()
        _row("Entry regimes allowed", ", ".join(sorted(ENTRY_REGIMES)))

    return m


def _section_components(trades: pd.DataFrame) -> None:
    _section("2 · PER-COMPONENT PRECISION / RECALL / CALIBRATION")

    pnl = trades["pnl_pct"]

    components = [
        ("XGBoost",      "xgb_prob",       WEIGHTS["xgb"],       0.55,
         [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 1.01]),
        ("LSTM",         "lstm_prob",       WEIGHTS["lstm"],      0.50,
         [0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 1.01]),
        ("FinBERT",      "sentiment_score", WEIGHTS["sentiment"], 0.0,
         [-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.01]),
        ("Macro Score",  "macro_score",     WEIGHTS["macro"],     0.50,
         [0.30, 0.40, 0.50, 0.60, 0.70, 1.01]),
        ("Ensemble",     "ensemble_score",  1.0,                  BUY_THRESHOLD,
         [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 1.01]),
    ]

    for label, col, weight, threshold, edges in components:
        if col not in trades.columns:
            continue
        _sub(f"{label}  (weight={weight:.2f}, threshold≥{threshold})")
        scores = trades[col]
        pr = precision_recall(scores, pnl, score_threshold=threshold)
        _row("Precision (≥thr → profitable)",
             f"{pr['precision']*100:.1f}%  TP={pr['tp']} FP={pr['fp']}")
        _row("Recall (profitable → ≥thr)",
             f"{pr['recall']*100:.1f}%  FN={pr['fn']}")
        _row("F1 score",               f"{pr['f1']:.3f}")
        _row("Signals above threshold", str(pr["n_above_threshold"]))
        _row("Total profitable trades", str(pr["n_profitable"]))

        cal = calibration_buckets(scores, pnl, edges)
        print(f"\n    {'Score bucket':<14}  {'n':>4}  {'Win rate':>8}  {'Avg return':>10}")
        _sep("·")
        for b in cal:
            if b["count"] > 0:
                bar = "█" * min(int(b["win_rate"] * 20), 20)
                print(
                    f"    {b['range']:<14}  {b['count']:>4}  "
                    f"{b['win_rate']*100:>7.1f}%  "
                    f"{b['avg_return']*100:>+9.2f}%  {bar}"
                )


def _section_regime(trades: pd.DataFrame, equity: pd.Series, spy: pd.Series) -> None:
    if "regime" not in trades.columns:
        return
    _section("3 · REGIME BREAKDOWN")
    print(f"  {'Regime':<22}  {'n':>3}  {'win':>6}  {'avg':>7}  {'sharpe':>6}  {'sortino':>7}  {'pf':>5}")
    _sep()
    for regime, grp in sorted(trades.groupby("regime")):
        m  = summary(grp["pnl_pct"], grp["holding_days"])
        sr = f"{m['sortino']:.2f}" if m["sortino"] != float("inf") else " inf"
        in_entry = "✓" if regime in ENTRY_REGIMES else "✗"
        print(
            f"  {in_entry} {str(regime):<20}  {m['n_trades']:>3}"
            f"  {_pct(m['win_rate']):>6}"
            f"  {_pct(m['avg_return'], sign=True):>7}"
            f"  {m['sharpe']:>6.2f}"
            f"  {sr:>7}"
            f"  {m['profit_factor']:>5.2f}"
        )


def _section_ablation(
    trades: pd.DataFrame,
    equity: pd.Series,
    spy: pd.Series,
    signal_log: pd.DataFrame,
) -> None:
    _section("4 · ABLATION ANALYSIS  (disable one component at a time)")

    # Gate-level summary from signal_log
    if not signal_log.empty:
        ga = gate_analysis(signal_log)
        if ga:
            _sub("Hard-gate counterfactual (from signal_log)")
            _row("Total BUY-intent signals",    str(ga["total_buy_signals"]))
            _row("Blocked by regime gate",
                 f"{ga['blocked_by_regime']}  ({ga['pct_blocked_by_regime']*100:.1f}% of signals)")

    # Per-scenario metrics table
    print()
    hdr = (
        f"  {'Scenario':<28}  {'n':>3}"
        f"  {'win':>6}  {'avg':>7}  {'dd':>7}"
        f"  {'shp':>5}  {'srt':>5}  {'pf':>5}"
        f"  {'hold':>6}  {'α ann.':>7}"
    )
    print(hdr)
    _sep()

    # Ablation baseline: apply regime gate so it matches the current live config.
    # Section 1 shows the full historical record; ablation shows forward-looking
    # impact under the current ENTRY_REGIMES setting.
    _base_df = simulate(trades, disabled=set(), disable_regime_gate=False)
    base_m   = summary(_base_df["pnl_pct"], _base_df["holding_days"])
    base_a   = alpha_vs_spy(_base_df["sell_ts"], _base_df["pnl_pct"], spy)

    print(f"  (Note: ablation baseline uses ENTRY_REGIMES filter; section 1 shows full history)")

    for name, cfg in SCENARIOS.items():
        df = simulate(
            trades,
            disabled=cfg["disabled"],
            disable_regime_gate=cfg["regime"],
            flat_sizing=cfg["flat"],
        )
        if df.empty:
            print(f"  {name:<28}  (no trades)")
            continue
        m  = summary(df["pnl_pct"], df["holding_days"])
        al = alpha_vs_spy(df["sell_ts"], df["pnl_pct"], spy)
        sr = f"{m['sortino']:.2f}" if m["sortino"] != float("inf") else " inf"
        marker = "  ←" if name == "baseline" else ""
        print(
            f"  {name:<28}  {m['n_trades']:>3}"
            f"  {_pct(m['win_rate']):>6}"
            f"  {_pct(m['avg_return'], sign=True):>7}"
            f"  {_pct(m['max_drawdown'], sign=True):>7}"
            f"  {m['sharpe']:>5.2f}"
            f"  {sr:>5}"
            f"  {m['profit_factor']:>5.2f}"
            f"  {m['avg_holding_days']:>5.1f}d"
            f"  {_pct(al / 252 if al else 0.0, sign=True):>7}"
            f"{marker}"
        )


def _section_impact(trades: pd.DataFrame, spy: pd.Series) -> None:
    _section("5 · COMPONENT PORTFOLIO IMPACT  (delta vs ablation baseline)")

    # Use regime-filtered baseline to match section 4
    _base_df = simulate(trades, disabled=set(), disable_regime_gate=False)
    base_m = summary(_base_df["pnl_pct"], _base_df["holding_days"])
    base_a = alpha_vs_spy(_base_df["sell_ts"], _base_df["pnl_pct"], spy)

    print(
        f"  {'Scenario':<28}  {'Δ trades':>8}  {'Δ win rate':>10}"
        f"  {'Δ sharpe':>8}  {'Δ avg return':>12}  {'Δ α ann.':>9}"
    )
    _sep()
    print(f"  {'baseline':<28}  {'0':>8}  {'—':>10}  {'—':>8}  {'—':>12}  {'—':>9}")

    for name, cfg in SCENARIOS.items():
        if name == "baseline":
            continue
        df = simulate(trades, disabled=cfg["disabled"],
                      disable_regime_gate=cfg["regime"])
        m  = summary(
            df["pnl_pct"]      if not df.empty else pd.Series(dtype=float),
            df["holding_days"] if not df.empty else pd.Series(dtype=float),
        )
        al = alpha_vs_spy(df["sell_ts"], df["pnl_pct"], spy) if not df.empty else 0.0
        dn = m["n_trades"] - base_m["n_trades"]
        dw = (m["win_rate"]   - base_m["win_rate"])   * 100
        ds =  m["sharpe"]     - base_m["sharpe"]
        dr = (m["avg_return"] - base_m["avg_return"]) * 100
        da = (al - base_a) / 252 * 100 if al and base_a else 0.0

        s = lambda x: "+" if x >= 0 else ""
        print(
            f"  {name:<28}"
            f"  {s(dn)}{dn:>7}"
            f"  {s(dw)}{dw:>8.1f}pp"
            f"  {s(ds)}{ds:>8.2f}"
            f"  {s(dr)}{dr:>+10.2f}pp"
            f"  {s(da)}{da:>+7.2f}pp"
        )

    print()
    print("  Interpretation:")
    print("    Δ trades < 0  → component is FILTERING OUT trades when active")
    print("    Δ trades > 0  → component ADDS trades when active (should not happen")
    print("                     for filter-ablation; indicates regime-gate scenario)")
    print("    Δ win rate < 0 → removing the component improves win rate (reconsider)")
    print("    Δ sharpe  < 0 → removing the component improves Sharpe  (reconsider)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate investment engine components and print a full report."
    )
    parser.add_argument(
        "--days", type=int, default=180,
        help="Look-back window in calendar days (default: 180)",
    )
    parser.add_argument(
        "--db", type=str, default=TRADE_DB_PATH,
        help="Path to trades.db SQLite file",
    )
    args = parser.parse_args()

    print(f"\nLoading data from {args.db}  (last {args.days} days)…")
    trades     = load_completed_trades(args.db, args.days)
    equity     = load_equity_curve(args.db, args.days)
    spy        = fetch_spy_daily(args.days)
    signal_log = load_signal_log(args.db, args.days)

    if trades.empty:
        print("No completed trades in this date range. Run the bot first.")
        sys.exit(0)

    _section_baseline(trades, equity, spy)
    _section_components(trades)
    _section_regime(trades, equity, spy)
    _section_ablation(trades, equity, spy, signal_log)
    _section_impact(trades, spy)

    print(f"\n{'═' * W}")
    print(f"  End of report — {len(trades)} trades analysed")
    print(f"{'═' * W}\n")


if __name__ == "__main__":
    main()
