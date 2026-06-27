# TradeGenius AI — Success Metrics

Last updated: 2026-06-27

All metrics are computed from live paper-trading data in `trades.db`.
The Paper Trading Scorecard panel (`render_paper_trading_scorecard()`) shows live values.
Run `python tests/measure_performance.py` to capture a point-in-time snapshot.

## Primary Metrics (must all pass before real-money deployment)

| Metric | Target | Current | Source |
|--------|--------|---------|--------|
| Win Rate | ≥ 60% | See scorecard | `trades` table: closed BUY→SELL pairs |
| Total Return vs SPY | Beat SPY by ≥ 5 pp | See scorecard | yfinance SPY price since first trade |
| Total Return vs QQQ | Beat QQQ by ≥ 0 pp | See scorecard | yfinance QQQ price since first trade |
| Max Drawdown | ≤ 12% | See scorecard | `portfolio_snapshots`: peak-to-trough |
| Sharpe Ratio | ≥ 1.0 | See scorecard | `analytics_service.get_sharpe_ratio()` |
| AI Follow Rate | ≥ 70% | See scorecard | BUY recs executed within same day (30d window) |

## Secondary Metrics (operational health)

| Metric | Target | How to Check |
|--------|--------|--------------|
| Average hold days | 3–25 days | `trades` table: avg(days between BUY and SELL) |
| Average P&L per trade | > 0% | `trades` table: avg(pnl_pct) |
| Largest single loss | < -8% | `trades` table: min(pnl_pct) |
| Daily loss limit breaches | 0 | `risk_state` table: `daily_loss_exceeded` flag |
| PDT trades used | < 3 / 5 days | `risk_state` table: `day_trades_today` |
| Positions at max (8) | < 20% of days | `portfolio_snapshots`: count open_positions = 8 |
| Cash reserve maintained | ≥ 10% at end of each day | `portfolio_snapshots`: cash / portfolio_value |

## Model Quality Metrics (checked at each retraining)

| Metric | Target | Source |
|--------|--------|--------|
| XGBoost validation AUC | ≥ 0.65 | `scripts/confidence_check.py` |
| Backtest win rate (last 6M) | ≥ 55% | `python scripts/backtest_gate.py` |
| Backtest Sharpe | ≥ 0.8 | `backtest/metrics.py` |

## Non-Functional Metrics

See `docs/NFR.md` for latency, memory, and reliability targets with real measured numbers.

## Definition of Done

A feature is "Done" when:
1. Code merged to `main` with no arch-review WARN or BLOCK
2. `python tests/ui_tester.py` → 0 FAIL 0 WARN
3. `pytest tests/ -q` → all pass
4. `docs/REQUIREMENTS.md` updated (run `python tests/requirements_tracker.py`)
5. At least one happy-path test exists for any new function
6. For risk/execution changes: explicit approval from owner after reviewing config.py diff
