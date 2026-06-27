# TradeGenius AI — Risk Register

Last updated: 2026-06-27

Risks are rated by Likelihood (1–5) × Impact (1–5) = Score.
Score ≥ 15 = High, 8–14 = Medium, < 8 = Low.

## Active Risks

| ID | Risk | L | I | Score | Category | Mitigation | Owner |
|----|------|---|---|-------|----------|------------|-------|
| R-001 | Model overfits training data, degrades on live market | 3 | 5 | 15 | ML | Walk-forward backtesting; pre-market retraining monthly; win-rate dashboard | ksri77 |
| R-002 | Alpaca API outage during market hours blocks all orders | 2 | 4 | 8 | Infra | Retry logic in `alpaca_client.py`; HALT_TRADING env override; Telegram alert on failure | ksri77 |
| R-003 | GitHub Actions cron job misses market open | 2 | 3 | 6 | Infra | `workflow_dispatch` backup; 5-min loop means next cycle catches up | ksri77 |
| R-004 | VIX spike triggers halt but positions still open | 3 | 3 | 9 | Risk | VIX ≥ 28 halts new buys; existing exits still run each cycle | ksri77 |
| R-005 | PDT rule triggered on account < $25K | 2 | 4 | 8 | Regulatory | `PDT_MAX_DAY_TRADES=3` gate in `risk_manager.py`; tested in `test_risk_manager.py` | ksri77 |
| R-006 | HuggingFace Spaces restart loses in-memory cache | 3 | 2 | 6 | Infra | Cache rebuild on next DB read; 60-second refresh cycle | ksri77 |
| R-007 | SQLite file corruption (rare, concurrent writes) | 1 | 5 | 5 | Data | WAL mode; `test_sqlite_threading.py`; daily snapshot to `portfolio_snapshots` table | ksri77 |
| R-008 | yfinance rate-limit or API change breaks data ingestion | 3 | 3 | 9 | Data | Module-level 1-hour cache reduces call frequency; fallback to Alpaca bars | ksri77 |
| R-009 | FRED API unavailable, macro signals missing | 2 | 2 | 4 | Data | 4-hour DB-backed cache; graceful degradation (macro_score=0, no halt) | ksri77 |
| R-010 | NewsAPI / Reddit unavailable, sentiment missing | 3 | 2 | 6 | Data | Ensemble falls back to XGB+LSTM only; sentiment weight reduced | ksri77 |
| R-011 | Runaway loss beyond daily limit (gap-down open) | 2 | 5 | 10 | Financial | Gap-down floor exit fires first each cycle; ATR stop + 4% flat stop | ksri77 |
| R-012 | Dashboard leaks strategy logic publicly | 1 | 3 | 3 | Security | Dashboard shows signals/rationale; model weights and config stay server-side | ksri77 |

## Resolved Risks

| ID | Risk | Resolution Date | How Resolved |
|----|------|-----------------|--------------|
| R-013 | `log_exception()` called with wrong arg count silently discarded errors | 2026-06-14 | Fixed 5 call sites; SPEC 55 tests verify signature |
| R-014 | Extreme concentration, catastrophic loss, and large profit signals suppressed in sell score | 2026-06-14 | Calibrated thresholds in `get_sell_analysis()`; 77 tests added |

## Risk Appetite

Financial: max 5% daily loss, 10% weekly, 12% drawdown from high — hard limits, not targets.
Operational: bot may miss one full trading cycle (5 min) without intervention required.
Data: stale macro/sentiment data (< 4 hours) is acceptable; stale price data (> 15 min) triggers a hold.
