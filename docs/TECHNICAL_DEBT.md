# TradeGenius AI — Technical Debt

Last updated: 2026-06-27

Items are rated by effort (S/M/L/XL) and urgency (High/Medium/Low).
High-urgency items should be resolved before any real-money deployment.

## Active Debt

| ID | Item | Effort | Urgency | Notes |
|----|------|--------|---------|-------|
| TD-001 | `bot/main.py` is 556 lines (over 500-line limit) | S | Medium | SPY pre-fetch setup could move to `_main_market.py`; deferred due to risk of breaking the trading loop |
| TD-002 | Missing unit test files for 8 bot modules | L | High | `_main_cycle.py`, `_main_db.py`, `_main_market.py`, `_main_positions.py`, `_main_runner.py`, `_main_signals.py`, `api_guard.py`, `signal_gate.py` all lack `tests/test_<module>.py`. Stubs without coverage are disingenuous — needs meaningful integration tests |
| TD-003 | `bot/monitor/` has 8 legacy private modules (`_dashboard_*.py`) | M | Low | Pre-refactor modules kept for backward compatibility. Should be deleted once confirmed not imported anywhere |
| TD-004 | DuckDB analytics not yet surfaced in dashboard | M | Low | `AnalyticsService.get_sharpe_ratio()` and `get_max_drawdown()` compute from DuckDB but the scorecard panel reads directly from SQLite. Should switch to analytics service for consistency |
| TD-005 | `bot/strategy/rl_agent.py` PPO model rarely invoked | M | Low | RL agent used for position sizing but Kelly criterion is primary sizer. Unclear if RL adds value — needs ablation study |
| TD-006 | No database migration tooling | M | Medium | Schema changes require manual SQL. If trades.db schema evolves, existing data needs migration script. Consider Alembic or manual versioned migrations |
| TD-007 | `backtest/engine.py` not connected to live parameter changes | M | Medium | Backtest uses hardcoded params; should read from `config.py` so a param change triggers a backtest validation |
| TD-008 | `tests/measure_performance.py` needs periodic re-run | S | Low | NFR numbers in `docs/NFR.md` are point-in-time. Re-run quarterly or after any infrastructure change |

## Resolved Debt

| ID | Item | Resolved | How |
|----|------|----------|-----|
| TD-009 | `dashboard/app.py` was 3979 lines (god file) | 2026-06-14 | SPEC 52: split into 15 component modules; app.py now 322 lines |
| TD-010 | `dashboard/components/history.py` was 694 lines | 2026-06-27 | Split: news → `news.py`, recommendation history → `recommendation_history.py` |
| TD-011 | Silent exceptions (`except: pass`) in 5 bot files | 2026-06-14 | SPEC 51: all upgraded to `log_exception()` or `logger.debug()` |
| TD-012 | `log_exception()` called with wrong signature in 5 places | 2026-06-14 | SPEC 55: corrected to 3-arg form; test coverage added |
| TD-013 | All render functions lacked error handling | 2026-06-14 | `@safe_render` decorator wraps all render_* functions |

## Debt Acceptance Policy

Debt is accepted when:
- Risk is explicitly documented here
- An owner and target resolution date are agreed
- The item does not affect financial safety (risk limits, PDT, stop-loss)

Items that affect real-money safety (TD-002 in particular) must be resolved before
any real-money deployment, regardless of timeline.
