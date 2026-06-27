# TradeGenius AI — Release History

Last updated: 2026-06-27

Versions follow the requirements tracker version (`_ver(state)` in `requirements_tracker.py`).
Each release corresponds to a batch of completed SPECs.

---

## v1.3 — Dashboard Completion (2026-06-26)

**SPECs completed**: SPEC 9, SPEC 38, SPEC 39

- **SPEC 38**: Rebalance Suggestions panel — grouped reduce/add action plan with net cash
  change and sector shift. `dashboard/components/rebalance.py: render_rebalance_suggestions()`
- **SPEC 39**: Paper Trading Scorecard — bot return vs SPY/QQQ, Sharpe, max drawdown, win
  rate, AI follow rate. `dashboard/components/models.py: render_paper_trading_scorecard()`
- **SPEC 9**: Living Requirements Tracker marked complete; all 53 SPECs now tracked

### Cleanup (2026-06-27)
- Split `history.py` (694 → 418 lines) into `news.py` (161) + `recommendation_history.py` (128)
- Deleted 8 stale req/UI snapshots
- All 535 tests passing; ui_tester 0 FAIL 0 WARN

---

## v1.2 — Critical Tests + Exception Hardening (2026-06-14)

**SPECs completed**: SPEC 50, SPEC 51, SPEC 52, SPEC 53, SPEC 54, SPEC 55

- **SPEC 55**: 77 unit tests for recommendation engine; fixed 4 bugs in `get_sell_analysis()`
  and `get_portfolio_health()`; fixed `log_exception()` signature in 5 places
- **SPEC 54**: Analytics exception hardening — `save_daily_snapshot()`, `get_sharpe_ratio()`,
  `get_max_drawdown()` all log on failure; `check_health()` added and wired into startup
- **SPEC 52**: God-file split — `app.py` 3979 → 322 lines; 15 component modules created
- **SPEC 53**: View model pattern — `viewmodels.py`, `builders.py`; 20 viewmodel tests
- **SPEC 51**: Exception hardening — `@safe_render`, `@timed`, `log_exception()` upgrades
- **SPEC 50**: DuckDB analytics layer — schema, repository, service, 6 integration tests

---

## v1.1 — Five Investor Questions + UX Polish (2026-06-13/14)

**SPECs completed**: SPEC 27–49

- **SPEC 27–31**: 5 new dashboard panels (portfolio actions, AI committee, symbol detail,
  position sizing, today's actions)
- **SPEC 32**: Recommendation engine — 5 shared helpers as single source of truth
- **SPEC 33–37**: Health hero, rebuilt actions, sell analysis, why panel, position sizing v2
- **SPEC 40**: Design system v1.0 — tokens, component helpers, mobile CSS, GROUP 7 test
- **SPEC 41–45**: Positions redesign, tab nav fix, HOLD visual fix, badge/empty-state audit
- **SPEC 46–49**: Decision center, rebalance panel, layout reorganization, consistency checks

---

## v1.0 — Core Platform (2026-06-13)

**SPECs completed**: SPEC 1–26

### Frontend (SPEC 1–9)
- Dashboard hero with portfolio health score (SPEC 1)
- Rich Telegram alerts with SHAP drivers (SPEC 2)
- Since Yesterday comparison panel (SPEC 3)
- AI Action Column per position (SPEC 4)
- Daily summary alert at 4:05 PM ET (SPEC 5)
- Portfolio performance periods 1D–All Time (SPEC 6A)
- Per-stock yfinance performance columns (SPEC 6B)
- SVG sparkline charts (SPEC 6C)
- UI/UX test suite — 14 files (SPEC 7)
- UI change log tracker (SPEC 8)
- Requirements tracker (SPEC 9)

### Backend (SPEC 10–26)
- 5-min GitHub Actions trading loop with market-hours detection (SPEC 10)
- Pre-market screener with RL agent ranking (SPEC 11)
- Technical feature engineering — ATR, RSI, EMA, volume ratio (SPEC 12)
- Market regime classifier (SPEC 13)
- XGBoost signal model with SHAP (SPEC 14)
- LSTM sequence model (SPEC 15)
- FinBERT sentiment + Reddit WSB (SPEC 16)
- FRED macro signals + VIX halt (SPEC 17)
- Weighted ensemble signal (SPEC 18)
- 10-gate entry filter suite (SPEC 19)
- 7-step exit logic suite (SPEC 20)
- Risk manager — daily/weekly limits, PDT, drawdown CB (SPEC 21)
- Alpaca execution engine (SPEC 22)
- SQLite 8-table schema (SPEC 23)
- HuggingFace DB bridge (SPEC 24)
- Telegram alert system (SPEC 25)
- Position reconciliation at startup (SPEC 26)

---

## Upcoming

No planned releases. All 53 SPECs complete. Next work will be driven by:
- Real-money deployment gate (win rate, drawdown targets — see `docs/SUCCESS_METRICS.md`)
- Technical debt resolution (see `docs/TECHNICAL_DEBT.md`)
- Post-paper-trading model retrain with real performance data
