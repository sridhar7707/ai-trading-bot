# TradeGenius — Living Requirements Document

Auto-generated and auto-updated.
Last updated: 2026-07-12 20:32:26
Version: 1.3.86

---

## PROJECT OVERVIEW
Name: TradeGenius AI
Type: AI Portfolio Copilot
Stack: Python, Gradio, Alpaca, HuggingFace Spaces
Purpose: Long-term and swing investment guidance with AI recommendations,
         portfolio health monitoring, and automated execution with full explainability.

---

## FEATURE STATUS

### CORE FEATURES
| Feature | Status | Spec | Last Updated | Notes |
|---------|--------|------|--------------|-------|
| Dashboard Hero & Health Cards | ✅ Complete | SPEC 1 | 2026-06-13 | Portfolio value, P&L, cash, VIX, health score cards |
| Portfolio Health Score | ✅ Complete | SPEC 1 | 2026-06-13 | Score 0-100 from VIX/cash/concentration/drawdown |
| Rich Telegram BUY/SELL Alerts | ✅ Complete | SPEC 2 | 2026-06-13 | Confidence %, SHAP drivers, sector %, cash after |
| Since Yesterday Panel | ✅ Complete | SPEC 3 | 2026-06-13 | render_whats_changed(): ensemble/regime/sentiment delta |
| AI Action Column (HOLD/TRIM/EXIT) | ✅ Complete | SPEC 4 | 2026-06-13 | Per-position sell score 0-100, sub-row with top reasons |
| Daily Summary Telegram Alert | ✅ Complete | SPEC 5 | 2026-06-13 | 4:05pm ET: portfolio value, best/worst trade, health score |
| Portfolio Performance Periods | ✅ Complete | SPEC 6A | 2026-06-13 | 1D/1W/1M/3M/YTD/1Y/All Time tabs with headline stats |
| Per-Stock Performance Columns | ✅ Complete | SPEC 6B | 2026-06-13 | yfinance 1D/1W/1M/1Y/All Time % per position (1h cache) |
| Sparkline Charts | ✅ Complete | SPEC 6C | 2026-06-13 | 80×32 SVG 30-day trend column in positions table |
| UI/UX Test Suite | ✅ Complete | SPEC 7 | 2026-06-13 | 14 test files in tests/ including test_dashboard_render.py |
| UI Change Log | ✅ Complete | SPEC 8 | 2026-06-13 | tests/ui_changelog.py; 20 render_* components tracked |
| Living Requirements Tracker | ✅ Complete | SPEC 9 | 2026-06-26 | tests/requirements_tracker.py — scan, --status, --bug, --fix, --complete, --dry-run. SPEC 38/39 added. |
| Portfolio Actions Panel | ✅ Complete | SPEC 27 | 2026-06-13 | render_portfolio_actions(): HOLD/WATCH/TRIM/EXIT per open position from ensemble score + sell score; Dashboard tab above market intelligence row |
| Position Sizing Recommendation | ✅ Complete | SPEC 28 | 2026-06-13 | render_position_sizing(): conviction-based target allocation (3-12%) from ensemble score; Portfolio tab above positions table |
| Today's Actions Summary | ✅ Complete | SPEC 29 | 2026-06-13 | render_todays_actions(): today BUY/SELL timeline with time, price, P&L, reason; Dashboard tab between hero and whats_changed |
| AI Investment Committee | ✅ Complete | SPEC 30 | 2026-06-13 | render_ai_committee(): XGBoost/LSTM/Sentiment vote chips per open position with majority verdict; Dashboard tab right column above watchlist |
| Symbol Detail Action Card | ✅ Complete | SPEC 31 | 2026-06-13 | Enhancement to render_symbol_detail(): action card (HOLD/WATCH/TRIM/EXIT), conviction bar, sizing guidance injected at top of detail card |
| Recommendation Engine | ✅ Complete | SPEC 32 | 2026-06-13 | bot/core/recommendation_engine.py — 5 shared helpers implemented and tested: get_portfolio_action, get_position_sizing, get_sell_analysis, get_recommendation_explanation, get_portfolio_health. Single source of truth for all panels. |
| Portfolio Health Hero Panel | ✅ Complete | SPEC 33 | 2026-06-13 | render_portfolio_health_hero() — replaces render_dashboard_hero() in Dashboard tab. 5-component health score (Risk/Diversification/Cash/Momentum/Quality) + grade (A/B+/B/C/D) + biggest risk callout. |
| Today's Actions Panel v2 | ✅ Complete | SPEC 34 | 2026-06-13 | render_todays_actions() rebuilt — sorted action list (EXIT/SELL/TRIM/ADD/BUY/WATCH/HOLD) with sizing hint from get_position_sizing(). Calls get_portfolio_action() per position. |
| Sell Analysis Panel | ✅ Complete | SPEC 35 | 2026-06-13 | render_sell_analysis() — sell score 0-100 table with HOLD/WATCH/TRIM/SELL/EXIT signal, P&L, weight, top reason, trim action. Portfolio tab. Calls get_sell_analysis(). |
| Why Panel | ✅ Complete | SPEC 36 | 2026-06-13 | Why panel embedded in render_symbol_detail() — action badge + conviction bar + bullish/bearish factors from get_recommendation_explanation() + sizing from get_position_sizing(). |
| Position Sizing Panel v2 | ✅ Complete | SPEC 37 | 2026-06-13 | render_position_sizing_panel() — current→target weight with dual-bar visualisation, delta $+shares. Portfolio tab. Replaces render_position_sizing(). |
| Design System v1.0 | ✅ Complete | SPEC 40 | 2026-06-13 | Permanent design system: 4 font sizes (36/20/15/11px), 3 text colors, 7 action colors with bg fills, 13 component helpers (_card/_label/_hero_value/_section_title/_action_badge/_symbol/_confidence_bar/_metric_row/_progress_bar/_divider/_empty_state/_action_row/_table), mobile CSS (390px), GROUP 7 compliance test (0 FAIL). docs/DESIGN_SYSTEM.md is single source of truth. |
| Positions Table Redesign | ✅ Complete | SPEC 41 | 2026-06-13 | render_positions() columns changed to Symbol|Action|Weight|Target|Confidence|P&L. Removed: Shares, Invested, Current Value, Cost Basis. Action from recommendation engine. Visual hierarchy by urgency (EXIT/SELL red row, TRIM amber, BUY green, HOLD dimmed). |
| Tab Navigation Visibility Fix | ✅ Complete | SPEC 42 | 2026-06-13 | Fixed tab visibility on HuggingFace Spaces: (1) triple CSS selectors with TEXT1 at 0.6 opacity (inactive) and 1.0 (active/hover), ACTION_BUY green underline on selected; (2) custom gr.themes.Base().set() theme replacing bare gr.themes.Base(); (3) JS MutationObserver fallback that enforces white text on all tab buttons and updates on class change. |
| HOLD Visual Fix | ✅ Complete | SPEC 43 | 2026-06-14 | Fix HOLD/WATCH rows in _action_row(): change sym_color from TEXT2 (#b0b7c3) to TEXT1 (#ffffff) so symbol is always readable. ACTION_HOLD badge (#64b5f6) already correct — only the row symbol text needs fixing. |
| Badge Standardization Audit | ✅ Complete | SPEC 44 | 2026-06-14 | Audit all render_* functions: replace any inline badge HTML (inline-block span with BUY/SELL/HOLD/TRIM/EXIT text) with _action_badge() calls. All _badge() shim calls already delegate to _action_badge(). Target: zero inline badge HTML in any render function. |
| Empty State Standardization Audit | ✅ Complete | SPEC 45 | 2026-06-14 | Audit all render_* functions: replace custom empty state HTML (f-strings with 'No open positions', 'No trades', 'No signal', 'No data', etc.) with _empty_state(icon, title, subtitle) calls. Target: zero custom empty state HTML in any render function. |
| Decision Center Panel | ✅ Complete | SPEC 46 | 2026-06-14 | render_decision_center(): consolidates render_portfolio_actions + render_sell_analysis + render_position_sizing_panel into one unified panel. Per-position row: action badge + sell_score/100 + current%→target% + dollar amount + top reasons (✗/✓). Sorted EXIT→SELL→TRIM→ADD/BUY→WATCH→HOLD. Goes in Portfolio tab. Old render_portfolio_actions/sell_analysis/position_sizing_panel become dead (called only from decision_center). |
| Rebalance Panel | ✅ Complete | SPEC 47 | 2026-06-14 | render_rebalance(): new panel. Current vs Target allocation table with delta column + action badge. Cash row shows direction change. Summary footer: net $ to rebalance, cash after, sector risk before/after, health score before/after. _simulate_health_after_rebalance(d, sizing_results) helper used for before/after comparison. Portfolio tab below decision_center. |
| Dashboard Layout Reorganization | ✅ Complete | SPEC 48 | 2026-06-14 | Rebuild Gradio layout: Dashboard tab = EXACTLY 5 panels (health_hero → todays_actions → ai_recommendation → risk_panel → whats_changed). Signals tab = add mkt_intel + watchlist after existing timeline/signals. Portfolio tab = add committee + decision_center + rebalance before positions. Update all timer.tick() calls to include new outputs. |
| Consistency Checks Pass | ✅ Complete | SPEC 49 | 2026-06-14 | All render_* functions pass: (1) section titles ≤4 words, (2) only design system hex colors, (3) only 36/20/15/11px font sizes, (4) _symbol() for all ticker symbols, (5) _confidence_bar() for all confidence displays. Add dead code comments to render_portfolio_actions/sell_analysis/position_sizing_panel (called internally by decision_center). Add superseded comment to render_dashboard_hero. |
| Dashboard App.py Modular Refactor | ✅ Complete | SPEC 52 | 2026-06-14 | Split 3979-line dashboard/app.py into focused modules: design_system.py (constants+helpers), data.py (cache+DB layer), charts.py (Plotly chart functions), components/ (12 component files grouped by domain). App.py becomes 300-500 line wiring-only file. Zero logic changes â€” pure file split. |
| Dashboard View Model Pattern | ✅ Complete | SPEC 53 | 2026-06-14 | Separate business logic from rendering. Created viewmodels.py (pure Python dataclasses, no HTML), builders.py (8 builder functions), refactored 6 component files (portfolio, overview, actions, decision, rebalance, ai_panel). design_system.py split: CSS/HTML extracted to layout.py (338 lines, under 500). 20 viewmodel tests pass. Zero UI changes. |
| Critical Function Test Coverage | ✅ Complete | SPEC 55 | 2026-06-14 | Comprehensive tests for the three critical recommendation engine functions (get_portfolio_action, get_sell_analysis, get_portfolio_health). Created tests/fixtures.py with 10 portfolio scenarios (healthy, oversized, all_cash, all_invested, high_vix, single_stock, large_gain, large_loss, high_concentration, tiny). 77 tests: structural, signal, cross-consistency, and parametrized crash-safety. Fixed 4 bugs found by tests: get_sell_analysis underscored extreme concentration (now 60pts for >50% position), catastrophic losses (now 45pts for <-40%), large profits (now 30pts for >100% gain); _max_sector_conc now uses pv as denominator so cash dilutes concentration; latest_buy_signal used as fallback when trades_df is None; get_portfolio_health momentum uses latest_buy_signal fallback. Also fixed log_exception call signature bug (was 2-arg, now correct 3-arg) in 5 places. |
| Analytics Exception Hardening | ✅ Complete | SPEC 54 | 2026-06-14 | Fixed all swallowed exceptions in analytics_service.py and analytics_repository.py. save_daily_snapshot() now logs at WARNING when health_score=0, INFO on success, and uses log_exception() on any failure. Added try/except+log_exception to get_sharpe_ratio() and get_max_drawdown(). Added check_health() method. Upgraded all 5 CLASS B handlers in analytics_repository.py to log_exception(). Wired check_health() into run_loop() startup and upgraded end_of_day call site in bot/main.py. Added test_analytics_service_check_health to test_duckdb_integration.py and test_analytics_check_health to ui_tester.py GROUP 8. |
| Rebalance Suggestions | ✅ Complete | SPEC 38 | 2026-06-26 | P1: render_rebalance_suggestions() — full rebalance summary card: reduce/exit/add rows, net cash change, positions before/after, sector risk delta. Portfolio tab below position sizing. |
| Paper Trading Scorecard | ✅ Complete | SPEC 39 | 2026-06-26 | P1: render_paper_trading_scorecard() — return vs SPY/QQQ (yfinance), Sharpe, max DD, win rate, AI-follow rate. Models tab under Performance Tracking. |

### BACKEND FEATURES
| Feature | Status | Spec | Last Updated | Notes |
|---------|--------|------|--------------|-------|
| 5-min Trading Loop | ✅ Complete | SPEC 10 | 2026-06-13 | GitHub Actions cron; market-hours + holiday detection; HALT_TRADING emergency override |
| Pre-market Screener | ✅ Complete | SPEC 11 | 2026-06-13 | universe_today.json → screener_log; RL agent ranks candidates; separate premarket job |
| Technical Feature Engineering | ✅ Complete | SPEC 12 | 2026-06-13 | compute_features(): ATR, RSI, EMA, volume ratio, 15-min RSI via 5-min bars |
| Market Regime Classifier | ✅ Complete | SPEC 13 | 2026-06-13 | TRENDING / RANGING / BEARISH / VOLATILE labels; entry gated to TRENDING + RANGING only |
| XGBoost Signal Model | ✅ Complete | SPEC 14 | 2026-06-13 | Probability-calibrated; SHAP feature_drivers stored per trade; pre-market retrain |
| LSTM Signal Model | ✅ Complete | SPEC 15 | 2026-06-13 | 30-bar rolling window; loaded once in run_loop to avoid per-cycle startup cost |
| Sentiment Pipeline | ✅ Complete | SPEC 16 | 2026-06-13 | FinBERT premarket batch (NewsAPI) + Reddit/WSB dynamic weighting (log1p mentions, 5-min cache) |
| FRED Macro Signals | ✅ Complete | SPEC 17 | 2026-06-13 | VIX >= 40 halts all buys; macro score + size cap; 4-hour DB-backed cache |
| Ensemble Signal | ✅ Complete | SPEC 18 | 2026-06-13 | Weighted: XGB + LSTM + sentiment + macro → STRONG_BUY / BUY / HOLD / SELL |
| Entry Gate Suite | ✅ Complete | SPEC 19 | 2026-06-13 | 10 gates: VIX halt / regime / volume / 15-min RSI / RS / open-order / earnings / correlation / wash-sale / stop re-entry + Kelly sizing |
| Exit Logic Suite | ✅ Complete | SPEC 20 | 2026-06-13 | Gap-down floor → take-profit (3xATR, 6-8%) → ATR stop → trailing stop → drift trim → time-exit → ensemble sell |
| Risk Manager | ✅ Complete | SPEC 21 | 2026-06-13 | Daily 2% / weekly 5% loss limits; PDT 3-trade gate; drawdown circuit-breaker; portfolio-high tracking |
| Alpaca Execution Engine | ✅ Complete | SPEC 22 | 2026-06-13 | Limit buy + fill confirmation; limit/market sell with stop-timeout escalation; slippage logging |
| SQLite Data Layer | ✅ Complete | SPEC 23 | 2026-06-13 | 8 tables: trades, position_state, risk_state, earnings_cache, macro_cache, portfolio_snapshots, signal_log, screener_log |
| HuggingFace DB Bridge | ✅ Complete | SPEC 24 | 2026-06-13 | sync_db.py pushes trades.db at most every 15 min; dashboard reads from HF dataset repo |
| Telegram Alert System | ✅ Complete | SPEC 25 | 2026-06-13 | BUY / SELL / stop-loss / risk-warning / VIX-halt / daily-summary / weekly-report alerts |
| Position Reconciliation | ✅ Complete | SPEC 26 | 2026-06-13 | Startup sync: removes stale DB entries, logs SELL_RECONCILE; seeds externally-opened positions |
| DuckDB Analytics Foundation Layer | ✅ Complete | SPEC 50 | 2026-06-14 | Columnar analytics DB alongside SQLite; price_history, portfolio_snapshots, recommendation_history tables; AnalyticsRepository + AnalyticsService; BUY signal and daily snapshot wired into bot/main.py; 6 integration tests pass |

Status legend:
✅ Complete — built and tested  🔄 In Progress — currently being worked on
⏳ Planned — specified but not started  ❌ Blocked — cannot proceed
🐛 Has Bug — working but known issue exists

---

## ENHANCEMENTS LOG
Chronological list of all improvements:

### [2026-06-13] Portfolio Health Score (SPEC 1)
- Added to render_dashboard_hero(); score from VIX, cash, concentration, drawdown
- Green ≥ 75, purple ≥ 50, red < 50 with progress bar and weakest-component subtitle

### [2026-06-13] Rich Telegram Alerts (SPEC 2)
- BUY: ensemble %, XGBoost/LSTM %, SHAP drivers in plain English, sector %, cash %
- SELL: exit reason label, freed cash %. Daily summary at 4:05pm ET (SPEC 5)

### [2026-06-13] Since Yesterday Panel (SPEC 3)
- render_whats_changed(): compares latest-per-symbol between today and yesterday
- Detects ensemble_score (>0.05), regime label, sentiment direction changes

### [2026-06-13] AI Action Column (SPEC 4)
- HOLD/WATCH/TRIM/EXIT badge from sell score (size risk + profit + confidence + DD)
- Sub-row below each position row shows top two scoring reasons in plain text

### [2026-06-13] Portfolio Performance + yfinance + Sparklines (SPEC 6A/6B/6C)
- 1D/1W/1M/3M/YTD/1Y/All Time Radio tabs at top of Portfolio tab
- Per-stock % columns in positions table via yfinance with 1-hour module-level cache
- 80×32px SVG sparkline (last 30 closes) as second column in positions table

### [2026-06-13] UI Change Tracker (SPEC 8)
- tests/ui_changelog.py: snapshots all render_* functions via ast, diffs on each run
- Appends entries to docs/UI_CHANGELOG.md; supports --diff, --history, --reset

### [2026-06-13] 5 New Dashboard Components (SPEC 27-31)
- render_todays_actions(): today BUY/SELL trade timeline with P&L and exit reason labels
- render_portfolio_actions(): HOLD/WATCH/TRIM/EXIT recommendation panel per open position
- render_ai_committee(): XGBoost/LSTM/Sentiment vote chips per position with majority verdict
- render_position_sizing(): conviction-based target allocation guidance (Portfolio tab)
- render_symbol_detail(): enhanced with action card, conviction bar, sizing guidance at top
- Added _SELL_REASON module-level dict (latent bug fix)

### [2026-06-13] UX Polish Sprint (SPEC 43-49)
- SPEC 43: HOLD visual fix — _action_row() symbol color TEXT2→TEXT1 for HOLD/WATCH rows
- SPEC 44: Badge standardization — all inline badge HTML replaced with _action_badge()
- SPEC 45: Empty state standardization — all custom empties replaced with _empty_state()
- SPEC 46: render_decision_center() — consolidates 3 panels into one sortable table
- SPEC 47: render_rebalance() — current vs target with delta + before/after health score
- SPEC 48: Dashboard layout — exactly 5 panels; panels moved to correct tabs
- SPEC 49: Consistency checks — section titles ≤4 words, design system colors only

### [2026-06-14] Exception Handling Hardening (SPEC 51)
- docs/exception_audit.txt: full audit of 7 files with CLASS A/B/C classification
- bot/core/error_logger.py: safe_render() + timed() decorators added; log_exception upgraded to 4-arg
- dashboard/app.py: @safe_render on 25 HTML render functions; @timed on 5 slow functions
- dashboard/app.py: 17 CLASS A silent-fail exceptions fixed with logger.debug; 3 CLASS B upgraded to opt(exception=True).warning
- bot/main.py: 5 CLASS A exceptions fixed (spy_bars, spy_today, WSB, sleep timer)
- database/services/analytics_service.py: 1 CLASS A fixed
- tests/ui_tester.py: GROUP 8 exception check added; unicode encoding fix made conditional

### [2026-06-14] DuckDB Analytics Foundation Layer (SPEC 50)
- database/duckdb/schema.sql: price_history, portfolio_snapshots, recommendation_history tables + 2 indexes
- database/repositories/analytics_repository.py: AnalyticsRepository — save/load price history, snapshots, recommendations
- database/services/analytics_service.py: AnalyticsService — get_sharpe_ratio, get_max_drawdown, save_daily_snapshot, save_recommendation
- bot/main.py: BUY ensemble score saved before client.buy(); daily snapshot saved after end_of_day_summary() alert
- bot/core/error_logger.py: setup_logger(name) added
- requirements.txt + requirements_space.txt: duckdb>=1.0.0 added
- tests/test_duckdb_integration.py: 6 integration tests — all pass

### [2026-06-13] Master Product Vision — Five Investor Questions (SPEC 32-39)
- SPEC 32: bot/core/recommendation_engine.py — 5 shared helpers, single source of truth
- SPEC 33: render_portfolio_health_hero() — 5-component health score, replaces hero panel
- SPEC 34: render_todays_actions() rebuilt — sorted action list with sizing guidance
- SPEC 35: render_sell_analysis() — sell score table + expanded detail card
- SPEC 36: render_why_panel() — bullish/bearish split + model breakdown per symbol
- SPEC 37: render_position_sizing_panel() — current→target weight flow with cash check
- P1: SPEC 38 rebalance suggestions, SPEC 39 paper trading scorecard

### [2026-06-26] Rebalance Suggestions (SPEC 38)
- render_rebalance_suggestions(): groups positions into reduce/add buckets
- Shows net cash freed vs deployed, dominant sector gaining/losing weight
- Portfolio tab below render_rebalance()

### [2026-06-26] Paper Trading Scorecard (SPEC 39)
- render_paper_trading_scorecard(): bot return vs SPY and QQQ since first trade (1-hour benchmark cache)
- Win rate, Sharpe, max drawdown, AI follow rate (BUY signals executed in last 30 days)
- Always-visible panel at top of Models tab

### [2026-06-26] Living Requirements Tracker complete (SPEC 9)
- Added SPEC 38/39 to _FEATURES and _SPEC_DEFS in requirements_tracker.py
- req_state.json patched; REQUIREMENTS.md regenerated
- All 53 specs now tracked; 53/53 complete


---

## BUG TRACKER

| ID | Description | Severity | Status | File | Discovered | Fixed |
|----|-------------|----------|--------|------|------------|-------|
| BUG-001 | get_sell_analysis underscored extreme concentration | 🟡 Medium | 🔄 In Progress | recommendation_engine.py | 2026-06-14 | 2026-06-14 |
| BUG-002 | get_sell_analysis underscored catastrophic losses | 🟠 High | 🔄 In Progress | recommendation_engine.py | 2026-06-14 | 2026-06-14 |
| BUG-003 | get_sell_analysis underscored large profits | 🟡 Medium | 🔄 In Progress | recommendation_engine.py | 2026-06-14 | 2026-06-14 |
| BUG-004 | log_exception() called with wrong arg count (2 vs 4) in 5 places | 🟡 Medium | 🔄 In Progress | multiple | 2026-06-14 | 2026-06-14 |

Severity: 🔴 Critical  🟠 High  🟡 Medium  🟢 Low

---

## TECHNICAL DECISIONS

### One Gradio app not three
Single shared portfolio and AI committee view reduces maintenance overhead.

### Pure Python / Gradio over React
Matches ML codebase; all UI rendered as HTML strings inside gr.HTML components.

### yfinance for historical prices (SPEC 6B/6C)
Alpaca handles live intraday data; yfinance provides multi-year history for free.

### SQLite synced via HuggingFace dataset repo
Simple file-based persistence without external DB. trades.db pushed after each cycle.

### Module-level _CACHE with 55-second TTL
All render functions share one DB read per 60-second refresh to prevent N×DB calls.

### GitHub Actions cron + HF Spaces auto-deploy
Bot runs on scheduled GH Actions workflows. Dashboard auto-deploys from main branch.


---

## KNOWN LIMITATIONS

- Day trading not implemented — intentional to avoid the PDT rule
- Paper trading only — real money deployment gated behind win rate ≥ 60% sustained over 90 days
- SQLite not suitable for high-frequency writes; intentional for swing-trading cadence
- Dashboard read-only — no manual order entry by design (all orders through bot logic)
- Historical performance shows --- until enough bot history accumulates (< 30 days live)
- yfinance slow on first load with 10+ open positions (cached for 1 hour after)
- Dashboard refreshes every 60 seconds (Gradio Timer) — real-time would require websockets
- DuckDB analytics not yet surfaced in scorecard panel (reads SQLite directly instead)

---

## NEXT PRIORITIES
Auto-updated based on planned specs and open bugs:

All planned features complete. 🎉
