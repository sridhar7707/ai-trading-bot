# TradeGenius — Living Requirements Document

Auto-generated and auto-updated.
Last updated: 2026-06-13 19:29:31
Version: 1.0.8

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
| Living Requirements Tracker | 🔄 In Progress | SPEC 9 | 2026-06-13 | tests/requirements_tracker.py — this file |
| Portfolio Actions Panel | ✅ Complete | SPEC 27 | 2026-06-13 | render_portfolio_actions(): HOLD/WATCH/TRIM/EXIT per open position from ensemble score + sell score; Dashboard tab above market intelligence row |
| Position Sizing Recommendation | ✅ Complete | SPEC 28 | 2026-06-13 | render_position_sizing(): conviction-based target allocation (3-12%) from ensemble score; Portfolio tab above positions table |
| Today's Actions Summary | ✅ Complete | SPEC 29 | 2026-06-13 | render_todays_actions(): today BUY/SELL timeline with time, price, P&L, reason; Dashboard tab between hero and whats_changed |
| AI Investment Committee | ✅ Complete | SPEC 30 | 2026-06-13 | render_ai_committee(): XGBoost/LSTM/Sentiment vote chips per open position with majority verdict; Dashboard tab right column above watchlist |
| Symbol Detail Action Card | ✅ Complete | SPEC 31 | 2026-06-13 | Enhancement to render_symbol_detail(): action card (HOLD/WATCH/TRIM/EXIT), conviction bar, sizing guidance injected at top of detail card |
| Recommendation Engine | ⏳ Planned | SPEC 32 | 2026-06-13 | P0: bot/core/recommendation_engine.py — 5 shared helpers: get_portfolio_action, get_position_sizing, get_sell_analysis, get_recommendation_explanation, get_portfolio_health. Single source of truth for all panels. |
| Portfolio Health Hero Panel | ⏳ Planned | SPEC 33 | 2026-06-13 | P0: render_portfolio_health_hero() — replaces render_dashboard_hero(). 5-component health score (Risk/Diversification/Cash/Momentum/Quality) + biggest risk. Calls get_portfolio_health(d). |
| Today's Actions Panel v2 | ⏳ Planned | SPEC 34 | 2026-06-13 | P0: render_todays_actions() rebuilt — sorted action list (EXIT/SELL/TRIM/ADD/BUY/WATCH/HOLD) with sizing from get_position_sizing(). Replaces SPEC 29 (executed trades view). |
| Sell Analysis Panel | ⏳ Planned | SPEC 35 | 2026-06-13 | P0: render_sell_analysis() — sell score 0-100 table (position size + profit + conviction + drawdown) with expanded detail card for highest-scored position. Portfolio tab. |
| Why Panel | ⏳ Planned | SPEC 36 | 2026-06-13 | P0: render_why_panel(symbol) — bullish/bearish factor split with model breakdown (XGBoost/LSTM/Sentiment) and plain-English summary. Shown above render_symbol_detail() on symbol select. |
| Position Sizing Panel v2 | ⏳ Planned | SPEC 37 | 2026-06-13 | P0: render_position_sizing_panel() — current→target weight flow with delta $+shares, cash sufficiency footer. Replaces render_position_sizing() (SPEC 28). Portfolio tab. |
| Rebalance Suggestions | ⏳ Planned | SPEC 38 | 2026-06-13 | P1: render_rebalance_suggestions() — full rebalance summary card: reduce/exit/add rows, net cash change, positions before/after, sector risk delta. Portfolio tab below position sizing. |
| Paper Trading Scorecard | ⏳ Planned | SPEC 39 | 2026-06-13 | P1: render_paper_trading_scorecard() — return vs SPY/QQQ (yfinance), Sharpe, max DD, win rate, AI-follow rate. Models tab under Performance Tracking. |

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

### [2026-06-13] Master Product Vision — Five Investor Questions (SPEC 32-39)
- SPEC 32: bot/core/recommendation_engine.py — 5 shared helpers, single source of truth
- SPEC 33: render_portfolio_health_hero() — 5-component health score, replaces hero panel
- SPEC 34: render_todays_actions() rebuilt — sorted action list with sizing guidance
- SPEC 35: render_sell_analysis() — sell score table + expanded detail card
- SPEC 36: render_why_panel() — bullish/bearish split + model breakdown per symbol
- SPEC 37: render_position_sizing_panel() — current→target weight flow with cash check
- P1: SPEC 38 rebalance suggestions, SPEC 39 paper trading scorecard


---

## BUG TRACKER

| ID | Description | Severity | Status | File | Discovered | Fixed |
|----|-------------|----------|--------|------|------------|-------|
| — | No bugs recorded | — | — | — | — | — |

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
- Historical performance shows '—' until enough bot history accumulates
- yfinance slow on first load with 10+ open positions (cached for 1 hour after)
- Dashboard refreshes every 60 seconds (Gradio Timer)
- Paper trading only — real money deployment gated behind confidence threshold
- SQLite not suitable for high-frequency writes; fine for swing-trading cadence

---

## NEXT PRIORITIES
Auto-updated based on planned specs and open bugs:

1. SPEC 9 — Living Requirements Tracker (in progress)
2. SPEC 32 — Recommendation Engine
3. SPEC 33 — Portfolio Health Hero Panel
4. SPEC 34 — Today's Actions Panel v2
5. SPEC 35 — Sell Analysis Panel
6. SPEC 36 — Why Panel
7. SPEC 37 — Position Sizing Panel v2
8. SPEC 38 — Rebalance Suggestions
9. SPEC 39 — Paper Trading Scorecard
