# TradeGenius — Living Requirements Document

Auto-generated and auto-updated.
Last updated: 2026-06-13 15:33:45
Version: 1.0.2

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
