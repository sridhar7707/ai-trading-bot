# TradeGenius AI — Architecture

Last updated: 2026-06-27

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  GitHub Actions (cron, every 5 min, market hours)               │
│                                                                   │
│  bot/main.py → _main_runner.py → _main_cycle.py                 │
│       │              │                  │                         │
│       │         market check       per-symbol loop               │
│       │              │           (signals → risk → execute)       │
│       │              │                                            │
│       └──────► sync_db.py → HuggingFace Dataset (trades.db)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (dashboard reads)
┌─────────────────────────────────────────────────────────────────┐
│  HuggingFace Spaces (always-on, auto-deployed from main)        │
│                                                                   │
│  dashboard/app.py                                                │
│       ├── gr.Timer(60s) → render_* functions                    │
│       ├── dashboard/data.py (55s TTL cache, DB reads)           │
│       └── dashboard/components/ (15 component modules)          │
└─────────────────────────────────────────────────────────────────┘
```

## 6-Layer Bot Architecture

Data flows strictly **downward** — upper layers may not import lower layers.

```
┌────────────────────────────────────────────────────────────────────┐
│ Layer 1 — Data Ingestion                                           │
│  bot/strategy/features.py      compute_features(): ATR,RSI,EMA    │
│  bot/strategy/macro.py         FRED VIX + T-bill, 4h DB cache     │
│  bot/strategy/sentiment.py     FinBERT (NewsAPI) + Reddit WSB      │
├────────────────────────────────────────────────────────────────────┤
│ Layer 2 — Regime Classification                                    │
│  bot/strategy/regime_classifier.py   TRENDING_UP/RANGING/etc      │
├────────────────────────────────────────────────────────────────────┤
│ Layer 3 — Signal Generation                                        │
│  bot/strategy/xgb_predictor.py    XGBoost probability + SHAP      │
│  bot/strategy/lstm_predictor.py   LSTM 30-bar sequence score       │
│  bot/strategy/ensemble.py         Weighted blend → BUY/HOLD/SELL   │
│  bot/strategy/rl_agent.py         PPO RL agent (position sizer)    │
│  bot/strategy/signal_gate.py      10-gate entry filter             │
├────────────────────────────────────────────────────────────────────┤
│ Layer 4 — Risk Management                                          │
│  bot/risk/risk_manager.py         Daily/weekly loss limits         │
│                                    PDT guard, drawdown CB           │
│                                    Position sizing (Kelly)          │
│                                    Stop-loss check                  │
├────────────────────────────────────────────────────────────────────┤
│ Layer 5 — Execution                                                │
│  bot/execution/alpaca_client.py   Limit buy + fill confirmation    │
│                                    Limit/market sell + escalation   │
├────────────────────────────────────────────────────────────────────┤
│ Layer 6 — Monitoring                                               │
│  bot/monitor/telegram_bot.py      BUY/SELL/risk/daily alerts       │
│  bot/monitor/sync_db.py           trades.db push to HuggingFace    │
│  dashboard/                       Gradio read-only dashboard        │
│  database/                        DuckDB analytics service          │
└────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
ai-trading-bot/
├── bot/
│   ├── _main_cycle.py       per-symbol entry/exit logic
│   ├── _main_db.py          DB write helpers for trades/state
│   ├── _main_market.py      market open check, SPY bar fetch
│   ├── _main_positions.py   position reconciliation at startup
│   ├── _main_runner.py      outer loop orchestration
│   ├── _main_signals.py     signal preparation per symbol
│   ├── main.py              entrypoint (run_loop)
│   ├── core/
│   │   ├── api_guard.py           rate-limit guard for Alpaca
│   │   ├── error_logger.py        safe_render, timed, log_exception
│   │   ├── recommendation_engine.py  5 shared dashboard helpers
│   │   └── recommendation_portfolio.py
│   ├── execution/
│   │   └── alpaca_client.py
│   ├── monitor/
│   │   ├── sync_db.py         HuggingFace DB push
│   │   └── telegram_bot.py
│   ├── risk/
│   │   └── risk_manager.py
│   └── strategy/
│       ├── ensemble.py
│       ├── features.py
│       ├── lstm_predictor.py
│       ├── macro.py
│       ├── reddit_sentiment.py
│       ├── regime_classifier.py
│       ├── rl_agent.py
│       ├── sentiment.py
│       ├── signal_gate.py
│       └── xgb_predictor.py
├── backtest/
│   ├── engine.py            walk-forward simulator
│   └── metrics.py           Sharpe, win rate, drawdown
├── dashboard/
│   ├── app.py               Gradio wiring (300 lines, no logic)
│   ├── builders.py          8 view-model builder functions
│   ├── charts.py            Plotly chart renderers
│   ├── data.py              55s-TTL cache + DB reader
│   ├── design_system.py     tokens + component helpers
│   ├── layout.py            CSS + static HTML
│   ├── viewmodels.py        pure-Python dataclasses
│   └── components/
│       ├── actions.py       today's actions + portfolio actions
│       ├── ai_panel.py      AI recommendation + committee
│       ├── analysis.py      sell analysis + position sizing
│       ├── decision.py      decision center (unified panel)
│       ├── history.py       performance periods + since-yesterday
│       ├── market_mood.py   regime + macro mood tile
│       ├── models.py        model metrics + paper scorecard
│       ├── news.py          Yahoo Finance news feed
│       ├── overview.py      health hero + benchmark comparison
│       ├── portfolio.py     positions table + trade history
│       ├── rebalance.py     rebalance plan + suggestions
│       ├── recommendation_history.py  AI decision log (14 days)
│       ├── risk.py          risk panel + market intelligence
│       ├── signal_history.py  high-confidence signal tracker
│       ├── signals.py       screener watchlist + timeline
│       └── symbol_detail.py  per-symbol drilldown
├── database/
│   ├── repositories/
│   │   └── analytics_repository.py   DuckDB CRUD
│   └── services/
│       └── analytics_service.py      Sharpe, drawdown, snapshot
├── config.py                all trading parameters + env vars
├── scripts/                 maintenance + deployment scripts
├── tests/                   535 tests across 28 test files
└── docs/                    this directory
```

## SQLite Schema (trades.db)

8 tables, managed by `bot/_main_db.py` and `bot/monitor/sync_db.py`:

| Table | Purpose |
|-------|---------|
| `trades` | Every BUY/SELL with price, P&L, ensemble score, SHAP drivers |
| `position_state` | Current open positions (reconciled at startup) |
| `risk_state` | Daily/weekly loss totals, PDT counter, drawdown state |
| `earnings_cache` | Upcoming earnings dates (±2 day block window) |
| `macro_cache` | FRED VIX + macro score (4-hour TTL) |
| `portfolio_snapshots` | Hourly portfolio value + health score snapshots |
| `signal_log` | All ensemble signals per symbol per cycle |
| `screener_log` | Pre-market screener rankings and scores |

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| SQLite over PostgreSQL | File-based; syncs as a single binary to HuggingFace; no external DB to manage |
| Gradio + HTML strings | Matches Python ML codebase; avoids React build pipeline |
| GitHub Actions cron | Free tier; 5-min granularity; easily observable via Actions UI |
| Single trades.db push | Avoids partial-state on dashboard; atomic file replace |
| 55-second cache TTL | Prevents N×DB reads per 60-second refresh cycle across all render functions |
