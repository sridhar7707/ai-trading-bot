# TradeGenius AI — External Services

Last updated: 2026-06-27

All credentials are stored as environment variables (never in code).
Production secrets live in GitHub Actions repository secrets and HuggingFace Space secrets.

## Broker

| Service | Purpose | Config Key | Tier |
|---------|---------|------------|------|
| Alpaca Markets | Order execution, portfolio data, intraday bars | `ALPACA_KEY`, `ALPACA_SECRET`, `ALPACA_BASE_URL` | Paper (free) |

Default URL: `https://paper-api.alpaca.markets`. Set to live URL only via env override.
Rate limit: 200 req/min on free tier. Bot batches symbol calls to stay under limit.

## Data Providers (free tier)

| Service | Purpose | Config Key | Cache TTL |
|---------|---------|------------|-----------|
| yfinance | Historical price data, sparklines, benchmark comparison | none (no key needed) | 1 hour module-level |
| FRED API | VIX, T-bill rate, yield curve for macro signals | `FRED_API_KEY` | 4 hours (DB-backed) |
| NewsAPI | Financial headlines for FinBERT sentiment | `NEWSAPI_KEY` | Per-session (premarket batch) |
| Reddit / Pushshift | WSB mention counts for sentiment weighting | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` | 5 minutes (memory) |
| Finnhub | Analyst upgrades/downgrades for screener scoring | `FINNHUB_API_KEY` | Per screener run |

All free tiers — no credit card required. Bot degrades gracefully when any provider
is unavailable: macro/sentiment weight falls to 0, screener omits that factor.

## Infrastructure

| Service | Purpose | Config Key |
|---------|---------|------------|
| GitHub Actions | Bot cron scheduling (every 5 min, market hours) | N/A (workflow files) |
| HuggingFace Spaces | Dashboard hosting (Gradio app) | Auto-deploy from main branch |
| HuggingFace Dataset Repo | `trades.db` file storage (pushed by `sync_db.py`) | `HF_TOKEN`, `HF_DB_REPO_ID` |

## Alerts

| Service | Purpose | Config Key |
|---------|---------|------------|
| Telegram Bot API | BUY/SELL/risk/daily-summary alerts | `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` |

Alert types: BUY signal, SELL signal (with reason), stop-loss triggered, daily loss warning,
VIX halt, daily summary (4:05 PM CT), weekly report.

## Dependency Map

```
GitHub Actions cron
    └─► bot/main.py
            ├─► Alpaca API (orders, positions, account)
            ├─► yfinance (intraday bars, historical)
            ├─► FRED API (VIX, macro indicators)
            ├─► NewsAPI (headlines → FinBERT)
            ├─► Reddit API (WSB mentions)
            ├─► Finnhub (screener, analyst signals)
            ├─► Telegram Bot API (alerts)
            └─► HuggingFace Dataset (trades.db push)

HuggingFace Spaces (dashboard)
    └─► HuggingFace Dataset (trades.db pull)
            └─► yfinance (benchmark, sparklines, news)
```

## Outage Handling

- **Alpaca down**: orders queued; bot logs WARNING and skips cycle
- **yfinance timeout**: cached data used; sparklines show last known value
- **FRED unavailable**: macro_score=0, no VIX halt (conservative: allows buys)
- **Telegram down**: alert logged at WARNING; bot continues trading
- **HuggingFace push fails**: bot continues locally; next sync picks up the gap
