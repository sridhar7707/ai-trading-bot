# API Keys Setup Guide

Every service used in this bot is free. Register for each key below, then add it to your `.env` file and as a GitHub Actions secret.

---

## 1. Alpaca (paper trading + market data)
**Already set up.** Your paper trading key is in `.env`.

- Dashboard: https://app.alpaca.markets
- Regenerate secret: Settings → API Keys → Regenerate

| `.env` variable | GitHub Secret |
|---|---|
| `ALPACA_KEY` | `ALPACA_KEY` |
| `ALPACA_SECRET` | `ALPACA_SECRET` |

---

## 2. Telegram Bot (trade alerts)
1. Open Telegram and message **@BotFather**: https://t.me/botfather
2. Send `/newbot`, follow the prompts, copy the **token**
3. Start a conversation with your new bot
4. Get your **chat ID**: message **@userinfobot** https://t.me/userinfobot

| `.env` variable | GitHub Secret |
|---|---|
| `TELEGRAM_TOKEN` | `TELEGRAM_TOKEN` |
| `TELEGRAM_CHAT_ID` | `TELEGRAM_CHAT_ID` |

---

## 3. HuggingFace (model storage + ZeroGPU retraining + dashboard)
1. Register at https://huggingface.co/join
2. Go to Settings → Access Tokens: https://huggingface.co/settings/tokens
3. Create a **Write** token
4. Create a new model repo at https://huggingface.co/new (e.g. `your-username/ai-trading-bot`)

| `.env` variable | GitHub Secret |
|---|---|
| `HF_TOKEN` | `HF_TOKEN` |
| `HF_REPO_ID` | `HF_REPO_ID` (format: `username/ai-trading-bot`) |

---

## 4. FRED API (macro signals — yield curve, VIX, Fed rate)
1. Register at https://fredaccount.stlouisfed.org/login/secure/
2. Go to API Keys: https://fredaccount.stlouisfed.org/apikeys
3. Request a key (instant, free, no approval needed)

| `.env` variable | GitHub Secret |
|---|---|
| `FRED_API_KEY` | `FRED_API_KEY` |

**What it unlocks:** Macro regime context — yield curve inversion detection, VIX fear gauge, Fed funds rate. Bot reduces position sizes automatically in risky macro conditions.

---

## 5. NewsAPI (financial headlines for FinBERT sentiment)
1. Register at https://newsapi.org/register
2. Your API key is shown on the dashboard immediately
3. Free tier: **100 requests/day** (enough for daily sentiment runs)

| `.env` variable | GitHub Secret |
|---|---|
| `NEWSAPI_KEY` | `NEWSAPI_KEY` |

**What it unlocks:** Live news headlines fed into FinBERT to score sentiment per stock. SEC EDGAR filings (no key needed) are used as a fallback.

---

## 6. Reddit API (WSB/retail sentiment)
1. Log in to Reddit and go to https://www.reddit.com/prefs/apps
2. Click **Create App** (or **Create Another App**)
3. Fill in:
   - Name: `ai-trading-bot`
   - Type: **script**
   - Redirect URI: `http://localhost:8080`
4. Copy the **client ID** (under the app name) and **client secret**

| `.env` variable | GitHub Secret |
|---|---|
| `REDDIT_CLIENT_ID` | `REDDIT_CLIENT_ID` |
| `REDDIT_CLIENT_SECRET` | `REDDIT_CLIENT_SECRET` |
| `REDDIT_USER_AGENT` | `REDDIT_USER_AGENT` |

Set `REDDIT_USER_AGENT` to something like: `ai-trading-bot/1.0 by ksri77`

**What it unlocks:** WallStreetBets + investing subreddit mention tracking and sentiment scoring per ticker.

---

## 7. Unusual Whales (options flow — optional)
1. Register at https://unusualwhales.com
2. Go to Account → API: https://unusualwhales.com/api
3. Free tier is limited — skip this for now, add later

| `.env` variable | GitHub Secret |
|---|---|
| `UNUSUAL_WHALES_TOKEN` | `UNUSUAL_WHALES_TOKEN` |

**Status:** Bot runs fine without this. The options flow signal is stubbed out and returns neutral (0.5) if no token is set.

---

## 8. UptimeRobot (HuggingFace Space keep-alive — optional)
1. Register at https://uptimerobot.com
2. Add a new monitor → HTTP(S) → paste your HF Space URL
3. Set interval: 30 minutes
4. No API key needed — it just pings the URL

GitHub Actions `keepalive.yml` already pings the Space daily as a backup.

---

## Adding GitHub Secrets

Go to your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add all of the following:

| Secret Name | Value |
|---|---|
| `ALPACA_KEY` | Your Alpaca API key |
| `ALPACA_SECRET` | Your Alpaca secret key |
| `TELEGRAM_TOKEN` | Your Telegram bot token |
| `TELEGRAM_CHAT_ID` | Your Telegram chat ID |
| `HF_TOKEN` | Your HuggingFace write token |
| `HF_REPO_ID` | e.g. `ksri77/ai-trading-bot` |
| `FRED_API_KEY` | Your FRED API key |
| `NEWSAPI_KEY` | Your NewsAPI key |
| `REDDIT_CLIENT_ID` | Your Reddit app client ID |
| `REDDIT_CLIENT_SECRET` | Your Reddit app client secret |
| `REDDIT_USER_AGENT` | e.g. `ai-trading-bot/1.0 by ksri77` |

---

## Quick Start: Local Development

```bash
# 1. Copy the template and fill in your keys
cp .env.example .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download historical data
python scripts/download_data.py

# 4. Train all models (regime classifier + XGBoost + LSTM + PPO)
python scripts/train_model.py

# 5. Run the backtest
python backtest/engine.py

# 6. Start paper trading locally
python bot/main.py --mode paper
```
