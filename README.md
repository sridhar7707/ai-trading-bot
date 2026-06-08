# 🤖 AI Trading Bot

A self-learning AI trading bot that targets index outperformance (beating the S&P 500) using Reinforcement Learning, trained on Stocks and ETFs via Alpaca (paper trading) with a fully autonomous execution pipeline.

> **Status:** 🟡 Paper Trading Phase — not yet deployed with real money

---

## 🎯 Goals

- Beat the S&P 500 annual return (~10%)
- Trade Stocks + ETFs autonomously
- Start with paper trading, graduate to real money only after passing confidence checks
- Run 100% free on GitHub Actions + Hugging Face + Alpaca

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FULLY FREE STACK                        │
├───────────────┬──────────────────┬──────────────────────────┤
│ Hugging Face  │  GitHub Actions  │       Alpaca API         │
│               │                  │                          │
│ • RL Model    │ • Runs bot every │ • Paper trading          │
│   storage     │   5 min (market  │ • Free market data       │
│ • ZeroGPU     │   hours)         │ • Order execution        │
│   training    │ • Sunday retrain │ • Portfolio tracking     │
│ • Dashboard   │ • Keep-alive     │                          │
│   (Gradio)    │   ping           │                          │
└───────────────┴──────────────────┴──────────────────────────┘
        ↓                                      ↓
  Telegram Bot                        SQLite (trade log)
  (free alerts)                       (committed to repo)
```

### Data Flow (Every 5 Minutes)
```
1. Fetch latest prices + indicators (Alpaca)
2. Classify market regime (Trending / Ranging / Volatile)
3. Feed observation to RL agent
4. Risk manager approves or blocks action
5. Execute order on Alpaca (paper)
6. Log outcome to SQLite
7. Send Telegram alert
```

### Weekly Retraining (Every Sunday 2am UTC)
```
1. Pull last 30 days of trade history
2. Retrain RL agent on HuggingFace ZeroGPU
3. Backtest new model vs old model
4. Deploy new model only if it outperforms
5. Send weekly performance report via Telegram
```

---

## 📁 Project Structure

```
ai-trading-bot/
│
├── bot/                        # Core bot logic
│   ├── execution/              # Order execution (Alpaca)
│   │   ├── __init__.py
│   │   └── alpaca_client.py    # Alpaca API wrapper
│   ├── strategy/               # Trading strategies
│   │   ├── __init__.py
│   │   ├── rl_agent.py         # PPO Reinforcement Learning agent
│   │   ├── regime_classifier.py# Market regime detection
│   │   └── features.py         # Technical indicator engineering
│   ├── risk/                   # Risk management (hard overrides)
│   │   ├── __init__.py
│   │   └── risk_manager.py     # Stop-loss, position sizing, PDT guard
│   └── monitor/                # Alerting & health checks
│       ├── __init__.py
│       └── telegram_bot.py     # Telegram notification system
│
├── data/                       # Data storage
│   ├── raw/                    # Raw OHLCV data from Alpaca
│   ├── processed/              # Cleaned + feature-engineered data
│   └── features/               # Computed indicators cache
│
├── models/                     # ML models
│   ├── saved/                  # Trained model checkpoints
│   └── training/               # Training scripts + configs
│
├── backtest/                   # Backtesting engine
│   ├── __init__.py
│   ├── engine.py               # Core backtesting loop
│   └── metrics.py              # Sharpe, drawdown, win rate etc.
│
├── dashboard/                  # HuggingFace Spaces Gradio app
│   ├── app.py                  # Main Gradio dashboard
│   ├── templates/
│   └── static/
│
├── scripts/                    # Utility scripts
│   ├── download_data.py        # Historical data downloader
│   ├── train_model.py          # Offline model training
│   ├── confidence_check.py     # Go-live readiness checker
│   ├── save_model_hf.py        # Push model to HuggingFace Hub
│   └── load_model_hf.py        # Pull model from HuggingFace Hub
│
├── notebooks/                  # Jupyter notebooks for research
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_backtesting.ipynb
│   └── 04_model_training.ipynb
│
├── .github/
│   └── workflows/
│       ├── trade.yml           # Trading bot (every 5 min, market hours)
│       ├── retrain.yml         # Weekly model retraining
│       └── keepalive.yml       # HuggingFace Space keep-alive ping
│
├── trades.db                   # SQLite trade history
├── config.py                   # Configuration (symbols, parameters)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── .gitignore
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-trading-bot.git
cd ai-trading-bot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Get your API keys (all free)

| Service | Purpose | Link |
|---|---|---|
| Alpaca | Paper trading + market data | [alpaca.markets](https://alpaca.markets) |
| Telegram | Trade alerts | [@BotFather](https://t.me/botfather) |
| HuggingFace | Model storage + dashboard | [huggingface.co](https://huggingface.co) |

### 5. Run backtests first
```bash
python scripts/download_data.py
python scripts/train_model.py
python backtest/engine.py
```

### 6. Start paper trading
```bash
python bot/main.py --mode paper
```

---

## 🔐 GitHub Secrets Required

Add these in your repo → Settings → Secrets → Actions:

| Secret | Description |
|---|---|
| `ALPACA_KEY` | Alpaca API key |
| `ALPACA_SECRET` | Alpaca secret key |
| `TELEGRAM_TOKEN` | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Your Telegram chat ID |
| `HF_TOKEN` | HuggingFace write token |
| `HF_REPO_ID` | HuggingFace model repo (e.g. username/ai-trading-bot) |

---

## 📊 Confidence Check — Before Going Live

The bot will NOT use real money until it passes ALL of these:

| Metric | Minimum |
|---|---|
| Paper trading duration | 60+ days |
| Win rate | ≥ 52% |
| Sharpe ratio | ≥ 1.0 |
| Max drawdown | ≤ 15% |
| vs S&P 500 | Outperforming |
| Consecutive losing days | ≤ 4 |

Run the check anytime:
```bash
python scripts/confidence_check.py
```

---

## 🛡️ Risk Management Rules (Hard-Coded, Non-Negotiable)

- **Max position size:** 20% of portfolio per trade
- **Stop-loss:** Auto-sell if position drops 4%
- **Daily loss limit:** Bot halts if daily P&L hits -5%
- **PDT guard:** Tracks day trades, halts before hitting the 3-trade limit (for sub-$25K accounts)
- **Sector concentration:** Max 2 stocks per sector

---

## 📈 Trading Strategy

### Market Regime Detection
The bot first classifies the current market condition:

| Regime | Strategy |
|---|---|
| Trending UP | Momentum — ride winners |
| Trending DOWN | Defensive — reduce exposure |
| Ranging | Mean reversion — buy dips, sell rips |
| High Volatility | Reduce position sizes, tighten stops |

### RL Agent (PPO)
- **Algorithm:** Proximal Policy Optimization (PPO) via Stable Baselines3
- **Observation space:** Price features + technical indicators + portfolio state + regime label
- **Action space:** Buy / Sell / Hold (with fractional share support)
- **Reward function:** Risk-adjusted returns (Sharpe ratio) — not raw profit
- **Retraining:** Every Sunday on HuggingFace ZeroGPU

---

## 📱 Telegram Alerts

The bot sends alerts for:
- ✅ Every trade executed (buy/sell/hold)
- 📊 Daily P&L summary at 4:05pm EST
- ⚠️ Stop-loss triggers
- 🚨 Daily loss limit reached (bot halted)
- 🔴 Bot offline / health check failures
- 🚀 Confidence check passed (ready for real money)
- 📈 Weekly performance report vs S&P 500

---

## 🗓️ Build Roadmap

- [ ] **Phase 1** — Data pipeline (Alpaca OHLCV + indicators)
- [ ] **Phase 2** — Backtesting engine
- [ ] **Phase 3** — Market regime classifier
- [ ] **Phase 4** — RL agent (PPO) — train locally
- [ ] **Phase 5** — Risk manager
- [ ] **Phase 6** — Alpaca paper trading execution
- [ ] **Phase 7** — Telegram alerts
- [ ] **Phase 8** — GitHub Actions scheduling
- [ ] **Phase 9** — HuggingFace model storage + ZeroGPU retraining
- [ ] **Phase 10** — Gradio dashboard on HF Spaces
- [ ] **Phase 11** — UptimeRobot keep-alive for HF Space
- [ ] **Phase 12** — Confidence check → graduate to real money

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Algorithmic trading involves significant financial risk. Past performance does not guarantee future results. Always paper trade extensively before risking real capital. The authors are not financial advisors.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
