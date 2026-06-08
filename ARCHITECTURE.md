# 🤖 AI Trading Bot — Full Architecture & Technical Discussion

> Complete record of all architecture, technical, and code decisions made during planning.
> Use this file to resume the project on any device or with any AI assistant.

---

## 👤 Project Owner
- **Email:** ksri77@gmail.com
- **Repo:** ai-trading-bot

---

## 🎯 Core Goals & Decisions

| Question | Decision | Reason |
|---|---|---|
| Primary goal | Beat the market (S&P 500 outperformance) | ~10% annual return is the benchmark to beat |
| Asset types | Stocks (individual) + ETFs | Best liquidity and data availability |
| Autonomy level | Fully autonomous | No manual approval needed per trade |
| Starting capital | Under $1,000 | Low risk while learning |
| Trading timeframe | Mixed / adaptive | Switch strategy based on market conditions |
| Hosting | 100% free stack | GitHub Actions + HuggingFace + Alpaca |
| Notifications | Telegram Bot | Free, reliable, developer-friendly |
| Broker | Alpaca (paper first) → Robinhood (live later) | Alpaca has official paper trading API |
| Go-live strategy | Paper trade → confidence check → real money | Never risk real money until bot proves itself |

---

## 🏗️ Full System Architecture

### The 6-Layer System

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AI TRADING BOT                               │
├──────────────┬──────────────┬────────────┬────────────┬─────────────┤
│   Layer 1    │   Layer 2    │  Layer 3   │  Layer 4   │   Layer 5   │
│    Data      │   Regime     │    RL      │   Risk     │  Execution  │
│  Ingestion   │ Classifier   │   Agent    │  Manager   │   Engine    │
│              │              │   (PPO)    │            │  (Alpaca)   │
└──────────────┴──────────────┴────────────┴────────────┴─────────────┘
                                                              ↓
                                                        Layer 6
                                                       Monitoring
                                                    (Telegram + DB)
```

### Data Flow (Every 5 Minutes During Market Hours)

```
1. Fetch latest prices + indicators (Alpaca API)
2. Classify market regime (Trending / Ranging / Volatile)
3. Feed observation to RL agent (PPO model)
4. Agent outputs action: Buy / Sell / Hold
5. Risk manager approves or BLOCKS the action
6. Execution engine places order on Alpaca
7. Outcome logged to SQLite database
8. Telegram alert sent
```

### Weekly Retraining Flow (Every Sunday 2am UTC)

```
1. Pull last 30 days of trade history from SQLite
2. Fetch latest market data from Alpaca
3. Retrain RL agent on HuggingFace ZeroGPU (free GPU)
4. Backtest new model vs current model
5. Deploy new model ONLY if it outperforms old model
6. Push new model to HuggingFace Hub
7. Send weekly performance report via Telegram
```

---

## 💰 The 100% Free Cloud Stack

```
┌─────────────────────────────────────────────────────────────┐
│                     FULLY FREE STACK                        │
├───────────────┬──────────────────┬──────────────────────────┤
│ Hugging Face  │  GitHub Actions  │       Alpaca API         │
│               │                  │                          │
│ • RL Model    │ • Runs bot every │ • Paper trading          │
│   storage     │   5 min (market  │ • Free real-time data    │
│ • ZeroGPU     │   hours)         │ • Order execution        │
│   training    │ • Sunday retrain │ • Portfolio tracking     │
│ • Dashboard   │ • Keep-alive     │                          │
│   (Gradio)    │   ping           │                          │
└───────────────┴──────────────────┴──────────────────────────┘
        ↓                                      ↓
  Telegram Bot                        SQLite (trade log)
  (free alerts)                       (committed to repo)
        ↓
  UptimeRobot
  (free HF Space pinger)
```

### Cost Breakdown

| Component | Tool | Cost |
|---|---|---|
| Bot execution (scheduled) | GitHub Actions | Free |
| Model storage & versioning | Hugging Face Hub | Free |
| Model training (GPU) | HF ZeroGPU | Free |
| Paper trading + market data | Alpaca | Free |
| Dashboard | HF Spaces (Gradio) | Free |
| Alerts | Telegram Bot API | Free |
| Trade history DB | SQLite in GitHub repo | Free |
| Keep-alive pinger | UptimeRobot | Free |
| **Total** | | **$0/month** |

---

## 📊 Layer 1 — Data Ingestion

### Data Sources
- **Primary:** Alpaca API (real-time + historical OHLCV, free)
- **Backup:** yfinance (historical data only, free)

### Technical Indicators Computed
| Category | Indicators |
|---|---|
| Momentum | RSI (14), Stochastic Oscillator |
| Trend | MACD, MACD Signal, EMA 20, EMA 50, SMA 20 |
| Volatility | Bollinger Bands (high/low/width), ATR |
| Volume | OBV, Volume SMA, Volume Ratio |
| Price Action | Returns, Log Returns, High-Low Ratio, Normalised Close |

### Trading Universe
**Stocks:** AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META

**ETFs:** VOO, QQQ, SPY, VTI, ARKK

**Benchmark:** SPY (S&P 500)

---

## 🧠 Layer 2 — Market Regime Classifier

### Why This Matters
Instead of one strategy running all the time, the bot first classifies the current market condition, then picks the right behavior for it. This is what makes it **adaptive**.

### Four Regimes

| Regime | Condition | Bot Behavior |
|---|---|---|
| TRENDING_UP | Strong uptrend, RSI > 55 | Momentum strategy — ride winners |
| TRENDING_DOWN | Downtrend, RSI < 45 | Defensive — reduce exposure |
| RANGING | Sideways, low trend | Mean reversion — buy dips, sell rips |
| HIGH_VOLATILITY | ATR ratio > 3% | Reduce position sizes, tighten stops |

### Implementation
- **Algorithm:** Random Forest Classifier (sklearn)
- **Features:** 20-day trend, 5-day trend, volatility, ATR ratio, EMA crossovers, RSI, Bollinger width, volume ratio
- **Fallback:** Rule-based detection when model isn't trained yet
- **File:** `bot/strategy/regime_classifier.py`

---

## 🤖 Layer 3 — The RL Agent (Self-Learning Core)

### Why Reinforcement Learning?
The bot learns by trial-and-error in a simulated trading environment. It gets **rewarded for risk-adjusted profits** and penalized for losses and overtrading. Over time it discovers which actions lead to consistent outperformance.

### Algorithm: PPO (Proximal Policy Optimization)
- Implemented via **Stable Baselines3**
- More stable than vanilla policy gradient
- Works well with financial time series
- Supports continuous retraining without catastrophic forgetting

### Observation Space (What the Bot Sees)
```
16 technical indicators
+ balance ratio (current balance / starting balance)
+ shares held (normalised)
+ market regime (0-3 encoded)
= 19-dimensional observation vector
```

### Action Space
```
0 = Hold  (do nothing)
1 = Buy   (fractional shares, 20% of available cash)
2 = Sell  (liquidate entire position)
```

### Reward Function
**Sharpe ratio proxy** — not raw profit.

This is critical. Rewarding raw profit encourages the bot to take huge risks. Rewarding Sharpe ratio encourages **consistent, risk-adjusted** returns — exactly what's needed to reliably beat an index.

```python
reward = mean(returns) / (std(returns) + 1e-8)
```

### Training Config
| Parameter | Value |
|---|---|
| Timesteps per training run | 100,000 |
| Learning rate | 3e-4 |
| Steps per update | 2,048 |
| Batch size | 64 |
| Epochs per update | 10 |
| Retraining schedule | Every Sunday |

### Files
- `bot/strategy/rl_agent.py` — agent + custom Gym environment
- `scripts/train_model.py` — offline training script
- `models/saved/` — model checkpoints

---

## 🛡️ Layer 4 — Risk Manager (Hard Overrides)

### Critical Design Principle
The risk manager runs as a **hard override** — the RL agent CANNOT bypass it. No matter what the model predicts, these rules always apply.

### Rules

| Rule | Threshold | Why |
|---|---|---|
| Max position size | 20% of portfolio per trade | Diversification, max 5 open positions |
| Stop-loss | Auto-sell if position drops 4% | Capital preservation |
| Daily loss limit | Halt trading if daily P&L hits -5% | Prevent death spiral |
| PDT guard | Track day trades, halt before 3-in-5-days limit | Robinhood/Alpaca rule for sub-$25K accounts |
| Sector concentration | Max 2 stocks per sector | Avoid overexposure |

### Pattern Day Trader (PDT) Rule — Important
For accounts under $25,000, brokers limit you to **3 day trades in any 5-business-day period**. Violating this gets your account flagged. The bot tracks this automatically and halts intraday trading when approaching the limit.

### File
`bot/risk/risk_manager.py`

---

## ⚡ Layer 5 — Execution Engine

### Broker: Alpaca
- Official REST API (unlike Robinhood which has no official API)
- Dedicated paper trading environment
- Supports fractional shares (critical for sub-$1K capital)
- Free real-time market data
- Switch paper → live by changing one environment variable

### Order Types
| Scenario | Order Type |
|---|---|
| Entries (buying) | Market order (immediate fill) |
| Exits (selling) | Market order (immediate fill) |
| Stop-loss | Market order (triggered by risk manager) |

### Fractional Shares
Since capital is under $1K, fractional shares allow buying into expensive stocks:
```python
# Buy $200 worth of AMZN regardless of share price
api.submit_order(symbol="AMZN", notional=200, side="buy", ...)
```

### Robinhood (Future)
Robinhood will be added as the live trading execution layer later.
- Uses `robin_stocks` (unofficial reverse-engineered API)
- Same strategy/risk logic — just swap the execution client
- Paper trade on Alpaca first, go live on Robinhood when confident

### File
`bot/execution/alpaca_client.py`

---

## 📱 Layer 6 — Monitoring & Alerting

### Telegram Bot Alerts

| Alert Type | When Triggered |
|---|---|
| 🟢 BUY executed | After every buy order |
| 🔴 SELL executed | After every sell order |
| ⚪ HOLD | Every cycle when holding |
| 📊 Daily P&L summary | 4:05pm EST every trading day |
| ⚠️ Stop-loss triggered | When a position hits -4% |
| 🚨 Daily loss limit | When daily P&L hits -5% (bot halts) |
| 🔴 Bot offline | If health check misses 2 pings |
| 🚀 Confidence check passed | When bot is ready for real money |
| 📈 Weekly report | Every Sunday — performance vs S&P 500 |

### Sample Trade Alert Format
```
🟢 BUY — AAPL
   Shares: 3 (fractional)
   Price: $189.42
   Regime: TRENDING_UP 📈
   Confidence: 81%
   Portfolio: $987.21
   vs S&P 500 today: +1.5% 🏆
```

### Sample Daily Summary Format
```
📊 Daily P&L Report — June 6, 2026
   Day Return:     +2.3% ✅
   vs S&P 500:     +0.8%
   Outperformed:   +1.5% 🏆
   Open Positions: AAPL, VOO
   Cash Available: $312.00
   Trades Today:   2
   Day Trades Used: 1/3 (PDT)
```

### Health Check
- Runs every 15 minutes
- Silent if healthy
- Telegram alert if 2 consecutive checks missed

### File
`bot/monitor/telegram_bot.py`

---

## ☁️ GitHub Actions Workflows

### 1. Trading Bot (`trade.yml`)
```yaml
Schedule: Every 5 minutes, Mon-Fri, 9:30am-4pm EST
          cron: '*/5 14-21 * * 1-5'
Steps:
  1. Pull latest code
  2. Install dependencies
  3. Download RL model from HuggingFace Hub
  4. Run trade.py (fetch data → classify → predict → risk check → execute)
  5. Commit trade log back to repo
```

### 2. Weekly Retraining (`retrain.yml`)
```yaml
Schedule: Every Sunday at 2am UTC
          cron: '0 2 * * 0'
Steps:
  1. Fetch 30 days of market data
  2. Train new PPO model on HuggingFace ZeroGPU
  3. Backtest new vs old model
  4. If new model wins: push to HuggingFace Hub
  5. Send weekly report via Telegram
```

### 3. Keep-Alive Ping (`keepalive.yml`)
```yaml
Schedule: Every day at noon UTC
          cron: '0 12 * * *'
Steps:
  1. curl the HuggingFace Space URL
  2. Prevents Space from going to sleep (48hr timeout)
```

---

## 🤗 HuggingFace Integration

### What HuggingFace Does in This System

| Feature | Used For |
|---|---|
| HF Hub (model storage) | Store/version RL model checkpoints |
| ZeroGPU (free GPU) | Weekly model retraining |
| Spaces (app hosting) | Gradio dashboard (read-only) |

### HuggingFace Spaces Sleep Problem & Solution
Free Spaces go to sleep after 48 hours of no traffic.

**Solution — Two-layer keep-alive:**
1. **UptimeRobot** (free) — pings Space every 30 minutes
2. **GitHub Actions** `keepalive.yml` — pings Space daily as backup

Combined = Space stays online 24/7 for free.

### Dashboard (Gradio on HF Spaces)
Read-only view of:
- Current portfolio value
- Today's P&L vs S&P 500
- Last 10 trades
- Open positions
- Model confidence score
- Current market regime

---

## 📈 Paper Trading → Live Trading Pipeline

### Phase 1 — Backtesting (Weeks 1-4)
- Run bot against 2-3 years of historical data (free from yfinance/Alpaca)
- No real market, no real time — replay the past
- Fast: simulate a full year in seconds
- Output: win rate, Sharpe ratio, max drawdown, vs S&P 500

### Phase 2 — Paper Trading (Months 1-3)
- Bot runs live during real market hours with fake money
- Real prices, real decisions, zero financial risk
- Uses Alpaca's dedicated paper trading API
- Retrains weekly on real market data

### Phase 3 — Confidence Check (The Gate)
Bot only gets real money when ALL of these pass:

| Metric | Minimum Threshold |
|---|---|
| Paper trading duration | 60+ days |
| Win rate | ≥ 52% |
| Sharpe ratio | ≥ 1.0 |
| Max drawdown | ≤ 15% |
| vs S&P 500 | Outperforming |
| Consecutive losing days | ≤ 4 |

Run anytime: `python scripts/confidence_check.py`

### Phase 4 — Live Trading (Gradual Scale-Up)
```
Week 1-2:   $100 real money — validate execution works
Week 3-4:   $250 if week 1-2 profitable
Month 2:    $500 if month 1 profitable
Month 3+:   Full capital if consistently beating S&P 500
```

---

## 📊 Realistic Success Rate Expectations

### Win Rate Benchmarks
| Scenario | Win Rate | Annual Return | Likelihood |
|---|---|---|---|
| Poorly tuned bot | 40-45% | -10% to -30% | Very common |
| Decent bot, good market | 50-55% | 5-15% | Achievable |
| Well-tuned, beats market | 55-60% | 15-25% | Difficult but possible |
| Exceptional (hedge fund) | 60%+ | 25%+ | Very rare |

> A 55% win rate is considered **very good** in trading.
> You don't need to win most of the time — you need winners to be bigger than losers.

### Realistic Timeline
```
Month 1-2:   Build & backtest — bot likely loses on paper
Month 3-4:   Tune the model — approaching breakeven
Month 5-6:   Live trading with $100-300 — small losses expected
Month 7-12:  Model matures — realistic to match market returns
Year 2+:     With retraining — realistic shot at outperforming
```

### Key Risks
| Risk | Mitigation |
|---|---|
| PDT Rule (3 day trades/5 days under $25K) | Built-in PDT guard in risk manager |
| Overfitting to historical data | Weekly retraining on fresh data |
| Market regime shift (bull → bear) | Regime classifier adapts strategy |
| Robinhood outages | Use Alpaca as primary, Robinhood later |
| Small capital friction | Fractional shares minimize this |

---

## 📁 Project Structure

```
ai-trading-bot/
│
├── bot/                          # Core bot logic
│   ├── execution/
│   │   └── alpaca_client.py      # Alpaca API wrapper (paper + live)
│   ├── strategy/
│   │   ├── features.py           # Technical indicator engineering
│   │   ├── regime_classifier.py  # Market regime detection (Random Forest)
│   │   └── rl_agent.py           # PPO RL agent + custom Gym environment
│   ├── risk/
│   │   └── risk_manager.py       # Hard override rules (stop-loss, PDT etc)
│   └── monitor/
│       └── telegram_bot.py       # Telegram notification system
│
├── data/
│   ├── raw/                      # Raw OHLCV from Alpaca
│   ├── processed/                # Cleaned data
│   └── features/                 # Computed indicator cache
│
├── models/
│   ├── saved/                    # Trained PPO + regime classifier
│   └── training/                 # Training configs
│
├── backtest/
│   ├── engine.py                 # Core backtesting loop
│   └── metrics.py                # Sharpe, drawdown, win rate
│
├── dashboard/
│   └── app.py                    # Gradio dashboard (HF Spaces)
│
├── scripts/
│   ├── download_data.py          # Historical data downloader
│   ├── train_model.py            # Offline PPO training
│   ├── confidence_check.py       # Go-live readiness checker
│   ├── save_model_hf.py          # Push model to HuggingFace
│   └── load_model_hf.py          # Pull model from HuggingFace
│
├── notebooks/                    # Jupyter research notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_backtesting.ipynb
│   └── 04_model_training.ipynb
│
├── .github/workflows/
│   ├── trade.yml                 # Trading bot (every 5 min, market hours)
│   ├── retrain.yml               # Weekly model retraining (Sunday 2am)
│   └── keepalive.yml             # HF Space keep-alive (daily ping)
│
├── config.py                     # All configuration + parameters
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── trades.db                     # SQLite trade history
├── README.md                     # Project overview
└── ARCHITECTURE.md               # This file
```

---

## 🔑 Environment Variables

```env
# Alpaca (paper trading)
ALPACA_KEY=your_alpaca_api_key
ALPACA_SECRET=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets   # paper
# ALPACA_BASE_URL=https://api.alpaca.markets       # live (later)

# Telegram
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# HuggingFace
HF_TOKEN=your_huggingface_write_token
HF_REPO_ID=your_username/ai-trading-bot

# Trading Config
TRADING_MODE=paper
INITIAL_CAPITAL=1000
MAX_POSITION_PCT=0.20
STOP_LOSS_PCT=0.04
DAILY_LOSS_LIMIT_PCT=0.05
```

### GitHub Secrets (for Actions)
Add at: repo → Settings → Secrets → Actions

`ALPACA_KEY` · `ALPACA_SECRET` · `TELEGRAM_TOKEN` · `TELEGRAM_CHAT_ID` · `HF_TOKEN` · `HF_REPO_ID`

---

## 🐍 Key Dependencies

```
alpaca-trade-api     # Broker API
stable-baselines3    # PPO reinforcement learning
gymnasium            # RL environment
torch                # Neural network backend
ta                   # Technical analysis indicators
scikit-learn         # Regime classifier (Random Forest)
huggingface-hub      # Model storage
gradio               # Dashboard UI
python-telegram-bot  # Telegram alerts
pandas / numpy       # Data processing
yfinance             # Backup market data
schedule             # Local task scheduling
sqlalchemy           # Database ORM
loguru               # Logging
python-dotenv        # Environment variable management
```

---

## 🗓️ Build Order (Updated — Full Stack)

| Phase | What to Build | Models / Data Added |
|---|---|---|
| 1 | ✅ Architecture planned | This document |
| 2 | ⬜ Data pipeline | Alpaca OHLCV + technical indicators |
| 3 | ⬜ Backtesting engine + metrics | — |
| 4 | ⬜ Market regime classifier | Random Forest |
| 5 | ⬜ RL Agent (PPO) | Stable Baselines3 |
| 6 | ⬜ XGBoost predictor | XGBoost / LightGBM |
| 7 | ⬜ News sentiment layer | FinBERT + NewsAPI + SEC EDGAR |
| 8 | ⬜ LSTM price predictor | PyTorch LSTM |
| 9 | ⬜ Macro signal layer | FRED API |
| 10 | ⬜ Ensemble signal combiner | Weighted voting |
| 11 | ⬜ Reddit sentiment | PRAW + FinBERT |
| 12 | ⬜ Options flow signal | Unusual Whales free tier |
| 13 | ⬜ Risk manager | Hard override rules |
| 14 | ⬜ Alpaca paper trading execution | — |
| 15 | ⬜ Telegram alerts | — |
| 16 | ⬜ GitHub Actions scheduling | — |
| 17 | ⬜ HuggingFace model storage + ZeroGPU | — |
| 18 | ⬜ Gradio dashboard on HF Spaces | — |
| 19 | ⬜ UptimeRobot keep-alive | — |
| 20 | ⬜ Confidence check → real money | — |

---

## 🧠 Prediction Models Layer (Added)

### Why Multiple Models?
Professional hedge funds don't use one model. They combine signals from multiple models into an **ensemble** — each model captures different patterns, and together they're more reliable than any single approach.

### Model 1 — XGBoost / LightGBM ✅ (Build First)
The most widely used model in quant finance. Consistently outperforms deep learning on tabular financial data.

| | Details |
|---|---|
| What it predicts | Next 5-min / daily return direction (up or down) |
| Input | All technical indicators from Layer 1 |
| Output | Probability of price going up (0.0 – 1.0) |
| Why pros use it | Handles non-linear relationships, fast, interpretable |
| Library | `xgboost`, `lightgbm` |
| Cost | Free |

```python
from xgboost import XGBClassifier

# Label: will price be higher in 5 candles?
def create_labels(df, forward_periods=5):
    df["target"] = (
        df["close"].shift(-forward_periods) > df["close"]
    ).astype(int)
    return df

model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# Probability price goes UP
prob_up = model.predict_proba(X_latest)[:, 1]
```

**File:** `bot/strategy/xgb_predictor.py`

---

### Model 2 — LSTM (Long Short-Term Memory) ✅ (Build Second)
Captures sequential patterns in price data that XGBoost misses — good at learning things like "after 3 red candles + high volume, price usually bounces."

| | Details |
|---|---|
| What it predicts | Next N price movements as a sequence |
| Input | Last 60 candles of OHLCV + indicators |
| Output | Probability of price going up |
| Why pros use it | Handles time dependencies naturally |
| Library | `torch` (PyTorch) |
| Cost | Free (trains on HF ZeroGPU) |

```python
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=19, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # probability of going up
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
```

**File:** `bot/strategy/lstm_predictor.py`

---

### Model 3 — FinBERT (News Sentiment) ✅ (High Value Add)
Specifically trained on financial text. Outperforms general LLMs on financial sentiment tasks by a large margin.

| | Details |
|---|---|
| What it predicts | Sentiment of news headlines (-1 negative → +1 positive) |
| Input | News headlines, earnings call summaries, SEC filings |
| Output | Sentiment score per stock |
| Model | `ProsusAI/finbert` on HuggingFace |
| Cost | Free |

```python
from transformers import pipeline

sentiment = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert"
)

def get_news_sentiment(headline: str) -> float:
    result = sentiment(headline)[0]
    score = result["score"]
    if result["label"] == "positive": return score
    elif result["label"] == "negative": return -score
    return 0.0
```

**File:** `bot/strategy/sentiment.py`

---

### Model 4 — Temporal Fusion Transformer (TFT) ⚠️ (Advanced — Phase 3+)
State-of-the-art for financial time series. Multi-horizon forecasts with uncertainty estimates.

| | Details |
|---|---|
| What it predicts | Returns for next 1 / 5 / 20 days with confidence intervals |
| Input | Price history + known future inputs (earnings dates, macro calendar) |
| Library | `pytorch-forecasting` |
| Complexity | High — build after simpler models are working |

---

### Ensemble Signal Combiner ✅ (The Real Edge)

Combines all model outputs into one weighted signal:

```python
def ensemble_signal(xgb_prob, lstm_prob, sentiment_score, regime):
    weights = {
        "xgb":       0.30,
        "lstm":      0.30,
        "sentiment": 0.20,
        "regime":    0.20,
    }

    regime_score = {
        "TRENDING_UP":    1.0,
        "RANGING":        0.5,
        "HIGH_VOLATILITY":0.2,
        "TRENDING_DOWN":  0.0,
    }.get(regime, 0.5)

    score = (
        weights["xgb"]       * xgb_prob +
        weights["lstm"]      * lstm_prob +
        weights["sentiment"] * (sentiment_score + 1) / 2 +
        weights["regime"]    * regime_score
    )

    if score > 0.70:   return "STRONG_BUY",  0.20
    elif score > 0.60: return "BUY",         0.12
    elif score < 0.30: return "STRONG_SELL", 1.00
    elif score < 0.40: return "SELL",        0.50
    else:              return "HOLD",         0.00
```

**File:** `bot/strategy/ensemble.py`

---

## 📡 Upgraded Architecture (With All Models)

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIGNAL GENERATION LAYER                      │
├───────────┬───────────┬───────────┬───────────┬────────────────┤
│ Technical │ XGBoost   │   LSTM    │  FinBERT  │  Macro (FRED)  │
│Indicators │ Predictor │ Predictor │ Sentiment │  + Options     │
│(existing) │  (new)    │  (new)    │  (new)    │  Flow (new)    │
└───────────┴───────────┴───────────┴───────────┴────────────────┘
                              ↓
                   Ensemble Signal Combiner
                   (weighted voting — new)
                              ↓
                  Regime Classifier (context)
                              ↓
                    RL Agent PPO (decision)
                              ↓
                  Risk Manager (hard rules)
                              ↓
                 Alpaca Execution Engine
```

---

## 📡 Free Data Sources (Added)

### 1. SEC EDGAR Filings ✅ (Free, Major Edge)
Every public company files earnings, insider trades, and institutional holdings here.

| Filing Type | Signal |
|---|---|
| 10-Q / 10-K | Earnings beat/miss, management tone (FinBERT) |
| Form 4 | Insider buying/selling |
| 13F | Institutional ownership changes |

```python
def get_recent_filings(ticker: str, form_type: str = "10-Q"):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    filings = requests.get(url).json()
    return filings["filings"]["recent"]
```

**API:** `https://efts.sec.gov` — completely free, no key needed

---

### 2. FRED Macro Data ✅ (Free)
Federal Reserve Economic Data — used by every serious quant fund.

| Indicator | FRED Series | Signal |
|---|---|---|
| Fed Funds Rate | `FEDFUNDS` | Rising = bearish for growth stocks |
| Inflation (CPI) | `CPIAUCSL` | High = Fed raises rates = bearish |
| Unemployment | `UNRATE` | Rising = recession risk |
| Yield Curve | `T10Y2Y` | Inverted (< 0) = recession signal |
| VIX | `VIXCLS` | > 30 = high fear = reduce positions |

```python
from fredapi import Fred
fred = Fred(api_key="YOUR_FREE_KEY")  # free at fred.stlouisfed.org

macro = {
    "fed_rate":    fred.get_series("FEDFUNDS").iloc[-1],
    "yield_curve": fred.get_series("T10Y2Y").iloc[-1],
    "vix":         fred.get_series("VIXCLS").iloc[-1],
}

# Yield curve inversion = be defensive
if macro["yield_curve"] < 0:
    max_position_size = 0.10  # halve position sizes
```

---

### 3. Reddit / WallStreetBets Sentiment ✅ (Free)
Retail sentiment can move small/mid cap stocks significantly.

```python
import praw

reddit = praw.Reddit(client_id="id", client_secret="secret", user_agent="bot")

def get_wsb_mentions(ticker: str) -> dict:
    subreddit = reddit.subreddit("wallstreetbets")
    mentions, scores = 0, []
    for post in subreddit.search(ticker, limit=100, time_filter="day"):
        mentions += 1
        scores.append(get_news_sentiment(post.title))
    return {
        "mentions": mentions,
        "avg_sentiment": sum(scores) / len(scores) if scores else 0
    }
```

**API:** free at `reddit.com/prefs/apps`

---

### 4. Earnings Calendar ✅ (Free via yfinance)
Never hold through earnings without knowing — massive gap risk.

```python
import yfinance as yf
from datetime import datetime, timedelta

def is_earnings_soon(ticker: str, days: int = 3) -> bool:
    stock = yf.Ticker(ticker)
    earnings = stock.calendar.get("Earnings Date")
    if earnings:
        return earnings[0] <= datetime.now() + timedelta(days=days)
    return False

# In risk manager — reduce before earnings
if is_earnings_soon(symbol, days=2):
    max_position = 0.05  # cut to 5% max
```

---

### 5. Unusual Options Activity ⚠️ (Free tier limited)
Smart money often shows up in options before the stock moves.

```python
def get_options_flow(ticker: str) -> float:
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/flow"
    headers = {"Authorization": "Bearer YOUR_FREE_TOKEN"}
    data = requests.get(url, headers=headers).json()
    call_put_ratio = data["call_volume"] / (data["put_volume"] + 1)
    return call_put_ratio  # > 2.0 = very bullish signal
```

**API:** unusualwhales.com — free tier available

---

### 6. NewsAPI ✅ (Free tier: 100 req/day)
Financial headlines for FinBERT sentiment analysis.

```python
import requests

def get_stock_news(ticker: str, company: str) -> list:
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{ticker} OR {company}",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": "YOUR_FREE_KEY",
        "language": "en"
    }
    articles = requests.get(url, params=params).json().get("articles", [])
    return [a["title"] for a in articles]
```

**API:** newsapi.org — free tier (100 requests/day)

---

### Full Free Data Stack Summary

| Source | Data | Cost | Signal Strength |
|---|---|---|---|
| Alpaca | OHLCV, real-time prices | Free | ⭐⭐⭐⭐⭐ |
| yfinance | Historical data, earnings dates | Free | ⭐⭐⭐⭐ |
| SEC EDGAR | Filings, insider trades | Free | ⭐⭐⭐⭐⭐ |
| FRED | Macro indicators | Free | ⭐⭐⭐⭐ |
| NewsAPI | Headlines for FinBERT | Free (100/day) | ⭐⭐⭐ |
| Reddit PRAW | WSB sentiment | Free | ⭐⭐⭐ |
| Unusual Whales | Options flow | Free tier | ⭐⭐⭐⭐ |

---

## 🐍 Updated Key Dependencies

```
# Existing
alpaca-trade-api     # Broker API
stable-baselines3    # PPO reinforcement learning
gymnasium            # RL environment
torch                # Neural network backend
ta                   # Technical analysis indicators
scikit-learn         # Regime classifier (Random Forest)
huggingface-hub      # Model storage
gradio               # Dashboard UI
python-telegram-bot  # Telegram alerts
pandas / numpy       # Data processing
yfinance             # Backup market data + earnings calendar
schedule             # Local task scheduling
sqlalchemy           # Database ORM
loguru               # Logging
python-dotenv        # Environment variable management

# New — Prediction Models
xgboost              # XGBoost predictor
lightgbm             # LightGBM predictor (alternative)
transformers         # FinBERT sentiment model
pytorch-forecasting  # Temporal Fusion Transformer (Phase 3+)

# New — Data Sources
fredapi              # FRED macro economic data
praw                 # Reddit API (WSB sentiment)
newsapi-python       # NewsAPI headlines
requests             # SEC EDGAR + Unusual Whales API calls
```

---

## 💬 How to Resume This Project

If continuing on a new device or new Claude session, share this file and say:

> *"Here is my ARCHITECTURE.md for my AI trading bot project. I want to continue building from Phase [X]. My email is ksri77@gmail.com and the GitHub repo is ai-trading-bot."*

Claude will have full context to continue exactly where we left off.

---

*Last updated: June 2026 — Added prediction models (XGBoost, LSTM, FinBERT, TFT), ensemble layer, and free data sources (EDGAR, FRED, Reddit, NewsAPI, Unusual Whales)*