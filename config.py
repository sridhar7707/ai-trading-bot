import os
from dotenv import load_dotenv

load_dotenv()

# --- Broker ---
ALPACA_KEY = os.getenv("ALPACA_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# --- Telegram ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- HuggingFace ---
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_REPO_ID = os.getenv("HF_REPO_ID", "")

# --- Trading universe ---
STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
ETFS = ["VOO", "QQQ", "SPY", "VTI", "ARKK"]
SYMBOLS = STOCKS + ETFS
BENCHMARK = "SPY"

# --- Trading parameters ---
TRADING_MODE = os.getenv("TRADING_MODE", "paper")
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", 1000))
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", 0.20))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", 0.04))
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", 0.05))
MAX_POSITIONS = 5
MAX_STOCKS_PER_SECTOR = 2

# --- RL agent ---
RL_TIMESTEPS = 100_000
RL_LEARNING_RATE = 3e-4
RL_N_STEPS = 2048
RL_BATCH_SIZE = 64
RL_N_EPOCHS = 10

# --- PDT rule ---
PDT_MAX_DAY_TRADES = 3
PDT_WINDOW_DAYS = 5

# --- Paths ---
MODEL_SAVE_PATH = "models/saved/ppo_trading_bot"
REGIME_MODEL_PATH = "models/saved/regime_classifier.pkl"
XGB_MODEL_PATH = "models/saved/xgb_predictor.pkl"
LSTM_MODEL_PATH = "models/saved/lstm_predictor.pt"
TRADE_DB_PATH = "trades.db"

# --- External signal APIs (all free) ---
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "ai-trading-bot/1.0")
