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

# --- Trading universe (live bot trades these) ---
# Balanced across 8 sectors so the model can rotate capital into whatever is leading,
# not just chase the tech names it was originally given.
STOCKS = [
    # Technology
    "AAPL", "MSFT", "NVDA",
    # Communication Services
    "GOOGL", "META",
    # Consumer Discretionary
    "AMZN", "TSLA",
    # Financials
    "JPM",
    # Healthcare
    "JNJ",
    # Energy
    "XOM",
    # Consumer Staples (defensive)
    "WMT",
]
ETFS = [
    "SPY", "QQQ", "VTI",   # broad market
    "XLF",                  # Financials ETF
    "XLV",                  # Healthcare ETF
    "XLE",                  # Energy ETF
    "GLD",                  # Gold — macro hedge / safe haven
]
SYMBOLS = STOCKS + ETFS
BENCHMARK = "SPY"

# --- Training universe (superset — more symbols = better model generalisation) ---
TRAINING_EXTRA = [
    "XLK", "XLI", "XLY", "XLP", "XLC",     # remaining GICS sector ETFs
    "TLT", "BRK-B",                          # bonds + value anchor
    "VOO", "ARKK",                           # previously in SYMBOLS, still train on them
]
TRAINING_SYMBOLS = SYMBOLS + TRAINING_EXTRA

# --- Trading parameters ---
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", 0.20))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", 0.04))       # fallback flat stop (no ATR data)
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", 0.05))
MAX_POSITIONS = 5
MAX_SECTOR_POSITIONS = 2                                        # max open positions per sector

# --- ATR-based exit rules ---
ATR_STOP_MULTIPLIER   = float(os.getenv("ATR_STOP_MULTIPLIER",   2.0))  # stop at entry - 2×ATR
ATR_TRAIL_MULTIPLIER  = float(os.getenv("ATR_TRAIL_MULTIPLIER",  1.5))  # trail at hwm - 1.5×ATR
ATR_MIN_STOP_PCT      = 0.015   # floor: never stop tighter than 1.5%
ATR_MAX_STOP_PCT      = 0.10    # ceiling: never stop wider than 10%

# --- Market timing buffers (skip volatile open/close windows) ---
MARKET_OPEN_BUFFER_MINS  = int(os.getenv("MARKET_OPEN_BUFFER_MINS",  15))
MARKET_CLOSE_BUFFER_MINS = int(os.getenv("MARKET_CLOSE_BUFFER_MINS", 10))

# --- Sector map for concentration limits ---
# MAX_SECTOR_POSITIONS=2 means at most 2 open positions per sector at any time.
SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Technology",    "MSFT": "Technology",    "NVDA": "Technology",
    "QQQ":  "Technology",    "XLK":  "Technology",    "ARKK": "Technology",
    # Communication Services
    "GOOGL": "Comm_Services", "META": "Comm_Services", "XLC": "Comm_Services",
    # Consumer Discretionary
    "AMZN": "Consumer_Disc",  "TSLA": "Consumer_Disc", "XLY": "Consumer_Disc",
    # Consumer Staples
    "WMT":  "Consumer_Staples", "XLP": "Consumer_Staples",
    # Financials
    "JPM":  "Financials",    "XLF":  "Financials",    "BRK-B": "Financials",
    # Healthcare
    "JNJ":  "Healthcare",    "XLV":  "Healthcare",
    # Energy
    "XOM":  "Energy",        "XLE":  "Energy",
    # Industrials
    "XLI":  "Industrials",
    # Broad / macro
    "SPY":  "Broad_ETF",     "VTI":  "Broad_ETF",     "VOO":  "Broad_ETF",
    "GLD":  "Commodities",   "TLT":  "Bonds",
}

INITIAL_CAPITAL      = float(os.getenv("INITIAL_CAPITAL", 10_000))
EARNINGS_WINDOW_DAYS = int(os.getenv("EARNINGS_WINDOW_DAYS", 2))   # block buys ±N days from earnings

# --- Advanced entry/exit parameters ---
MAX_HOLD_DAYS         = int(os.getenv("MAX_HOLD_DAYS", 5))          # force exit after N days with <1% gain
KELLY_LOOKBACK_TRADES = int(os.getenv("KELLY_LOOKBACK_TRADES", 30)) # trades used to estimate Kelly fraction
KELLY_FRACTION_MAX    = float(os.getenv("KELLY_FRACTION_MAX", 0.20))# half-Kelly upper cap
CORRELATION_THRESHOLD = float(os.getenv("CORRELATION_THRESHOLD", 0.85))  # block buy if corr > this with held pos
RS_LOOKBACK_BARS      = int(os.getenv("RS_LOOKBACK_BARS", 5))       # bars for 5-min relative strength vs SPY
# Only open new positions in these regimes — blocks entries in downtrends and high-vol whipsaws
ENTRY_REGIMES         = set(os.getenv("ENTRY_REGIMES", "TRENDING_UP,RANGING").split(","))

# --- RL agent ---
RL_TIMESTEPS = 1_000_000
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
