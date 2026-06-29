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
HF_DB_REPO_ID = os.getenv("HF_DB_REPO_ID", "ksri77/ai-trading-bot-db")

# --- Trading universe (live bot trades these) ---
# Balanced across 8 sectors so the model can rotate capital into whatever is leading.
STOCKS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "CRM",
    # Communication Services
    "GOOGL", "META", "NFLX",
    # Consumer Discretionary
    "AMZN", "TSLA", "NKE", "MCD",
    # Consumer Staples (defensive)
    "WMT", "COST", "PG",
    # Financials
    "JPM", "BAC", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "ABBV", "PFE",
    # Energy
    "XOM", "CVX",
    # Industrials
    "CAT", "HON",
]
ETFS = [
    "SPY", "QQQ", "VTI",   # broad market
    "IWM",                  # Small-cap Russell 2000
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

# --- Paper sim capital ---
# When > 0, the bot SIZES and RISK-CHECKS as if the account held this much, even
# though the real (paper) account is larger. Lets you dry-run small-account
# mechanics — tiny position sizes, min-notional rejections, and the PDT day-trade
# limit (which applies under $25k) — before going live with a small balance.
# 0 = disabled (use the real account value). Example: PAPER_SIM_CAPITAL=1000
PAPER_SIM_CAPITAL = float(os.getenv("PAPER_SIM_CAPITAL", "0") or 0)

# --- Trading parameters ---
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", 0.20))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", 0.04))       # fallback flat stop (no ATR data)
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", 0.05))
DAILY_LOSS_WARNING_PCT = DAILY_LOSS_LIMIT_PCT * 0.50          # warn at 50% of daily limit
WEEKLY_LOSS_LIMIT_PCT = float(os.getenv("WEEKLY_LOSS_LIMIT_PCT", 0.10))  # 10% weekly circuit breaker
PORTFOLIO_DRAWDOWN_LIMIT_PCT = float(os.getenv("PORTFOLIO_DRAWDOWN_LIMIT_PCT", 0.12))  # 12% from all-time high
MACRO_HALT_VIX = float(os.getenv("MACRO_HALT_VIX", 28.0))    # halt all new buys above this VIX level
MAX_RISK_PER_TRADE_PCT  = float(os.getenv("MAX_RISK_PER_TRADE_PCT",  0.015))  # max 1.5% of portfolio at risk per trade
MIN_RR_RATIO            = float(os.getenv("MIN_RR_RATIO",            1.0))    # 1× R:R for 1-week holds (was 1.5 for 3-week; tighter targets need looser gate)
MIN_TP_PCT              = float(os.getenv("MIN_TP_PCT",              0.01))   # 1% TP floor for short-term swing trades (was 5% for 3-week momentum holds)
RANGING_SIZE_FACTOR     = float(os.getenv("RANGING_SIZE_FACTOR",     0.75))   # reduce position by 25% in sideways markets
MAX_SECTOR_EXPOSURE_PCT = float(os.getenv("MAX_SECTOR_EXPOSURE_PCT", 0.30))   # max 30% of portfolio in any one sector
MAX_POSITION_DRIFT_PCT  = float(os.getenv("MAX_POSITION_DRIFT_PCT",  0.25))   # trim position back if it drifts above 25%
MIN_CASH_RESERVE_PCT    = float(os.getenv("MIN_CASH_RESERVE_PCT",    0.10))   # always keep 10% cash uninvested
MAX_POSITIONS = 8
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
    "AAPL": "Technology",  "MSFT": "Technology",  "NVDA": "Technology",
    "AMD":  "Technology",  "AVGO": "Technology",  "CRM":  "Technology",
    "QQQ":  "Technology",  "XLK":  "Technology",  "ARKK": "Technology",
    "PANW": "Technology",  "SNOW": "Technology",  "CSCO": "Technology",
    "NOW":  "Technology",  "ADBE": "Technology",  "INTU": "Technology",
    # Communication Services
    "GOOGL": "Comm_Services", "META": "Comm_Services", "NFLX": "Comm_Services",
    "XLC":   "Comm_Services", "DIS":  "Comm_Services", "BKNG": "Comm_Services",
    # Consumer Discretionary
    "AMZN": "Consumer_Disc", "TSLA": "Consumer_Disc",
    "NKE":  "Consumer_Disc", "MCD":  "Consumer_Disc", "XLY":  "Consumer_Disc",
    "HD":   "Consumer_Disc", "TGT":  "Consumer_Disc", "LOW":  "Consumer_Disc",
    "ABNB": "Consumer_Disc", "LULU": "Consumer_Disc",
    # Consumer Staples
    "WMT":  "Consumer_Staples", "COST": "Consumer_Staples",
    "PG":   "Consumer_Staples", "XLP":  "Consumer_Staples",
    "SBUX": "Consumer_Staples", "KO":   "Consumer_Staples", "PEP": "Consumer_Staples",
    # Financials
    "JPM":   "Financials", "BAC": "Financials", "V":   "Financials",
    "MA":    "Financials", "XLF": "Financials", "BRK-B": "Financials",
    "MS":    "Financials", "GS":  "Financials", "WFC": "Financials",
    "BLK":   "Financials", "AXP": "Financials",
    # Healthcare
    "JNJ":  "Healthcare", "UNH":  "Healthcare", "ABBV": "Healthcare",
    "PFE":  "Healthcare", "XLV":  "Healthcare", "LLY":  "Healthcare",
    "MRK":  "Healthcare", "TMO":  "Healthcare",
    # Energy
    "XOM":  "Energy", "CVX": "Energy", "XLE": "Energy",
    "COP":  "Energy", "SLB": "Energy",
    # Industrials
    "CAT":  "Industrials", "HON": "Industrials", "XLI": "Industrials",
    "GE":   "Industrials", "MMM": "Industrials", "DE":  "Industrials",
    "RTX":  "Industrials", "LMT": "Industrials", "BA":  "Industrials",
    # Broad / macro
    "SPY":  "Broad_ETF", "VTI": "Broad_ETF", "VOO": "Broad_ETF",
    "IWM":  "Broad_ETF",
    "GLD":  "Commodities", "TLT": "Bonds",
}

INITIAL_CAPITAL      = float(os.getenv("INITIAL_CAPITAL", 10_000))
EARNINGS_WINDOW_DAYS = int(os.getenv("EARNINGS_WINDOW_DAYS", 2))   # block buys ±N days from earnings

# --- Advanced entry/exit parameters ---
MAX_HOLD_DAYS         = int(os.getenv("MAX_HOLD_DAYS", 7))           # force exit after N days with <1% gain (7 = 1-week max for short-term swing trades)
KELLY_LOOKBACK_TRADES = int(os.getenv("KELLY_LOOKBACK_TRADES", 30)) # trades used to estimate Kelly fraction
KELLY_FRACTION_MAX    = float(os.getenv("KELLY_FRACTION_MAX", 0.20))# half-Kelly upper cap
CORRELATION_THRESHOLD   = float(os.getenv("CORRELATION_THRESHOLD",   0.85))   # block buy if corr > this with held pos (relaxed from 0.80 to allow faster turnover)
MACD_CONFIRMATION_MIN   = float(os.getenv("MACD_CONFIRMATION_MIN",   "-inf"))  # Gate 7.9: block entry if daily macd_diff <= this; -inf = disabled (default)
RS_LOOKBACK_BARS      = int(os.getenv("RS_LOOKBACK_BARS", 5))       # bars for 5-min relative strength vs SPY
# Regimes that allow new long entries. HIGH_VOLATILITY is included because the
# risk manager still caps position size and the stop-loss hard-overrides any exit.
ENTRY_REGIMES         = set(os.getenv("ENTRY_REGIMES", "TRENDING_UP,RANGING,HIGH_VOLATILITY").split(","))
# Minimum partial-day volume relative to 20-day average daily volume. Bot uses
# daily OHLCV bars from yfinance; at 10 AM ET only ~30-40% of daily volume has
# traded, so 0.7 never cleared until early afternoon. 0.3 = ~9:45-10:15 AM ET.
MIN_VOLUME_RATIO      = float(os.getenv("MIN_VOLUME_RATIO", 0.3))

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
# Finnhub free tier — used by screener for analyst upgrade/downgrade signals.
# Register at https://finnhub.io (free, no credit card). Add as GitHub secret FINNHUB_API_KEY.
# Without it the screener still runs; analyst factor is simply skipped.
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
