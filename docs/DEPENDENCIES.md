# TradeGenius AI — Dependencies

Last updated: 2026-06-27

Full pinned list in `requirements.txt` (bot + dashboard) and `requirements_space.txt`
(HuggingFace Spaces subset). This file captures the intent and upgrade policy.

## Runtime Dependencies

### Broker / Execution
| Package | Version | Purpose | Upgrade policy |
|---------|---------|---------|----------------|
| alpaca-trade-api | ≥ 3.0.0 | Order submission, portfolio data, market data | Pin minor on next release |
| aiohttp | ≥ 3.9.0, < 3.10.0 | Async HTTP (alpaca dependency) | Locked to 3.9.x (3.10 breaks alpaca SSL) |

### Machine Learning
| Package | Version | Purpose | Upgrade policy |
|---------|---------|---------|----------------|
| torch | ≥ 2.12.0, < 3.0 | LSTM model inference | Pin to training env version |
| scikit-learn | ≥ 1.9.0, < 2.0 | Model calibration, preprocessing | Pin minor |
| xgboost | ≥ 3.0.0, < 4.0 | XGBoost signal model | Pin major |
| transformers | ≥ 4.35.0 | FinBERT sentiment pipeline | Pin minor |
| stable-baselines3 | (transitive) | PPO RL agent | Follow sb3 releases |
| joblib | ≥ 1.3.0 | Model serialisation | Flexible |

### Data & Finance
| Package | Version | Purpose | Upgrade policy |
|---------|---------|---------|----------------|
| yfinance | ≥ 0.2.0 | Historical prices, news, sparklines | Flexible; monitor for API breaks |
| ta | ≥ 0.11.0 | Technical indicators (RSI, ATR, EMA) | Flexible |
| fredapi | ≥ 0.5.0 | FRED macro signals (VIX, T-bills) | Flexible |
| pandas | ≥ 2.0.0 | DataFrames for features and signals | Pin major |
| numpy | ≥ 1.24.0 | Numerical ops | Pin major |
| scipy | ≥ 1.11.0 | Statistics (Sharpe calculation) | Flexible |

### Dashboard
| Package | Version | Purpose | Upgrade policy |
|---------|---------|---------|----------------|
| gradio | ≥ 4.0.0 | Dashboard UI framework | Test UI_CHANGELOG on minor bumps |
| plotly | (transitive) | Equity / PnL charts | Follows gradio |
| matplotlib | ≥ 3.7.0 | Sparklines | Flexible |

### Infrastructure
| Package | Version | Purpose | Upgrade policy |
|---------|---------|---------|----------------|
| loguru | ≥ 0.7.0 | Structured logging | Flexible |
| huggingface-hub | ≥ 0.20.0 | `trades.db` push/pull | Pin minor |
| python-telegram-bot | ≥ 20.0 | Telegram alerts | Pin major (v20 async API) |
| duckdb | ≥ 1.0.0 | Analytics columnar DB | Pin major |
| sqlalchemy | ≥ 2.0.0 | SQLite ORM helpers | Pin major |
| python-dotenv | ≥ 1.0.0 | `.env` loading | Flexible |
| schedule | ≥ 1.2.0 | In-process job scheduler | Flexible |
| requests | ≥ 2.31.0 | HTTP (Reddit, Finnhub) | Flexible |

## Development / Test Dependencies

```
pytest ≥ 7.0
pytest-cov
pytest-asyncio (optional)
```

## Python Version

Minimum: Python 3.9 (f-strings with `=`, `typing.Union` shorthand).
Tested on: Python 3.9 (GitHub Actions ubuntu-latest, 2026-06-25).

## Upgrade Protocol

1. Update version in `requirements.txt`
2. Trigger `retrain.yml` if any ML package changed (models must be rebuilt in same env)
3. Run `pytest tests/ -q` → must pass
4. Run `python tests/ui_tester.py` → 0 FAIL 0 WARN
5. Monitor next 3 trading cycles for regressions

## Known Conflicts

- `aiohttp >= 3.10` breaks `alpaca-trade-api` SSL handshake (locked to 3.9.x)
- `numpy >= 2.0` deprecated `numpy.core` — shows DeprecationWarning via `stable-baselines3`;
  harmless but watch for sb3 upgrade that resolves it
