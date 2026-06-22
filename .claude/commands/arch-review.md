---
description: Review code against ai-trading-bot architecture, design rules, and coding standards
argument-hint: [file path(s) | --diff | blank = full scan]
---

You are performing an architecture and coding-standards review for **ai-trading-bot**.

## Step 1 — Run the static analyzer

```bash
cd "c:\Users\ksri7\projects\ai-trading-bot" && python scripts/arch_review.py $ARGUMENTS
```

If `$ARGUMENTS` is empty, the script scans all of `bot/`, `dashboard/`, `database/`, and `scripts/`.
Pass `--diff` to check only files changed since the last commit.

## Step 2 — Architectural rules (ARCHITECTURE.md)

The bot has a strict **6-layer, top-down** architecture. Data flows downward only:

| # | Layer | Key files | May call |
|---|-------|-----------|----------|
| 1 | Data Ingestion | `bot/strategy/features.py`, `macro.py`, `sentiment.py` | external APIs only |
| 2 | Regime Classifier | `bot/strategy/regime_classifier.py` | Layer 1 |
| 3 | RL Agent | `bot/strategy/rl_agent.py`, `ensemble.py`, `xgb_predictor.py` | Layers 1-2 |
| 4 | Risk Manager | `bot/risk/risk_manager.py` | Layers 1-3 |
| 5 | Execution Engine | `bot/execution/alpaca_client.py` | **Layer 4 only** |
| 6 | Monitoring | `bot/monitor/` | Any layer |

**Non-negotiable constraints:**
- Execution (Layer 5) **must** be gated through Risk Manager (Layer 4) — any bypass is a `[BLOCK]`
- Strategy layers (1-3) must **never** import or call `alpaca_client` directly
- `stop_loss = 0.04`, `daily_loss_limit = 0.05`, `max_position = 0.20` — changes need explicit approval
- PDT guard must remain active for accounts under $25K

## Step 3 — Coding standards

| Rule | Severity |
|------|----------|
| Hardcoded secrets / API keys | **BLOCK** |
| File > 500 lines | WARN |
| `print()` in `bot/`, `dashboard/`, `database/` (use `logging`) | WARN |
| Public function missing type hints | WARN |
| New `bot/` module without `tests/test_<module>.py` | WARN |
| Env-var names used as bare string literals (not via `os.getenv`) | WARN |
| Input validation missing at API/external boundaries | WARN |

## Step 4 — Report format

After running the script and reviewing the diff (`git diff HEAD` or `git diff --cached`), report findings as:

```
🚫 [BLOCK]   — must fix before merge (security violation, risk bypass, architecture breach)
✗  [ERROR]   — strong violation of a stated rule
⚠  [WARN]    — coding standards / missing tests / style
ℹ  [INFO]    — optional improvement
```

For each finding include: **file:line**, **rule name**, and a one-sentence fix suggestion.

Close with a summary line: `N block(s) · N error(s) · N warning(s)` and a `✓ PASS` or `✗ FAIL` verdict.

$ARGUMENTS
