# TradeGenius AI — Backup & Recovery

Last updated: 2026-06-27

## What Needs Backing Up

| Data | Location | Criticality | Backup method |
|------|----------|-------------|---------------|
| Trade history and portfolio state | `trades.db` (SQLite) | **Critical** | Local `backup_database()` + HuggingFace push |
| Analytics history | `analytics.duckdb` | Low | Derived from `trades.db`; rebuildable |
| ML models | `models/saved/` | High | GitHub (checked in); retrainable from data |
| Configuration | `config.py`, `.env` | Critical | GitHub (config.py); secrets in GH Actions / HF secrets |
| Documentation | `docs/` | Low | GitHub (checked in) |

## Local Backup

### Creating a Backup

```python
# One-liner
from bot.monitor.sync_db import backup_database
path = backup_database()   # → backups/trades_YYYYMMDD_HHMMSS.db
```

Or from the command line:

```bash
python -c "from bot.monitor.sync_db import backup_database; print(backup_database())"
```

### Backup Location

Default: `backups/` directory in the project root.
Custom: pass a directory path as argument.

```python
backup_database("/path/to/external/drive")
```

### Rotation Policy

`backup_database()` automatically deletes backups beyond the 30 most-recent.
At ~370 KB per backup, 30 backups ≈ 11 MB.

### First Backup Run (2026-06-27)

```
backup_database: backups/trades_20260627_065830.db (372 KB)
```

## Remote Backup (HuggingFace)

Every trading cycle, `bot/monitor/sync_db.py:push_db()` uploads `trades.db` to the
private HuggingFace dataset repo (`HF_DB_REPO_ID`). This serves as the primary
offsite backup and is also the source for the dashboard.

To force a manual push:

```bash
python -c "from bot.monitor.sync_db import push_db; push_db()"
```

## Recovery Procedures

### Scenario 1: Local trades.db corrupted or deleted

```bash
# Pull from HuggingFace (the latest bot push)
python -c "from bot.monitor.sync_db import pull_db; pull_db(force=True)"
```

### Scenario 2: HuggingFace copy also stale/missing

```bash
# Restore from most-recent local backup
cp backups/$(ls -t backups/ | head -1) trades.db
```

### Scenario 3: trades.db schema needs update (new column added)

```bash
# 1. Back up current DB before schema change
python -c "from bot.monitor.sync_db import backup_database; backup_database()"

# 2. Apply the migration manually in SQLite
sqlite3 trades.db "ALTER TABLE trades ADD COLUMN new_col TEXT DEFAULT NULL;"

# 3. Verify
sqlite3 trades.db ".schema trades"

# 4. Push updated DB to HuggingFace
python -c "from bot.monitor.sync_db import push_db; push_db()"
```

### Scenario 4: Full environment rebuild (new machine or Actions runner)

```bash
# 1. Clone repo
git clone https://github.com/ksri77/ai-trading-bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set secrets in .env (copy from secure store)
# ALPACA_KEY, ALPACA_SECRET, HF_TOKEN, HF_DB_REPO_ID, TELEGRAM_TOKEN, etc.

# 4. Pull latest DB from HuggingFace
python -c "from bot.monitor.sync_db import pull_db; pull_db(force=True)"

# 5. Verify
python -c "import sqlite3; c=sqlite3.connect('trades.db'); print(c.execute('SELECT COUNT(*) FROM trades').fetchone())"
```

## Recommended Backup Schedule

| Frequency | Action |
|-----------|--------|
| Every trading cycle (~5 min) | `push_db()` runs automatically |
| Daily (before market open) | Add `backup_database()` call to pre-market job |
| Before any schema migration | Manual `backup_database()` + verify count |
| Before any dependency upgrade | Manual `backup_database()` |
| Weekly | Verify HuggingFace dataset repo has recent trades.db |

## What Is NOT Backed Up Automatically

- `.env` file — store in a password manager or GitHub Actions secrets
- `analytics.duckdb` — ephemeral; rebuilt from trades.db via `AnalyticsService`
- Bot log files (`logs/`) — rotated; not critical for recovery
