"""Sync trades.db to/from a HuggingFace dataset repo.

Bot side:  call push_db() at the end of each trading cycle.
Space side: _con() in dashboard_data.py calls pull_db() automatically.

Default dataset repo: ksri77/ai-trading-bot-db
Set HF_DB_REPO_ID in .env or environment to override.
"""
from __future__ import annotations
import os
import shutil
import traceback
from pathlib import Path

from loguru import logger


def _get_cfg() -> tuple[str, str, str]:
    """Return (db_path, repo_id, token) from config/env."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import TRADE_DB_PATH, HF_DB_REPO_ID, HF_TOKEN
    token = HF_TOKEN or os.environ.get("HF_TOKEN", "")
    return TRADE_DB_PATH, HF_DB_REPO_ID, token


def push_db() -> bool:
    """Upload trades.db to HF dataset. Creates the repo if it doesn't exist."""
    db_path, repo_id, token = _get_cfg()
    if not token:
        logger.warning("push_db: HF_TOKEN is not set — skipping sync")
        return False
    if not repo_id:
        logger.warning("push_db: HF_DB_REPO_ID is not set — skipping sync")
        return False
    if not Path(db_path).exists():
        logger.warning(f"push_db: {db_path} does not exist — skipping sync")
        return False
    size_kb = Path(db_path).stat().st_size / 1024
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()  # tqdm flushes sys.stderr which fails on Windows non-TTY
        api = HfApi(token=token)
        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset")
        except Exception:
            logger.info(f"push_db: creating dataset repo {repo_id}")
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=db_path,
            path_in_repo="trades.db",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="bot: sync trades.db",
        )
        logger.info(f"push_db: uploaded {db_path} ({size_kb:.0f} KB) → {repo_id}")
        # Push model validation artifacts if present so the dashboard can display them
        _root = Path(__file__).parent.parent.parent
        for artifact in ("models/validation_report.json", "models/feature_importance.json",
                         "models/runtime_versions.json"):
            artifact_path = _root / artifact
            if artifact_path.exists():
                try:
                    api.upload_file(
                        path_or_fileobj=str(artifact_path),
                        path_in_repo=artifact_path.name,
                        repo_id=repo_id,
                        repo_type="dataset",
                        commit_message=f"bot: sync {artifact_path.name}",
                    )
                    logger.debug(f"push_db: synced {artifact_path.name}")
                except Exception as _ae:
                    logger.debug(f"push_db: artifact sync skipped ({artifact_path.name}): {_ae}")
        return True
    except Exception as exc:
        logger.error(f"push_db: upload failed — {exc}\n{traceback.format_exc()}")
        return False


def backup_database(backup_dir: str | None = None) -> str | None:
    """Create a timestamped local copy of trades.db.

    Returns the backup file path on success, None on failure.
    Keeps the last 30 backups; older ones are deleted automatically.

    Usage:
        from bot.monitor.sync_db import backup_database
        path = backup_database()          # → backups/trades_20260627_143000.db
        path = backup_database("/tmp/bk") # custom directory
    """
    import datetime
    db_path, _, _ = _get_cfg()
    src = Path(db_path)
    if not src.exists():
        logger.warning(f"backup_database: {db_path} not found — nothing to back up")
        return None

    dest_dir = Path(backup_dir) if backup_dir else src.parent / "backups"
    dest_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = dest_dir / f"trades_{ts}.db"

    try:
        shutil.copy2(src, dest)
        size_kb = dest.stat().st_size / 1024
        logger.info(f"backup_database: {dest} ({size_kb:.0f} KB)")

        # Prune: keep only the 30 most-recent backups
        backups = sorted(dest_dir.glob("trades_*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in backups[30:]:
            old.unlink(missing_ok=True)
            logger.debug(f"backup_database: removed old backup {old.name}")

        return str(dest)
    except Exception as exc:
        logger.error(f"backup_database: failed — {exc}")
        return None


def pull_db(force: bool = False) -> bool:
    """Download trades.db from HF dataset. Returns True on success."""
    db_path, repo_id, token = _get_cfg()
    if not repo_id:
        logger.warning("pull_db: HF_DB_REPO_ID is not set — cannot pull")
        return False
    local = Path(db_path)
    if not force and local.exists():
        import time
        age_s = time.time() - local.stat().st_mtime
        if age_s < 300:  # fresher than 5 min — skip download
            logger.debug(f"pull_db: {db_path} is {age_s:.0f}s old — skipping (use force=True to override)")
            return True
    # Run the download in a thread with a hard timeout so it never hangs the app
    import threading
    result: list[bool] = [False]
    error:  list[str]  = [""]

    def _download():
        try:
            from huggingface_hub import hf_hub_download
            tok = os.environ.get("HF_TOKEN") or token or None
            if not tok:
                error[0] = "HF_TOKEN not available in Space environment"
                return
            cached = hf_hub_download(
                repo_id=repo_id,
                filename="trades.db",
                repo_type="dataset",
                token=tok,
                force_download=True,
            )
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cached, local)
            size_kb = local.stat().st_size / 1024
            logger.info(f"pull_db: downloaded trades.db ({size_kb:.0f} KB) from {repo_id}")
            result[0] = True
        except Exception as exc:
            error[0] = str(exc)
            msg = str(exc).lower()
            if any(x in msg for x in ("404", "not found", "entry", "does not exist")):
                if local.exists():
                    local.unlink()
                    logger.info(f"pull_db: trades.db deleted from HF — removed local copy at {local}")
            logger.error(f"pull_db: download failed — {exc}\n{traceback.format_exc()}")

    t = threading.Thread(target=_download, daemon=True)
    t.start()
    t.join(timeout=20)  # give up after 20 s — never block app startup
    if not t.is_alive() and not result[0]:
        if error[0]:
            logger.error(f"pull_db: failed — {error[0]}")
        else:
            logger.error("pull_db: timed out after 20 s")
    return result[0]
