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
        return True
    except Exception as exc:
        logger.error(f"push_db: upload failed — {exc}\n{traceback.format_exc()}")
        return False


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
