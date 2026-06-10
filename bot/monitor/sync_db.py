"""Sync trades.db to/from a HuggingFace dataset repo.

Bot side:  call push_db() at the end of each trading cycle.
Space side: _con() in dashboard_data.py calls pull_db() automatically.

Default dataset repo: ksri77/ai-trading-bot-db
Set HF_DB_REPO_ID in .env or environment to override.
"""
from __future__ import annotations
import os
import shutil
from pathlib import Path


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
    if not token or not repo_id:
        return False
    if not Path(db_path).exists():
        return False
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset")
        except Exception:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=db_path,
            path_in_repo="trades.db",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="bot: sync trades.db",
        )
        return True
    except Exception:
        return False


def pull_db(force: bool = False) -> bool:
    """Download trades.db from HF dataset. Returns True on success."""
    db_path, repo_id, token = _get_cfg()
    if not repo_id:
        return False
    local = Path(db_path)
    if not force and local.exists():
        import time
        if time.time() - local.stat().st_mtime < 300:  # fresher than 5 min
            return True
    # Run the download in a thread with a hard timeout so it never hangs the app
    import threading
    result: list[bool] = [False]

    def _download():
        try:
            from huggingface_hub import hf_hub_download
            # token from env takes precedence (set as Space secret or in .env)
            tok = os.environ.get("HF_TOKEN") or token or None
            cached = hf_hub_download(
                repo_id=repo_id,
                filename="trades.db",
                repo_type="dataset",
                token=tok,
                force_download=True,
            )
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cached, local)
            result[0] = True
        except Exception:
            pass

    t = threading.Thread(target=_download, daemon=True)
    t.start()
    t.join(timeout=20)  # give up after 20 s — never block app startup
    return result[0]
