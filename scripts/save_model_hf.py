"""Push trained models to HuggingFace Hub."""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from huggingface_hub import HfApi
from loguru import logger
from config import HF_TOKEN, HF_REPO_ID, MODEL_SAVE_PATH, REGIME_MODEL_PATH


def push():
    if not HF_TOKEN or not HF_REPO_ID:
        raise EnvironmentError("HF_TOKEN and HF_REPO_ID must be set.")
    api = HfApi(token=HF_TOKEN)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = os.getenv("GITHUB_RUN_ID", "local")
    commit_msg = f"auto-retrain {timestamp} (run {run_id})"

    model_files = [
        f"{MODEL_SAVE_PATH}.zip",
        REGIME_MODEL_PATH,
        "models/saved/xgb_predictor.pkl",
        "models/saved/lstm_predictor.pt",
    ]
    uploaded = []
    for path in model_files:
        if os.path.exists(path):
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=os.path.basename(path),
                repo_id=HF_REPO_ID,
                commit_message=commit_msg,
            )
            logger.info(f"Pushed {path} to {HF_REPO_ID}")
            uploaded.append(os.path.basename(path))
        else:
            logger.warning(f"File not found (skipping): {path}")

    # Upload version manifest so callers can identify which HF commit to roll back to
    info = {"timestamp": timestamp, "github_run_id": run_id, "files": uploaded}
    api.upload_file(
        path_or_fileobj=json.dumps(info, indent=2).encode(),
        path_in_repo="model_info.json",
        repo_id=HF_REPO_ID,
        commit_message=commit_msg,
    )
    logger.info(f"Pushed model_info.json (run={run_id}, timestamp={timestamp})")


if __name__ == "__main__":
    push()
