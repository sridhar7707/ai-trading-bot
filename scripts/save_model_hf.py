"""Push trained models to HuggingFace Hub as a single atomic commit."""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from huggingface_hub import HfApi, CommitOperationAdd
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

    ops = []
    uploaded = []
    for path in model_files:
        if os.path.exists(path):
            ops.append(CommitOperationAdd(
                path_in_repo=os.path.basename(path),
                path_or_fileobj=path,
            ))
            uploaded.append(os.path.basename(path))
            logger.info(f"Queued {path} for upload")
        else:
            logger.warning(f"File not found (skipping): {path}")

    if not ops:
        logger.error("No model files found — aborting push")
        return

    info = {"timestamp": timestamp, "github_run_id": run_id, "files": uploaded}
    ops.append(CommitOperationAdd(
        path_in_repo="model_info.json",
        path_or_fileobj=json.dumps(info, indent=2).encode(),
    ))

    api.create_commit(
        repo_id=HF_REPO_ID,
        operations=ops,
        commit_message=commit_msg,
    )
    logger.info(f"Pushed {len(uploaded)} model(s) + model_info.json in one commit (run={run_id})")


if __name__ == "__main__":
    push()
