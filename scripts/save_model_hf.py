"""Push trained models to HuggingFace Hub."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from huggingface_hub import HfApi
from loguru import logger
from config import HF_TOKEN, HF_REPO_ID, MODEL_SAVE_PATH, REGIME_MODEL_PATH
import os


def push():
    if not HF_TOKEN or not HF_REPO_ID:
        raise EnvironmentError("HF_TOKEN and HF_REPO_ID must be set.")
    api = HfApi(token=HF_TOKEN)
    model_files = [
        f"{MODEL_SAVE_PATH}.zip",   # PPO RL agent
        REGIME_MODEL_PATH,           # Random Forest regime classifier
        "models/saved/xgb_predictor.pkl",
        "models/saved/lstm_predictor.pt",
    ]
    for path in model_files:
        if os.path.exists(path):
            api.upload_file(path_or_fileobj=path, path_in_repo=os.path.basename(path), repo_id=HF_REPO_ID)
            logger.info(f"Pushed {path} to {HF_REPO_ID}")
        else:
            logger.warning(f"File not found (skipping): {path}")


if __name__ == "__main__":
    push()
