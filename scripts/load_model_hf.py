"""Pull trained models from HuggingFace Hub."""
import os
import sys
import shutil
import traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from huggingface_hub import hf_hub_download
from loguru import logger
from config import HF_TOKEN, HF_REPO_ID, MODEL_SAVE_PATH, REGIME_MODEL_PATH


def pull():
    if not HF_TOKEN or not HF_REPO_ID:
        raise EnvironmentError("HF_TOKEN and HF_REPO_ID must be set.")
    for filename, dest in [
        ("regime_classifier.pkl", REGIME_MODEL_PATH),
        ("xgb_predictor.pkl", "models/saved/xgb_predictor.pkl"),
        ("lstm_predictor.pt", "models/saved/lstm_predictor.pt"),
        ("lstm_scaler.pkl", "models/saved/lstm_scaler.pkl"),
    ]:
        try:
            cached = hf_hub_download(repo_id=HF_REPO_ID, filename=filename, token=HF_TOKEN)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(cached, dest)
            logger.info(f"Downloaded {filename} → {dest}")
        except Exception as e:
            logger.warning(f"Could not download {filename}: {e}")


if __name__ == "__main__":
    try:
        pull()
    except Exception:
        tb = traceback.format_exc()
        logger.error("load_model_hf failed:\n" + tb)
        print(f"::error title=HuggingFace Pull Failed::{tb.splitlines()[-1]} — see step log for full traceback", flush=True)
        sys.exit(1)
