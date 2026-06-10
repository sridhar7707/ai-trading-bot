"""
One-shot deploy of the dashboard to HuggingFace Spaces.

Usage:  python scripts/deploy_space.py
Reads HF_TOKEN and HF_SPACE_ID from .env (or environment variables).
Default SPACE_ID: ksri77/ai-trading-bot
"""
import sys
import shutil
import os
from pathlib import Path

# Load .env if present
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

token    = os.environ.get("HF_TOKEN", "")
space_id = os.environ.get("HF_SPACE_ID", "ksri77/ai-trading-bot")

if not token:
    print("ERROR: HF_TOKEN not found in .env or environment.")
    sys.exit(1)

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Installing huggingface-hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub>=0.20.0", "-q"])
    from huggingface_hub import HfApi

root = Path(__file__).parent.parent

# Build staging directory
staging = Path("/tmp/hf_space_deploy")
if staging.exists():
    shutil.rmtree(staging)
staging.mkdir(parents=True)

# dashboard.py → app.py
shutil.copy(root / "scripts" / "dashboard.py", staging / "app.py")

# Full bot package
shutil.copytree(root / "bot", staging / "bot")

# config.py + minimal Space-only requirements (no ML/trading packages)
shutil.copy(root / "config.py",               staging / "config.py")
shutil.copy(root / "requirements_space.txt",  staging / "requirements.txt")

# Space README (only written if not already in the Space)
readme = staging / "README.md"
readme.write_text(
    "---\n"
    "title: AI Trading Bot Dashboard\n"
    "emoji: \U0001f4ca\n"
    "colorFrom: blue\n"
    "colorTo: indigo\n"
    "sdk: gradio\n"
    "sdk_version: 4.44.1\n"
    "app_file: app.py\n"
    "pinned: false\n"
    "---\n",
    encoding="utf-8",
)

print(f"Deploying to https://huggingface.co/spaces/{space_id} ...")
api = HfApi(token=token)
api.upload_folder(
    folder_path=str(staging),
    repo_id=space_id,
    repo_type="space",
    commit_message="chore: deploy dashboard",
)
print(f"Done → https://huggingface.co/spaces/{space_id}")
