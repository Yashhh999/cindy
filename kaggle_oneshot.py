# One cell for a Kaggle notebook with the GPU T4 x2 accelerator and Internet on.
# Add-ons -> Secrets -> HF_TOKEN = a write token for the Hugging Face model Yashhh999/cindy.

import os
import subprocess
import sys
from pathlib import Path

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not token:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HF_TOKEN"] = token
os.environ["HUGGING_FACE_HUB_TOKEN"] = token
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

repo = Path("/kaggle/working/cindy")
if not (repo / "cindy" / "train.py").exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/Yashhh999/cindy.git", str(repo)])
else:
    subprocess.check_call(["git", "-C", str(repo), "pull", "--ff-only"])

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "open_clip_torch", "datasets", "huggingface_hub", "scikit-learn",
])
os.chdir(repo)
env = os.environ.copy()
subprocess.check_call([
    "torchrun", "--standalone", "--nproc_per_node=2", "-m", "cindy.train",
    "--hf-repo", "Yashhh999/cindy",
    "--save-every", "50",
    "--hf-every", "2000",
    "--batch-size", "16",
    "--max-steps", "8000",
    "--dragon-train", "160000",
    "--per-model", "8000",
    "--v2-total", "70000",
    "--v2-per-gen", "8000",
    "--max-scan", "1200000",
    "--holdout", "lumina",
    "--v2-root", "/kaggle/input",
], env=env)
