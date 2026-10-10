# One cell. GPU T4 x2, Internet on, secret HF_TOKEN.
# Fresh start deletes the old Hugging Face weights. A later restart keeps going.

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
    subprocess.check_call(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", "main"])
    subprocess.check_call(["git", "-C", str(repo), "reset", "--hard", "origin/main"])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "open_clip_torch", "datasets", "huggingface_hub", "scikit-learn",
])
os.chdir(repo)
subprocess.check_call([
    "torchrun", "--standalone", "--nproc_per_node=2", "-m", "cindy.train",
    "--hf-repo", "Yashhh999/cindy",
    "--save-every", "50", "--hf-every", "2000",
    "--batch-size", "16", "--steps-per-era", "4000",
    "--per-gen", "2000", "--replay-per", "300",
    "--scan-chunk", "200000", "--min-free-gb", "3",
], env=os.environ.copy())
