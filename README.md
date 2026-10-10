# Cindy

Real vs AI-image detector for **2× NVIDIA T4**. The base is frozen OpenCLIP **ViT-B/16** (`laion2b_s34b_b88k`). It is not trained and it is not stored in the checkpoint. Rank-8 LoRA on query and value, a forensic CNN (high-pass, SRM, FFT), and a small head are the only trained weights, about **1.3M**. A second head is trained on the forensic vector alone so old GAN texture cannot be ignored.

## Data

[OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) `core/train` only. `core/test` and `reddit` are never read.

Held out even if they appear in train: GPT Image 1.5 and 2.0, nano-banana-pro, Flux.2 Klein 9B, Z-Image Turbo, Recraft v2/v3, Midjourney 7, Ideogram 2.0.

One pass over `core/train`. A chunk fills the free disk, about 200,000 images, then trains for one epoch and is deleted. Replay keeps 300 images per generator. No per-generator cap. The checkpoint stays about 20–40 MB.

## Restart

`cindy_ckpts/progress.json` stores the era, the stream row, per-generator counts, and the step. Local files win. A clean disk downloads that progress from `Yashhh999/cindy` and continues. A brand new run deletes the old Hugging Face weights first and does not load the v1 checkpoint.

Hugging Face uploads are one commit every 2,000 steps (`latest.pt`, `best.pt`, `progress.json`).

## Kaggle

GPU T4 x2, Internet on, secret `HF_TOKEN`. Cell: [kaggle_oneshot.py](kaggle_oneshot.py).

```python
import os, subprocess, sys
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
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch", "datasets", "huggingface_hub", "scikit-learn"])
os.chdir(repo)
subprocess.check_call([
    "torchrun", "--standalone", "--nproc_per_node=2", "-m", "cindy.train",
    "--hf-repo", "Yashhh999/cindy",
    "--save-every", "50", "--hf-every", "2000",
    "--batch-size", "16", "--chunk", "250000",
    "--per-gen", "0", "--replay-per", "300",
    "--min-free-gb", "3",
], env=os.environ.copy())
```

Score a folder with `python -m cindy.predict /path/to/images`.
