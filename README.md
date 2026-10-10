# Cindy

Detector for real vs AI-generated images, sized for **2× NVIDIA T4 (16 GB)**.

The backbone is OpenCLIP **ViT-B/16** (`laion2b_s34b_b88k`, about 86M). It stays frozen. Rank-8 LoRA on query and value is the only backbone update (about 0.3M). A small CNN reads three forensic cues that both old and 2025 generators leave behind:

- high-pass residual (VAE / upsampling texture)
- fixed SRM filters (classic noise residuals, including older GAN traces)
- log FFT magnitude (periodic spectral peaks)

Those are fused with the CLIP class token and the mean patch token. Trainable total is about **1.3M**. Input is 224, fp16, gradient checkpointing on.

## What it trains on

- **[DRAGON](https://huggingface.co/datasets/lesc-unifi/dragon)** (May 2025): **160,000** training fakes, at most 8,000 from any one generator. That covers SD 1.5 and SD 2.1, SDXL and its turbo/lightning/flash variants, SD3, Stable Cascade, SSD-1B, Kandinsky 3, Kolors, DeepFloyd IF, PixArt-α / Σ, Flux.1 schnell, LCM, Hyper-SD, Juggernaut XL, Realistic Stock Photo.
- **Lumina is held out** of training and kept for validation.
- **AI-Generated Image Detection v2** (June 2026): **70,000** images, split half real and half fake when both sides exist (paired COCO / ImageNet reals with SD 1.5, SDXL, FLUX.1-schnell, Kandinsky 2.2, PixArt-Sigma, Stable Cascade). Attach it under `/kaggle/input`. IEEE DataPort is login-gated, so the loader cannot download it. Folder or CSV names containing `real` / `fake` / `flux` / `sdxl` / `kandinsky` / `pixart` / `cascade` are picked up.
- **Reals:** COCO val2017, Flickr30k, v2 reals, and optional `--real-root`.

Training augmentations are JPEG 60–95, random crop, downscale-upscale, and blur, so phone and social-media uploads are not a separate domain.

## Checkpoints

Every 50 steps the trainer writes `/kaggle/working/cindy_ckpts/latest.pt` (LoRA + frequency CNN + head + Adam state, not the frozen CLIP weights). Hugging Face gets one commit every 2000 steps (`latest.pt`, `latest.json`, and `best.pt` together) so the hub rate limit is not burned. A 429 is retried with backoff.

Resume rule:

1. If `cindy_ckpts/latest.pt` already exists on the Kaggle disk, resume from that. Hugging Face is not pulled.
2. If the Kaggle disk is clean, download `checkpoints/latest.pt` from Hugging Face and continue.
3. If Hugging Face has no checkpoint yet, start at step 0.
4. SIGTERM (Kaggle killing the session) saves and pushes before exit.
5. If the saved step is already at `--max-steps`, the target is extended by another `--max-steps` so a finished run does not immediately exit.

`--best.pt` is written locally whenever validation AUC improves and is included in the next Hugging Face commit. Lumina AUC is the generalization check. In-domain accuracy is not.

## Kaggle

Accelerator **GPU T4 x2**, Internet **on**. Add a secret named `HF_TOKEN` (write token for the Hugging Face model `Yashhh999/cindy`), or attach the Hugging Face account so `HF_TOKEN` is already in the environment.

The repo must be public, or Kaggle cannot `git clone` it. One cell: [kaggle_oneshot.py](kaggle_oneshot.py).

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
    subprocess.check_call(["git", "-C", str(repo), "pull", "--ff-only"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch", "datasets", "huggingface_hub", "scikit-learn"])
os.chdir(repo)
subprocess.check_call([
    "torchrun", "--standalone", "--nproc_per_node=2", "-m", "cindy.train",
    "--hf-repo", "Yashhh999/cindy",
    "--save-every", "50", "--hf-every", "2000",
    "--batch-size", "16", "--max-steps", "8000",
    "--dragon-train", "160000", "--per-model", "8000",
    "--v2-total", "70000", "--v2-per-gen", "8000",
    "--max-scan", "1200000", "--holdout", "lumina",
    "--v2-root", "/kaggle/input",
], env=os.environ.copy())
```

If step 16 OOMs, change `--batch-size` to `8`. Caching 160k DRAGON JPEGs is the slow part (a few hours, about 8 GB). The 500-per-model cache from the earlier run is ignored and rebuilt. Training 8000 steps after that is about an hour at 0.2 s/step. Re-running the same cell on a **new** Kaggle session continues from Hugging Face. Re-running it in a session that still has `cindy_ckpts` continues from disk.

Score a folder after training:

```bash
python -m cindy.predict /path/to/images
```
