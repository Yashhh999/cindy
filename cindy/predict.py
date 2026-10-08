"""Score a folder of images with the latest local or Hugging Face checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from cindy.checkpoint import find_local, hf_token, resolve_resume
from cindy.model import build_model
from cindy.train import load_checkpoint

VAL = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--hf-repo", default="Yashhh999/cindy")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(grad_checkpoint=False).to(device)
    path = find_local()
    if path is None:
        if not hf_token():
            raise SystemExit("No local checkpoint. Set HF_TOKEN to pull Yashhh999/cindy.")
        path = resolve_resume(args.hf_repo)
    if path is None:
        raise SystemExit("No checkpoint found.")
    step, auc = load_checkpoint(model, None, None, path, device)
    print(f"loaded step {step} best_auc {auc:.4f} from {path}")
    model.eval()
    paths = [p for p in sorted(args.folder.rglob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    with torch.no_grad():
        for start in range(0, len(paths), args.batch_size):
            batch = paths[start : start + args.batch_size]
            images = torch.stack([VAL(Image.open(path).convert("RGB")) for path in batch]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prob = torch.sigmoid(model(images)).float().cpu()
            for path, score in zip(batch, prob.tolist()):
                label = "fake" if score >= 0.5 else "real"
                print(f"{score:.4f}\t{label}\t{path}")


if __name__ == "__main__":
    main()
