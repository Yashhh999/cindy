"""Train Cindy on 2x T4. Checkpoints every few steps.

Resume rule: if /kaggle/working/cindy_ckpts has a file, use it.
If that folder is empty, download checkpoints/latest.pt from the HF model repo.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms

from cindy.checkpoint import Uploader, atomic_torch_save, ckpt_dir, hf_token, resolve_resume
from cindy.data import prepare
from cindy.model import ARCH, build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-repo", default="Yashhh999/cindy")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--hf-every", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=8000)
    p.add_argument("--lr-lora", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--per-model", type=int, default=500)
    p.add_argument("--val-per-model", type=int, default=40)
    p.add_argument("--holdout-val", type=int, default=200)
    p.add_argument("--holdout", default="lumina")
    p.add_argument("--max-scan", type=int, default=300000)
    p.add_argument("--real-count", type=int, default=8000)
    p.add_argument("--v2-root", default="")
    p.add_argument("--real-root", default="")
    p.add_argument("--v2-per-gen", type=int, default=1500)
    p.add_argument("--val-every", type=int, default=400)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--no-grad-checkpoint", action="store_true")
    p.add_argument("--no-coco", action="store_true")
    p.add_argument("--no-flickr", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args, _unknown = p.parse_known_args()
    if args.smoke:
        args.per_model = 4
        args.val_per_model = 2
        args.holdout_val = 2
        args.max_steps = 4
        args.save_every = 2
        args.hf_every = 2
        args.real_count = 8
        args.max_scan = 400
        args.num_workers = 0
        args.batch_size = 2
        args.val_every = 2
    return args


def init_dist():
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    if "RANK" not in os.environ:
        return 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    return rank, world, local


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def jpeg_pil(image, quality):
    import io
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class ListDataset(Dataset):
    def __init__(self, items, train: bool):
        self.items = items
        self.train = train
        self.crop = transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1))
        self.val = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def _load(self, path):
        image = Image.open(path).convert("RGB")
        if not self.train:
            return self.val(image)
        image = self.crop(image)
        if random.random() < 0.5:
            image = transforms.functional.hflip(image)
        if random.random() < 0.35:
            side = random.choice((128, 160, 192))
            image = image.resize((side, side), Image.BILINEAR).resize((224, 224), Image.BICUBIC)
        if random.random() < 0.25:
            image = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))(image)
        if random.random() < 0.85:
            image = jpeg_pil(image, random.randint(60, 95))
        return transforms.ToTensor()(image)

    def __getitem__(self, index):
        path, label = self.items[index]
        try:
            return self._load(path), float(label)
        except Exception:
            return torch.zeros(3, 224, 224), float(label)


def make_balanced(manifest):
    reals = [(p, 0) for p in manifest["train_real"]]
    fakes = [(item["path"], 1) for item in manifest["train_fake"]]
    n = 2 * max(len(reals), len(fakes))
    items = []
    for i in range(n // 2):
        items.append(reals[i % len(reals)])
        items.append(fakes[i % len(fakes)])
    return items


def make_val(manifest):
    items = [(p, 0) for p in manifest["val_real"]]
    items += [(item["path"], 1) for item in manifest["val_fake"]]
    return items


def binary_auc(labels, scores):
    labels = np.asarray(labels).astype(np.int32)
    scores = np.asarray(scores).astype(np.float64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1
        ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_pos = ranks[labels == 1].sum()
    return float((sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def evaluate(model, loader, device):
    was_training = model.training
    model.eval()
    scores, labels = [], []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logit = model(images)
        scores.append(torch.sigmoid(logit.float()).cpu())
        labels.append(target)
    model.train(was_training)
    y = torch.cat(labels).numpy()
    s = torch.cat(scores).numpy()
    pred = (s >= 0.5).astype(np.float64)
    acc = float((pred == y).mean()) if len(y) else float("nan")
    return {"acc": acc, "auc": binary_auc(y, s), "n": int(len(y))}


def set_lr(opt, step, args):
    if step < args.warmup:
        scale = step / max(1, args.warmup)
    else:
        progress = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        scale = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    bases = [args.lr_lora, args.lr_head]
    for group, base in zip(opt.param_groups, bases):
        group["lr"] = base * scale


def trainable_params(model):
    lora, head = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora.append(param)
        else:
            head.append(param)
    return lora, head


def save_checkpoint(model, opt, scaler, step, best_auc, args, path: Path):
    raw = model.module if isinstance(model, DDP) else model
    payload = {
        "arch": ARCH,
        "step": step,
        "best_auc": best_auc,
        "model": raw.trainable_state(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
    }
    atomic_torch_save(payload, path)
    meta = {"step": step, "best_auc": best_auc, "arch": ARCH, "time": time.time()}
    (path.parent / "latest.json").write_text(json.dumps(meta))


def load_checkpoint(model, opt, scaler, path: Path, device):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if ckpt.get("arch") != ARCH:
        raise RuntimeError(f"checkpoint arch {ckpt.get('arch')} != {ARCH}")
    model.load_trainable(ckpt["model"])
    if opt is not None and ckpt.get("optimizer"):
        opt.load_state_dict(ckpt["optimizer"])
        for state in opt.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("step", 0)), float(ckpt.get("best_auc", 0.0))


def main():
    args = parse_args()
    rank, world, local = init_dist()
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    seed = 42 + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if rank == 0:
        if not hf_token():
            raise SystemExit("Set HF_TOKEN (write access to the Yashhh999/cindy model repo).")
        manifest = prepare(args, log=lambda m: log(0, m))
    if world > 1:
        dist.barrier()
    if rank != 0:
        from cindy.data import _load_partial
        manifest = _load_partial()
    if manifest is None or not manifest.get("train_fake"):
        raise RuntimeError("manifest missing after prepare")

    if rank == 0:
        model = build_model(
            rank0_log=lambda m: log(0, m),
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            grad_checkpoint=not args.no_grad_checkpoint,
        )
    if world > 1:
        dist.barrier()
    if rank != 0:
        model = build_model(
            rank0_log=lambda m: None,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            grad_checkpoint=not args.no_grad_checkpoint,
        )
    model = model.to(device)

    resume_path = None
    if rank == 0:
        resume_path = resolve_resume(args.hf_repo, log=lambda m: log(0, m))
    if world > 1:
        dist.barrier()
    if rank != 0:
        from cindy.checkpoint import find_local
        resume_path = find_local()

    lora_params, head_params = trainable_params(model)
    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr_lora},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    start, best_auc = 0, 0.0
    if resume_path is not None:
        start, best_auc = load_checkpoint(model, opt, scaler, resume_path, device)
        log(rank, f"loaded step {start} best_auc {best_auc:.4f} from {resume_path}")
    if start >= args.max_steps:
        args.max_steps = start + args.max_steps
        log(rank, f"checkpoint already reached the old target. extending max_steps to {args.max_steps}")

    if world > 1:
        model = DDP(model, device_ids=[local], output_device=local, find_unused_parameters=False)

    train_items = make_balanced(manifest)
    val_items = make_val(manifest)
    train_ds = ListDataset(train_items, train=True)
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True) if world > 1 else None
    if len(train_ds) < args.batch_size * world:
        raise RuntimeError(
            f"only {len(train_ds)} training rows for batch {args.batch_size} x {world} gpus"
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = None
    if rank == 0:
        val_loader = DataLoader(ListDataset(val_items, train=False), batch_size=args.batch_size, shuffle=False, num_workers=0)

    uploader = Uploader(args.hf_repo, hf_token(), log=lambda m: log(0, m)) if rank == 0 else None
    state = {"step": start, "best_auc": best_auc}

    def dump(reason: str):
        if rank != 0:
            return
        path = ckpt_dir() / "latest.pt"
        save_checkpoint(model, opt, scaler, state["step"], state["best_auc"], args, path)
        log(0, f"saved {path} at step {state['step']} ({reason})")
        if uploader is not None:
            uploader.submit(path)
            uploader.flush(180)

    def on_signal(signum, _frame):
        log(rank, f"signal {signum}. saving before exit.")
        dump("signal")
        raise SystemExit(0)

    if rank == 0:
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)

    step = start
    epoch = 0
    model.train()
    log(rank, f"training steps {step} -> {args.max_steps} on {world} gpu(s), batch {args.batch_size}/gpu")
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        t0 = time.time()
        for images, target in train_loader:
            if step >= args.max_steps:
                break
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            smooth = target * 0.96 + 0.02
            set_lr(opt, step, args)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logit = model(images)
                loss = F.binary_cross_entropy_with_logits(logit, smooth)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_((lora_params + head_params), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            state["step"] = step
            if step % args.log_every == 0:
                dt = (time.time() - t0) / args.log_every
                log(rank, f"step {step} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e} {dt:.2f}s/step")
                t0 = time.time()
            if rank == 0 and step % args.save_every == 0:
                path = ckpt_dir() / "latest.pt"
                save_checkpoint(model, opt, scaler, step, state["best_auc"], args, path)
                log(0, f"checkpoint step {step}")
                if uploader is not None and step % args.hf_every == 0:
                    uploader.submit(path)
            if step % args.val_every == 0 and val_loader is not None and len(val_items) > 0:
                raw = model.module if isinstance(model, DDP) else model
                metrics = evaluate(raw, val_loader, device)
                log(0, f"val step {step} acc {metrics['acc']:.4f} auc {metrics['auc']:.4f} n {metrics['n']}")
                if metrics["auc"] == metrics["auc"] and metrics["auc"] > state["best_auc"]:
                    state["best_auc"] = metrics["auc"]
                    best = ckpt_dir() / "best.pt"
                    save_checkpoint(model, opt, scaler, step, state["best_auc"], args, best)
                    if uploader is not None:
                        uploader.submit(ckpt_dir() / "latest.pt", also=[(best, "best.pt")])
            if world > 1 and step % args.val_every == 0:
                dist.barrier()
        epoch += 1

    dump("finished")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
