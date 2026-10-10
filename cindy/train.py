"""OpenFake era trainer for 2x T4.

Round 1 starts a new LoRA and head. Later rounds load latest.pt.
The frozen CLIP base never changes. A killed Kaggle session resumes from
progress.json (local first, otherwise Hugging Face).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms

from cindy.checkpoint import (
    Uploader,
    atomic_torch_save,
    ckpt_dir,
    delete_remote_weights,
    download_hf,
    download_progress,
    find_local,
    fresh_progress,
    hf_token,
    load_progress,
    save_progress,
)
from cindy.data import (
    delete_era_files,
    era_names,
    era_ready,
    fill_era,
    list_split,
    wipe_old_caches,
)
from cindy.model import ARCH, build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-repo", default="Yashhh999/cindy")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--hf-every", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--steps-per-era", type=int, default=0)
    p.add_argument("--per-gen", type=int, default=0)
    p.add_argument("--chunk", type=int, default=20000)
    p.add_argument("--replay-per", type=int, default=300)
    p.add_argument("--one-pass", action="store_true", default=True)
    p.add_argument("--min-free-gb", type=float, default=3.0)
    p.add_argument("--lr-lora", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--val-every", type=int, default=400)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--no-grad-checkpoint", action="store_true")
    args, _unknown = p.parse_known_args()
    return args


def init_dist():
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    if "RANK" not in os.environ:
        return 0, 1, 0
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=12))
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


def make_balanced(items):
    reals = [it for it in items if it[1] == 0]
    fakes = [it for it in items if it[1] == 1]
    if not reals or not fakes:
        return items
    n = 2 * max(len(reals), len(fakes))
    out = []
    for i in range(n // 2):
        out.append(reals[i % len(reals)])
        out.append(fakes[i % len(fakes)])
    return out


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
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    sum_pos = ranks[labels == 1].sum()
    return float((sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def evaluate(model, loader, device):
    was = model.training
    model.eval()
    scores, labels = [], []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logit = model(images)
        scores.append(torch.sigmoid(logit.float()).cpu())
        labels.append(target)
    model.train(was)
    y = torch.cat(labels).numpy()
    s = torch.cat(scores).numpy()
    pred = (s >= 0.5).astype(np.float64)
    acc = float((pred == y).mean()) if len(y) else float("nan")
    return {"acc": acc, "auc": binary_auc(y, s), "n": int(len(y))}


def set_lr(opt, step, args, total):
    if step < args.warmup:
        scale = step / max(1, args.warmup)
    else:
        span = max(1, total - args.warmup)
        progress = min(1.0, (step - args.warmup) / span)
        scale = 0.5 * (1 + math.cos(math.pi * progress))
    for group, base in zip(opt.param_groups, (args.lr_lora, args.lr_head)):
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


def save_checkpoint(model, opt, scaler, progress, args, path: Path):
    raw = model.module if isinstance(model, DDP) else model
    payload = {
        "arch": ARCH,
        "step": int(progress["global_step"]),
        "step_in_era": int(progress["step_in_era"]),
        "best_auc": float(progress["best_auc"]),
        "era": progress.get("era"),
        "era_index": int(progress["era_index"]),
        "model": raw.trainable_state(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
    }
    atomic_torch_save(payload, path)
    meta = {
        "arch": ARCH,
        "step": int(progress["global_step"]),
        "era": progress.get("era"),
        "era_index": int(progress["era_index"]),
        "best_auc": float(progress["best_auc"]),
        "time": time.time(),
    }
    (path.parent / "latest.json").write_text(json.dumps(meta))
    save_progress(progress)


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
    return ckpt


def adopt_one_pass(progress, log):
    if progress.get("one_pass"):
        return progress
    log("switching to a full core/train pass. scan restarts at row 0. weights kept.")
    for name in list(era_names()) + ["all"]:
        delete_era_files(name, log=log)
    progress["one_pass"] = True
    progress["era"] = "all"
    progress["era_index"] = 0
    progress["scanned"] = 0
    progress["stream_done"] = False
    progress["done"] = False
    progress["gen_counts"] = {}
    progress["real_count"] = 0
    progress["step_in_era"] = 0
    progress["chunk"] = 0
    return progress


def startup(args, log):
    local_progress = load_progress()
    if local_progress and local_progress.get("arch") == ARCH:
        log(f"resume local era={local_progress.get('era')} step={local_progress.get('global_step')}")
        return adopt_one_pass(local_progress, log), find_local()
    remote = download_progress(args.hf_repo, log=log)
    if remote and remote.get("arch") == ARCH:
        path = download_hf(args.hf_repo, log=log)
        log(f"resume huggingface era={remote.get('era')} step={remote.get('global_step')}")
        return adopt_one_pass(remote, log), path
    token = hf_token()
    if token:
        delete_remote_weights(args.hf_repo, token, log=log)
    wipe_old_caches(log=log)
    old = ckpt_dir() / "latest.pt"
    if old.is_file():
        old.unlink()
        log("removed local v1 checkpoint")
    progress = fresh_progress(ARCH)
    progress["one_pass"] = True
    progress["era"] = "all"
    save_progress(progress)
    log("fresh one-pass over core/train")
    return progress, None


def advance_era(progress, log):
    finished = progress.get("era")
    progress.setdefault("eras_finished", []).append({
        "era": finished,
        "fakes": sum((progress.get("gen_counts") or {}).values()),
        "reals": int(progress.get("real_count") or 0),
        "step": int(progress.get("global_step") or 0),
    })
    delete_era_files(finished, log=log)
    progress["era_index"] = int(progress["era_index"]) + 1
    progress["scanned"] = 0
    progress["gen_counts"] = {}
    progress["real_count"] = 0
    progress["stream_done"] = False
    progress["step_in_era"] = 0
    if progress["era_index"] >= len(era_names()):
        progress["done"] = True
        progress["era"] = "done"
    else:
        progress["era"] = era_names()[progress["era_index"]]
    log(f"finished {finished}. next={progress['era']} global_step={progress['global_step']}")
    return progress


def wait_for_rank0(rank, world):
    """Rank 1 sleeps until rank 0 publishes. No NCCL call, so a long download cannot time out."""
    if world <= 1:
        return
    ready = ckpt_dir() / "rank1.ready"
    go = ckpt_dir() / "rank0.go"
    if rank != 0:
        ready.write_text("1")
        while not go.is_file():
            time.sleep(2)
        go.unlink(missing_ok=True)
        ready.unlink(missing_ok=True)
        return
    while not ready.is_file():
        time.sleep(2)
    go.write_text("1")
    while ready.is_file():
        time.sleep(1)


def main():
    args = parse_args()
    rank, world, local = init_dist()
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    random.seed(42 + rank)
    np.random.seed(42 + rank)
    torch.manual_seed(42 + rank)

    progress, resume_path = None, None
    if rank == 0:
        if not hf_token():
            raise SystemExit("Set HF_TOKEN (write access to Yashhh999/cindy).")
        progress, resume_path = startup(args, log=lambda m: log(0, m))
        save_progress(progress)
    if world > 1:
        dist.barrier()
    if rank != 0:
        progress = load_progress()
    if progress is None:
        raise RuntimeError("progress.json missing")

    model = build_model(
        rank0_log=lambda m: log(rank, m),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        grad_checkpoint=not args.no_grad_checkpoint,
    ).to(device)
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
    if resume_path is not None and rank == 0:
        load_checkpoint(model, opt, scaler, resume_path, device)
        log(0, f"loaded weights step {progress['global_step']} era {progress['era']}")
    if world > 1:
        dist.barrier()
        if rank != 0 and find_local() is not None:
            load_checkpoint(model, opt, scaler, find_local(), device)
        model = DDP(model, device_ids=[local], output_device=local, find_unused_parameters=False)

    uploader = Uploader(args.hf_repo, hf_token(), log=lambda m: log(0, m)) if rank == 0 else None

    def dump(reason: str):
        if rank != 0:
            return
        path = ckpt_dir() / "latest.pt"
        save_checkpoint(model, opt, scaler, progress, args, path)
        log(0, f"saved step {progress['global_step']} era {progress['era']} ({reason})")
        if uploader is not None:
            uploader.submit(path)
            uploader.flush(180)

    def on_signal(signum, _frame):
        log(rank, f"signal {signum}. saving.")
        dump("signal")
        raise SystemExit(0)

    if rank == 0:
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)

    while not progress.get("done"):
        if rank == 0:
            (ckpt_dir() / "rank0.go").unlink(missing_ok=True)
            if not progress.get("stream_done"):
                progress = fill_era(progress, args, log=lambda m: log(0, m))
                save_progress(progress)
                fakes = sum((progress.get("gen_counts") or {}).values())
                log(0, f"era {progress['era']} fakes={fakes} reals={progress['real_count']} scanned={progress['scanned']}")
        wait_for_rank0(rank, world)
        progress = load_progress()

        if progress.get("stream_done") and not era_ready(progress):
            if rank == 0:
                log(0, f"era {progress['era']} has nothing to train. advancing.")
                progress = advance_era(progress, log=lambda m: log(0, m))
                save_progress(progress)
            wait_for_rank0(rank, world)
            progress = load_progress()
            continue

        train_items, val_items = list_split(progress["era"])
        train_items = make_balanced(train_items)
        if len(train_items) < args.batch_size * world:
            if rank == 0:
                log(0, "not enough images yet")
                if progress.get("stream_done"):
                    progress = advance_era(progress, log=lambda m: log(0, m))
                    save_progress(progress)
            wait_for_rank0(rank, world)
            progress = load_progress()
            continue

        train_ds = ListDataset(train_items, train=True)
        sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True) if world > 1 else None
        loader = DataLoader(
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
        if rank == 0 and val_items:
            val_loader = DataLoader(ListDataset(val_items, train=False), batch_size=args.batch_size, shuffle=False, num_workers=0)

        span = args.steps_per_era if args.steps_per_era > 0 else max(200, len(train_items) // max(1, args.batch_size * world))
        target = int(progress["step_in_era"]) + span
        log(rank, f"train chunk {progress.get('chunk', 0)} steps {progress['step_in_era']} -> {target}")
        model.train()
        epoch = 0
        done_steps = False
        while int(progress["step_in_era"]) < target:
            if sampler is not None:
                sampler.set_epoch(epoch)
            t0 = time.time()
            for images, target_y in loader:
                if int(progress["step_in_era"]) >= target:
                    done_steps = True
                    break
                images = images.to(device, non_blocking=True)
                target_y = target_y.to(device, non_blocking=True)
                smooth = target_y * 0.96 + 0.02
                set_lr(opt, int(progress["step_in_era"]), args, target)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    main, aux = model(images)
                    loss = F.binary_cross_entropy_with_logits(main, smooth)
                    loss = loss + 0.5 * F.binary_cross_entropy_with_logits(aux, smooth)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
                scaler.step(opt)
                scaler.update()
                progress["step_in_era"] = int(progress["step_in_era"]) + 1
                progress["global_step"] = int(progress["global_step"]) + 1
                step = int(progress["global_step"])
                if step % args.log_every == 0:
                    dt = (time.time() - t0) / args.log_every
                    log(rank, f"step {step} era {progress['era']} loss {loss.item():.4f} {dt:.2f}s/step")
                    t0 = time.time()
                if rank == 0 and step % args.save_every == 0:
                    path = ckpt_dir() / "latest.pt"
                    save_checkpoint(model, opt, scaler, progress, args, path)
                    if uploader is not None and step % args.hf_every == 0:
                        uploader.submit(path)
                if step % args.val_every == 0 and val_loader is not None and len(val_items) > 1:
                    raw = model.module if isinstance(model, DDP) else model
                    metrics = evaluate(raw, val_loader, device)
                    log(0, f"check step {step} era {progress['era']} acc {metrics['acc']:.3f} auc {metrics['auc']:.3f} n {metrics['n']}")
                    if metrics["auc"] == metrics["auc"] and metrics["auc"] > float(progress["best_auc"]):
                        progress["best_auc"] = metrics["auc"]
                        save_checkpoint(model, opt, scaler, progress, args, ckpt_dir() / "best.pt")
                        save_checkpoint(model, opt, scaler, progress, args, ckpt_dir() / "latest.pt")
                if world > 1 and step % args.val_every == 0:
                    dist.barrier()
            epoch += 1
            if done_steps:
                break
            if epoch > 100000:
                break

        if rank == 0:
            dump("era-chunk")
            if progress.get("stream_done"):
                progress["done"] = True
                log(0, f"core/train finished at step {progress['global_step']}")
            else:
                delete_era_files(progress["era"], log=lambda m: log(0, m))
                progress["step_in_era"] = 0
                progress["chunk"] = int(progress.get("chunk") or 0) + 1
                log(0, f"freed chunk. replay kept. next chunk {progress['chunk']} at row {progress['scanned']}")
            save_progress(progress)
        if world > 1:
            dist.barrier()
            progress = load_progress()

    if rank == 0:
        dump("finished")
        log(0, "all eras finished " + json.dumps(progress.get("eras_finished")))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
