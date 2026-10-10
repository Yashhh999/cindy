"""DRAGON (2025, 25 diffusion models) + optional AI-image v2 folder.

Reals come from COCO val, Flickr30k, an optional --real-root, and v2 reals.
Lumina is held out of training so there is one unseen generator.
"""

from __future__ import annotations

import csv
import io
import json
import random
import re
import zipfile
from pathlib import Path

from PIL import Image

from cindy.checkpoint import data_dir

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REAL_PARTS = {"real", "reals", "human", "authentic", "nature"}
FAKE_HINTS = (
    "fake", "synth", "generated", "sdxl", "flux", "kandinsky",
    "pixart", "cascade", "sd15", "stable", "gan", "midjourney",
)


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def is_holdout(name: str, holdouts: list[str]) -> bool:
    n = normalize(name)
    if not n:
        return False
    return any(h and (h in n or n in h) for h in holdouts)


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return (cleaned or "unknown")[:48]


def _open_image(value):
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, str) and Path(value).is_file():
        return Image.open(value).convert("RGB")
    return None


def example_image(ex: dict):
    for key in ("png", "jpg", "jpeg", "image", "webp"):
        if key in ex:
            image = _open_image(ex[key])
            if image is not None:
                return image
    return None


def example_model(ex: dict) -> str:
    text = ex.get("model.txt")
    if isinstance(text, str) and text.strip():
        return text.strip()
    blob = ex.get("json")
    if isinstance(blob, str):
        try:
            blob = json.loads(blob)
        except json.JSONDecodeError:
            blob = None
    if isinstance(blob, dict) and blob.get("model"):
        return str(blob["model"]).strip()
    if ex.get("model"):
        return str(ex["model"]).strip()
    return "unknown"


def example_key(ex: dict, model: str) -> str:
    key = ex.get("__key__")
    if key:
        return str(key)
    return f"{model}:{ex.get('json')}"


def save_jpeg(image: Image.Image, path: Path, quality=95):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    image.thumbnail((512, 512), Image.BICUBIC)
    tmp = path.with_suffix(".tmp")
    image.save(tmp, format="JPEG", quality=quality)
    tmp.replace(path)


def open_dragon(split: str):
    from datasets import get_dataset_config_names, load_dataset

    repo = "lesc-unifi/dragon"
    configs = [None]
    try:
        configs = [None] + list(get_dataset_config_names(repo))
    except Exception:
        pass
    errors = []
    seen = set()
    for cfg in configs:
        if cfg in seen:
            continue
        seen.add(cfg)
        try:
            if cfg is None:
                return load_dataset(repo, split=split, streaming=True)
            return load_dataset(repo, cfg, split=split, streaming=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{cfg}: {exc}")
    raise RuntimeError("Could not stream lesc-unifi/dragon\n" + "\n".join(errors))


def _download(url: str, dest: Path, log=print):
    import urllib.request

    if dest.is_file() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)


def ensure_coco(log=print) -> list[str]:
    root = data_dir() / "coco_val2017"
    if root.is_dir() and sum(1 for _ in root.glob("*.jpg")) > 1000:
        return [str(p) for p in root.glob("*.jpg")]
    zip_path = data_dir() / "val2017.zip"
    _download("http://images.cocodataset.org/zips/val2017.zip", zip_path, log)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir())
    extracted = data_dir() / "val2017"
    if extracted.is_dir() and not root.exists():
        extracted.rename(root)
    source = root if root.is_dir() else extracted
    return [str(p) for p in source.glob("*.jpg")]


def ensure_flickr(limit: int, log=print) -> list[str]:
    folder = data_dir() / "flickr"
    existing = sorted(str(p) for p in folder.glob("*.jpg"))
    if len(existing) >= limit:
        return existing[:limit]
    try:
        from datasets import load_dataset
        stream = load_dataset("nlphuji/flickr30k", split="test", streaming=True)
    except Exception as exc:  # noqa: BLE001
        log(f"flickr30k unavailable ({exc}). continuing without it.")
        return existing
    folder.mkdir(parents=True, exist_ok=True)
    failures = 0
    for ex in stream:
        if len(existing) >= limit:
            break
        try:
            image = example_image(ex)
            if image is None:
                raise RuntimeError("no image")
            path = folder / f"{len(existing):06d}.jpg"
            save_jpeg(image, path)
            existing.append(str(path))
            failures = 0
        except Exception:
            failures += 1
            if failures > 30 and len(existing) == 0:
                log("flickr30k had no readable images. skipping.")
                break
    log(f"flickr reals kept: {len(existing)}")
    return existing


def _label_from_parts(path: Path):
    decision = None
    gen = "v2"
    for part in path.parts:
        low = part.lower()
        if low in REAL_PARTS or low.startswith("real"):
            decision = "real"
        if any(hint in low for hint in FAKE_HINTS):
            decision = "fake"
            gen = part
    return decision, gen


def scan_v2(root: Path, per_gen: int, log=print, max_real: int = 10**9, max_fake: int = 10**9):
    reals, fakes = [], []
    counts: dict[str, int] = {}
    if not root or not root.exists():
        return reals, fakes
    log(f"scanning extra images under {root}")
    seen_files = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMG_EXT:
            continue
        seen_files += 1
        if seen_files > 800_000:
            break
        kind, gen = _label_from_parts(path)
        if kind == "real":
            if len(reals) < max_real:
                reals.append(str(path))
        elif kind == "fake":
            if len(fakes) >= max_fake or counts.get(gen, 0) >= per_gen:
                continue
            counts[gen] = counts.get(gen, 0) + 1
            fakes.append((str(path), gen))
        if len(reals) >= max_real and len(fakes) >= max_fake:
            break
    for csv_path in list(root.rglob("*.csv"))[:20]:
        try:
            with csv_path.open(newline="", encoding="utf-8", errors="ignore") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                fields = {name.lower(): name for name in reader.fieldnames}
                path_key = next((fields[k] for k in ("path", "filepath", "file", "image", "filename") if k in fields), None)
                label_key = next((fields[k] for k in ("label", "class", "target") if k in fields), None)
                gen_key = next((fields[k] for k in ("generator", "model", "source") if k in fields), None)
                if path_key is None or label_key is None:
                    continue
                for row in reader:
                    if len(reals) >= max_real and len(fakes) >= max_fake:
                        break
                    raw = row[path_key]
                    image = Path(raw) if Path(raw).is_file() else csv_path.parent / raw
                    if not image.is_file():
                        continue
                    label = str(row[label_key]).strip().lower()
                    gen = safe_name(row.get(gen_key, "v2")) if gen_key else "v2"
                    if label in {"0", "real", "human", "nature"}:
                        if len(reals) < max_real:
                            reals.append(str(image))
                    elif label in {"1", "fake", "ai", "synthetic"}:
                        if len(fakes) >= max_fake or counts.get(gen, 0) >= per_gen:
                            continue
                        counts[gen] = counts.get(gen, 0) + 1
                        fakes.append((str(image), gen))
        except Exception as exc:  # noqa: BLE001
            log(f"skipping manifest {csv_path}: {exc}")
    log(f"v2/extra reals={len(reals)} fakes={len(fakes)} by {counts}")
    return reals, fakes


def _take_v2(reals, fakes, total: int):
    """Keep `total` June 2026 images, half real and half fake when both exist."""
    rng = random.Random(1)
    rng.shuffle(reals)
    rng.shuffle(fakes)
    half = total // 2
    picked_r = list(reals[:half])
    picked_f = list(fakes[:half])
    rest = [("r", item) for item in reals[len(picked_r):]] + [("f", item) for item in fakes[len(picked_f):]]
    rng.shuffle(rest)
    for kind, item in rest:
        if len(picked_r) + len(picked_f) >= total:
            break
        if kind == "r":
            picked_r.append(item)
        else:
            picked_f.append(item)
    return picked_r, picked_f


def _partial_path() -> Path:
    return data_dir() / "manifest.json"


def _load_partial():
    path = _partial_path()
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _write_partial(payload: dict):
    path = _partial_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def _fill_dragon(split, holdouts, quota_for, payload, seen, log, max_scan, allow_train, max_train, max_val):
    stream = open_dragon(split)
    scanned = 0
    folder = data_dir() / "dragon" / split
    for ex in stream:
        scanned += 1
        if scanned > max_scan:
            log(f"stopped {split} scan at {max_scan}")
            break
        if not allow_train and len(payload["val_fake"]) >= max_val:
            log(f"{split} val cache is full ({len(payload['val_fake'])})")
            break
        if allow_train and sum(payload["train_counts"].values()) >= max_train and len(payload["val_fake"]) >= 150:
            log(f"{split} hit the DRAGON train cap ({max_train})")
            break
        model = example_model(ex)
        key = example_key(ex, model)
        if key in seen:
            continue
        hold = is_holdout(model, holdouts)
        if payload["val_counts"].get(model, 0) < quota_for(model, "val"):
            kind = "val"
        elif allow_train and not hold and payload["train_counts"].get(model, 0) < quota_for(model, "train"):
            kind = "train"
        else:
            continue
        image = example_image(ex)
        if image is None:
            continue
        bucket = "val_counts" if kind == "val" else "train_counts"
        idx = payload[bucket].get(model, 0)
        path = folder / kind / safe_name(model) / f"{idx:05d}.jpg"
        save_jpeg(image, path)
        payload["val_fake" if kind == "val" else "train_fake"].append({"path": str(path), "gen": model})
        payload[bucket][model] = idx + 1
        seen.add(key)
        total = len(payload["train_fake"]) + len(payload["val_fake"])
        if total % 100 == 0:
            payload["seen"] = list(seen)
            _write_partial(payload)
            log(
                f"{split} scanned={scanned} train_fakes={len(payload['train_fake'])} "
                f"val_fakes={len(payload['val_fake'])}"
            )
    payload["seen"] = list(seen)
    _write_partial(payload)


def prepare(args, log=print) -> dict:
    holdouts = [normalize(part) for part in args.holdout.split(",") if part.strip()]
    existing = _load_partial()
    dragon_target = int(getattr(args, "dragon_train", 160000))
    v2_target = int(getattr(args, "v2_total", 70000))
    if existing and existing.get("complete") and not args.smoke:
        same = existing.get("dragon_target") == dragon_target and existing.get("v2_target") == v2_target
        if same:
            log(
                "using cached manifest "
                f"({len(existing['train_fake'])} train fakes, {len(existing['train_real'])} train reals)"
            )
            return existing
        log(
            "cached mix is the smaller run "
            f"({len(existing.get('train_fake') or [])} fakes). "
            f"rebuilding {dragon_target} DRAGON + {v2_target} June 2026 images."
        )
        existing["train_fake"] = []
        existing["train_counts"] = {}
        existing["seen"] = []
        existing["complete"] = False

    payload = existing or {
        "complete": False,
        "holdouts": holdouts,
        "train_fake": [],
        "val_fake": [],
        "train_real": [],
        "val_real": [],
        "train_counts": {},
        "val_counts": {},
        "seen": [],
    }
    payload["holdouts"] = holdouts
    seen = set(payload.get("seen") or [])

    def quota_for(model, kind):
        if is_holdout(model, holdouts):
            return args.holdout_val if kind == "val" else 0
        return args.val_per_model if kind == "val" else args.per_model

    max_train = dragon_target
    max_val = args.val_per_model * 30 + args.holdout_val
    payload["dragon_target"] = dragon_target
    payload["v2_target"] = v2_target
    log(f"target {dragon_target} DRAGON train fakes, at most {args.per_model} per generator, plus {v2_target} June 2026 images")
    log("streaming DRAGON test split into the val cache")
    try:
        _fill_dragon(
            "test", holdouts, quota_for, payload, seen, log, args.max_scan,
            allow_train=False, max_train=max_train, max_val=max_val,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"DRAGON test split failed ({exc}). val will be carved from train.")
    log("streaming DRAGON train split")
    _fill_dragon(
        "train", holdouts, quota_for, payload, seen, log, args.max_scan,
        allow_train=True, max_train=max_train, max_val=max_val,
    )

    v2_root = Path(args.v2_root) if args.v2_root else Path("/kaggle/input")
    if v2_root.is_dir():
        v2_reals, v2_fakes = scan_v2(
            v2_root, args.v2_per_gen, log, max_real=v2_target, max_fake=v2_target,
        )
        v2_reals, v2_fakes = _take_v2(v2_reals, v2_fakes, v2_target)
        log(f"June 2026 kept reals={len(v2_reals)} fakes={len(v2_fakes)} (target {v2_target})")
    else:
        v2_reals, v2_fakes = [], []
        log("no v2 folder. training without the June 2026 paired set.")

    extra_reals = []
    if args.real_root:
        real_root = Path(args.real_root)
        if real_root.is_dir():
            labeled, _ = scan_v2(real_root, per_gen=0, log=log)
            extra_reals = labeled if len(labeled) >= 100 else [
                str(p) for p in real_root.rglob("*") if p.suffix.lower() in IMG_EXT
            ][:50000]

    try:
        coco = [] if args.no_coco else ensure_coco(log)
    except Exception as exc:  # noqa: BLE001
        log(f"COCO download failed ({exc})")
        coco = []
    try:
        flickr = [] if args.no_flickr else ensure_flickr(args.real_count, log)
    except Exception as exc:  # noqa: BLE001
        log(f"flickr skipped ({exc})")
        flickr = []
    reals = list(dict.fromkeys(v2_reals + extra_reals + coco + flickr))
    rng = random.Random(0)
    rng.shuffle(reals)
    if len(reals) < 64:
        raise RuntimeError(f"only {len(reals)} real images. Check COCO download or pass --real-root.")

    n_val_real = min(1000, max(32, len(reals) // 10))
    payload["val_real"] = reals[:n_val_real]
    payload["train_real"] = reals[n_val_real:]
    have = {item["path"] for item in payload["train_fake"]}
    added_v2 = 0
    for path, gen in v2_fakes:
        if path in have:
            continue
        have.add(path)
        payload["train_fake"].append({"path": path, "gen": f"v2_{gen}"})
        added_v2 += 1
    log(f"added {added_v2} June 2026 fakes")

    if len(payload["train_fake"]) < 32:
        raise RuntimeError(
            f"only {len(payload['train_fake'])} training fakes. DRAGON streaming did not yield images."
        )
    payload["complete"] = True
    payload["seen"] = []
    _write_partial(payload)
    log(
        "data ready "
        f"train_fake={len(payload['train_fake'])} train_real={len(payload['train_real'])} "
        f"val_fake={len(payload['val_fake'])} val_real={len(payload['val_real'])}"
    )
    log("train generators: " + json.dumps(payload["train_counts"], sort_keys=True))
    log("val generators: " + json.dumps(payload["val_counts"], sort_keys=True))
    if not v2_fakes:
        log(
            "WARNING: AI-image v2 folder not found. DRAGON already includes Flux, SDXL, "
            "SD3, SD1.5, PixArt, Kandinsky and Cascade. Attach the IEEE v2 set under "
            "/kaggle/input for paired COCO reals."
        )
    return payload
