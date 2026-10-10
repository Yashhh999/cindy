"""Stream OpenFake core/train from oldest era to newest.

core/test and reddit are never read. Held-out generators are skipped even if
they appear in train. Images are 512 px JPEGs. Replay copies are kept when
the era folder is deleted to free disk.
"""

from __future__ import annotations

import io
import re
import shutil
from pathlib import Path

from PIL import Image

from cindy.checkpoint import data_dir, replay_dir

HOLDOUT = {
    "gptimage15",
    "gptimage20",
    "nanobananapro",
    "flux2klein9b",
    "zimageturbo",
    "recraftv2",
    "recraftv3",
    "midjourney7",
    "ideogram20",
}
# (name, inclusive start YYYYMM or None, exclusive end YYYYMM or None)
ERAS = (
    ("early", None, 202307),
    ("sdxl", 202307, 202407),
    ("flux", 202407, 202507),
    ("newest", 202507, None),
    ("undated", None, None),
)
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def era_names():
    return [name for name, _start, _end in ERAS]


def model_key(name: str) -> str:
    text = (name or "").lower().replace("·", "")
    return re.sub(r"[^a-z0-9]", "", text)


def is_holdout(name: str) -> bool:
    return model_key(name) in HOLDOUT


def parse_yyyymm(value) -> int | None:
    if value is None:
        return None
    digits = re.sub(r"[^0-9]", "", str(value))
    if len(digits) < 4:
        return None
    year = int(digits[:4])
    month = int(digits[4:6] or "1")
    if year < 2000 or month < 1 or month > 12:
        return None
    return year * 100 + month


def era_index_for(label: str, release) -> int | None:
    if str(label).lower() in {"real", "0", "human"}:
        return None
    stamp = parse_yyyymm(release)
    if stamp is None:
        return len(ERAS) - 1
    for index, (_name, start, end) in enumerate(ERAS[:-1]):
        if (start is None or stamp >= start) and (end is None or stamp < end):
            return index
    return len(ERAS) - 1


def free_gb() -> float:
    root = Path("/kaggle/working")
    path = root if root.is_dir() else data_dir()
    return shutil.disk_usage(path).free / 1e9


def wipe_old_caches(log=print):
    root = data_dir()
    for name in ("dragon", "coco_val2017", "flickr", "val2017"):
        path = root / name
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            log(f"deleted old cache {path}")
    manifest = root / "manifest.json"
    if manifest.is_file():
        manifest.unlink()


def _open_train_stream():
    from datasets import load_dataset

    errors = []
    attempts = (
        {"path": "ComplexDataLab/OpenFake", "name": "core", "split": "train"},
        {"path": "ComplexDataLab/OpenFake", "split": "train"},
    )
    for kwargs in attempts:
        try:
            return load_dataset(**kwargs, streaming=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{kwargs}: {exc}")
    raise RuntimeError("Could not stream OpenFake core/train\n" + "\n".join(errors))


def _label_real(value) -> bool | None:
    text = str(value).strip().lower()
    if text in {"real", "0", "human", "nature"}:
        return True
    if text in {"fake", "1", "ai", "synthetic"}:
        return False
    return None


def _save_jpeg(image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    image.thumbnail((512, 512), Image.BICUBIC)
    tmp = path.with_suffix(".tmp")
    image.save(tmp, format="JPEG", quality=90)
    tmp.replace(path)


def _as_image(value):
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value))
    return None


def era_image_dir(era_name: str) -> Path:
    path = data_dir() / "era" / era_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _next_index(folder: Path) -> int:
    nums = []
    if folder.is_dir():
        for path in folder.glob("*.jpg"):
            if path.stem.isdigit():
                nums.append(int(path.stem))
    return (max(nums) + 1) if nums else 0


def _copy_replay(src: Path, key: str, cap: int):
    folder = replay_dir() / key
    folder.mkdir(parents=True, exist_ok=True)
    have = len(list(folder.glob("*.jpg")))
    if have >= cap:
        return
    dest = folder / f"{have:05d}.jpg"
    if not dest.exists():
        dest.write_bytes(src.read_bytes())


def fill_era(progress: dict, args, log=print) -> dict:
    """Save this era until the disk is low, the caps are met, or the stream ends."""
    names = era_names()
    index = int(progress["era_index"])
    if index >= len(names):
        progress["done"] = True
        return progress
    era = names[index]
    progress["era"] = era
    scanned = int(progress.get("scanned") or 0)
    counts = dict(progress.get("gen_counts") or {})
    reals = int(progress.get("real_count") or 0)
    log(f"fill era {era} from row {scanned}  gens={len(counts)} reals={reals}")
    stream = _open_train_stream()
    folder = era_image_dir(era)
    kept_since_log = 0
    seen_rows = 0
    for row_i, row in enumerate(stream):
        if row_i < scanned:
            if row_i > 0 and row_i % 50000 == 0:
                log(f"catch-up {row_i}/{scanned}")
            continue
        seen_rows += 1
        progress["scanned"] = row_i + 1
        if free_gb() < args.min_free_gb:
            log(f"disk low ({free_gb():.1f} GB free). stop fill.")
            break
        real = _label_real(row.get("label"))
        model = str(row.get("model") or "unknown")
        if real is False and is_holdout(model):
            continue
        if real is False:
            if era_index_for("fake", row.get("release_date")) != index:
                continue
            key = model_key(model) or "unknown"
            if counts.get(key, 0) >= args.per_gen:
                continue
            image = _as_image(row.get("image"))
            if image is None:
                continue
            dest_dir = folder / "fake" / key[:48]
            dest = dest_dir / f"{_next_index(dest_dir):05d}.jpg"
            _save_jpeg(image, dest)
            _copy_replay(dest, key[:48], args.replay_per)
            counts[key] = counts.get(key, 0) + 1
            kept_since_log += 1
        elif real is True and reals < sum(counts.values()):
            image = _as_image(row.get("image"))
            if image is None:
                continue
            dest_dir = folder / "real"
            dest = dest_dir / f"{_next_index(dest_dir):05d}.jpg"
            _save_jpeg(image, dest)
            _copy_replay(dest, "real", args.replay_per * 4)
            reals += 1
            kept_since_log += 1
        if kept_since_log >= 200:
            progress["gen_counts"] = counts
            progress["real_count"] = reals
            log(f"era {era} scanned={progress['scanned']} fakes={sum(counts.values())} reals={reals} free={free_gb():.1f}GB")
            kept_since_log = 0
        if seen_rows >= args.scan_chunk and sum(counts.values()) > 0 and reals > 0:
            log(f"scan chunk {args.scan_chunk} reached")
            break
    else:
        progress["stream_done"] = True
        log(f"era {era} stream finished")
    progress["gen_counts"] = counts
    progress["real_count"] = reals
    progress["era"] = era
    return progress


def list_split(era_name: str):
    """Train files are this era plus replay. 5% held out by path hash, both classes."""
    files = []
    era = era_image_dir(era_name)
    replay = replay_dir()
    for root, label in ((era / "fake", 1), (era / "real", 0), (replay, None)):
        if not root.is_dir():
            continue
        for path in root.rglob("*.jpg"):
            if label is None:
                y = 0 if path.parent.name == "real" else 1
            else:
                y = label
            files.append((str(path), y))
    train, val = [], []
    for path, y in files:
        if (sum(ord(ch) for ch in path) % 20) == 0:
            val.append((path, y))
        else:
            train.append((path, y))
    return train, val


def era_ready(progress: dict) -> bool:
    fakes = sum((progress.get("gen_counts") or {}).values())
    reals = int(progress.get("real_count") or 0)
    return fakes >= 64 and reals >= 64


def era_finished(progress: dict, per_gen: int) -> bool:
    if not progress.get("stream_done"):
        return False
    counts = progress.get("gen_counts") or {}
    if not counts:
        return True
    return all(n >= per_gen for n in counts.values())


def delete_era_files(era_name: str, log=print):
    path = data_dir() / "era" / era_name
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        log(f"deleted era images {path} (replay kept)")
