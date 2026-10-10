"""Local checkpoints, and Hugging Face only when Kaggle has none."""

from __future__ import annotations

import os
import threading
from pathlib import Path


def work_root() -> Path:
    kaggle = Path("/kaggle/working")
    if kaggle.is_dir():
        return kaggle
    return Path(".").resolve()


def ckpt_dir() -> Path:
    path = work_root() / "cindy_ckpts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    path = work_root() / "cindy_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def find_local() -> Path | None:
    folder = ckpt_dir()
    latest = folder / "latest.pt"
    if latest.is_file() and latest.stat().st_size > 0:
        return latest
    steps = sorted(folder.glob("step_*.pt"))
    return steps[-1] if steps else None


def atomic_torch_save(obj, path: Path):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


class Uploader:
    """Uploads the newest snapshot. A slow push never blocks the next step."""

    def __init__(self, repo: str, token: str | None, log=print):
        self.repo = repo
        self.token = token
        self.log = log
        self._lock = threading.Lock()
        self._pending: Path | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._error: str | None = None

    def submit(self, path: Path, also: list[tuple[Path, str]] | None = None):
        if not self.token:
            return
        snap = path.parent / ".upload_latest.pt"
        snap.write_bytes(path.read_bytes())
        extras = []
        for src, name in also or []:
            dest = path.parent / f".upload_{name}"
            dest.write_bytes(src.read_bytes())
            extras.append((dest, name))
        with self._lock:
            old = self._pending
            self._pending = snap
            self._extras = extras
            if not self._running:
                self._running = True
                self._thread = threading.Thread(target=self._drain, daemon=True)
                self._thread.start()
        if old is not None and old != snap:
            old.unlink(missing_ok=True)

    def _commit(self, api, files: list[tuple[Path, str]]):
        import time
        from huggingface_hub import CommitOperationAdd

        ops = [
            CommitOperationAdd(path_in_repo=f"checkpoints/{name}", path_or_fileobj=str(local))
            for local, name in files
            if local.is_file() and local.stat().st_size > 0
        ]
        if not ops:
            return
        delay = 8
        last = None
        for _attempt in range(6):
            try:
                api.create_commit(
                    repo_id=self.repo,
                    repo_type="model",
                    operations=ops,
                    commit_message="cindy checkpoint",
                    token=self.token,
                )
                return
            except Exception as exc:  # noqa: BLE001
                last = exc
                text = str(exc).lower()
                if "429" in text or "rate" in text or "503" in text or "too many" in text:
                    self.log(f"[hf] rate limit, sleeping {delay}s")
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                    continue
                raise
        raise RuntimeError(f"hf still rate-limited: {last}")

    def _drain(self):
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            api.create_repo(self.repo, repo_type="model", exist_ok=True, token=self.token)
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            self.log(f"[hf] create_repo failed: {exc}")
        while True:
            with self._lock:
                snap = self._pending
                extras = list(getattr(self, "_extras", []))
                self._pending = None
                self._extras = []
                if snap is None:
                    self._running = False
                    return
            files = [(snap, "latest.pt")]
            meta = snap.parent / "latest.json"
            if meta.is_file():
                files.append((meta, "latest.json"))
            best = snap.parent / "best.pt"
            best_snap = snap.parent / ".upload_best.pt"
            if best.is_file() and best.stat().st_size > 0:
                best_snap.write_bytes(best.read_bytes())
                files.append((best_snap, "best.pt"))
            else:
                best_snap = None
            files.extend(extras)
            try:
                self._commit(api, files)
                self.log(f"[hf] pushed {len(files)} file(s) in one commit -> {self.repo}")
                self._error = None
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                self.log(f"[hf] upload failed (training continues): {exc}")
            finally:
                snap.unlink(missing_ok=True)
                if best_snap is not None:
                    best_snap.unlink(missing_ok=True)
                for local, _name in extras:
                    local.unlink(missing_ok=True)

    def flush(self, timeout: float = 180):
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)


def download_hf(repo: str, log=print) -> Path | None:
    token = hf_token()
    if not token:
        raise RuntimeError(
            "No local checkpoint, and HF_TOKEN is unset. "
            "Add a write token for Yashhh999/cindy as the Kaggle secret HF_TOKEN."
        )
    from huggingface_hub import hf_hub_download

    try:
        remote = hf_hub_download(
            repo_id=repo,
            filename="checkpoints/latest.pt",
            repo_type="model",
            token=token,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"[hf] no remote checkpoint yet ({exc}). starting fresh.")
        return None
    dest = ckpt_dir() / "latest.pt"
    dest.write_bytes(Path(remote).read_bytes())
    log(f"[hf] downloaded {repo}/checkpoints/latest.pt")
    return dest


def resolve_resume(repo: str, log=print) -> Path | None:
    """Local checkpoint wins. Hugging Face is used only when nothing is saved."""
    local = find_local()
    if local is not None:
        log(f"resume from local checkpoint {local} (not pulling Hugging Face)")
        return local
    log("no checkpoint in the Kaggle working dir. pulling the latest from Hugging Face.")
    return download_hf(repo, log=log)
