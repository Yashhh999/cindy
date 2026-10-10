"""Push already-downloaded checkpoints in a single Hugging Face commit."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    repo = sys.argv[2] if len(sys.argv) > 2 else "Yashhh999/cindy"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN")
    names = ("latest.pt", "best.pt", "latest.json", "arch.json")
    ops = []
    for name in names:
        path = folder / name
        if path.is_file() and path.stat().st_size > 0:
            ops.append(CommitOperationAdd(path_in_repo=f"checkpoints/{name}", path_or_fileobj=str(path)))
            print(f"including {path} ({path.stat().st_size} bytes)")
    if not any(op.path_in_repo.endswith("latest.pt") for op in ops):
        raise SystemExit(f"no latest.pt in {folder}")
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", exist_ok=True, token=token)
    api.create_commit(
        repo_id=repo,
        repo_type="model",
        operations=ops,
        commit_message="upload local cindy checkpoints",
        token=token,
    )
    print(f"pushed {len(ops)} file(s) to {repo} in one commit")


if __name__ == "__main__":
    main()
