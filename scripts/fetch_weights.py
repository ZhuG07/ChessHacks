#!/usr/bin/env python3
"""
Make sure the model checkpoints are real files, not Git LFS pointers.

The .pt files are stored with Git LFS. A plain `git clone` without LFS (and
many hosting platforms) leaves small text pointers in their place. This
script finds those pointers, downloads the real files from GitHub's LFS
media endpoint, and checks them against the SHA-256 in the pointer.

    python scripts/fetch_weights.py                 # default checkpoint only
    python scripts/fetch_weights.py --all           # every checkpoint
    python scripts/fetch_weights.py best.pt last.pt # specific files
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "src" / "model_save"
REPO = os.getenv("WEIGHTS_REPO", "ZhuG07/ChessHacks")
REF = os.getenv("WEIGHTS_REF", "main")

sys.path.insert(0, str(ROOT))


def read_pointer(path: Path) -> dict | None:
    """Return {'oid': ..., 'size': ...} if path is an LFS pointer, else None."""
    if not path.exists() or path.stat().st_size > 1024:
        return None
    text = path.read_text(errors="ignore")
    if not text.startswith("version https://git-lfs"):
        return None
    fields = dict(line.split(" ", 1) for line in text.strip().splitlines() if " " in line)
    return {"oid": fields["oid"].split(":", 1)[1], "size": int(fields["size"])}


def download(name: str, pointer: dict) -> None:
    url = f"https://media.githubusercontent.com/media/{REPO}/{REF}/src/model_save/{name}"
    print(f"Downloading {name} ({pointer['size'] / 1e6:.1f} MB) from {url}")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as resp, tempfile.NamedTemporaryFile(dir=MODEL_DIR, delete=False) as tmp:
        while chunk := resp.read(1 << 20):
            digest.update(chunk)
            tmp.write(chunk)
    tmp_path = Path(tmp.name)
    if digest.hexdigest() != pointer["oid"]:
        tmp_path.unlink()
        raise RuntimeError(f"Checksum mismatch for {name}; refusing to use it.")
    tmp_path.chmod(0o644)
    tmp_path.replace(MODEL_DIR / name)
    print(f"  ok, sha256 verified")


def main():
    from src.engine import resolve_checkpoint

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="checkpoint file names inside src/model_save")
    parser.add_argument("--all", action="store_true", help="fetch every checkpoint")
    args = parser.parse_args()

    if args.all:
        names = sorted(p.name for p in MODEL_DIR.glob("*.pt"))
    elif args.names:
        names = args.names
    else:
        names = [resolve_checkpoint().name]

    for name in names:
        pointer = read_pointer(MODEL_DIR / name)
        if pointer is None:
            print(f"{name}: already present")
            continue
        download(name, pointer)


if __name__ == "__main__":
    main()
