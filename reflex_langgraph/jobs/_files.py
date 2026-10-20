"""
Files of a job: atomic JSON writes, locked updates, and content fingerprints.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

_CHUNK = 1 << 20


def write_json(path: Path, data: dict) -> None:
    """Replace ``path`` atomically. Each write has its own temporary file, so two writers
    never collide on it."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(data, indent=2, sort_keys=True, default=str))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def update_json(path: Path, fields: dict) -> dict:
    """Read, merge ``fields`` and write back, under an exclusive lock on ``<path>.lock``.

    The web server and the detached worker both update a job's state; without the lock one
    can overwrite the other's fields (e.g. drop a cancel request). Readers need no lock:
    :func:`write_json` replaces the file atomically.
    """
    with open(path.with_name(path.name + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = read_json(path)
        data.update(fields)
        write_json(path, data)
        return data


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(path: Path) -> dict:
    """Content fingerprint of a file or a directory tree (symlinks followed)."""
    path = Path(path)
    if path.is_file():
        return {"kind": "file", "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if not path.is_dir():
        return {"kind": "missing"}
    files: List[Dict[str, object]] = []
    digest = hashlib.sha256()
    for p in sorted(q for q in path.rglob("*") if q.is_file() and "__pycache__" not in q.parts):
        rel = str(p.relative_to(path))
        sha = sha256_file(p)
        files.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha})
        digest.update(f"{rel}\0{sha}\n".encode())
    return {"kind": "dir", "digest": digest.hexdigest(), "files": files}
