"""Shared utilities for the XAI-Dyslexia pipeline.

Contents:
    - Directory fingerprinting + persistent state markers (idempotency)
    - File-system helpers (junk filtering, counting)
    - Shared constants (image extensions) and Rich console
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from rich.console import Console

console = Console()

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMG_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')   # for str.endswith checks
TABULAR_EXTS = {".csv", ".tsv"}

SKIP_NAMES = {"__macosx", "__pycache__", ".git", "venv", "env"}

MAX_FILES_PER_FINGERPRINT = 500_000


# ------------------------------------------------------------------
# File-system helpers
# ------------------------------------------------------------------

def is_junk(path: Path) -> bool:
    """Skip macOS resource forks, hidden files, and OS noise."""
    name = path.name
    if name.startswith("._"):
        return True
    if name.lower() in SKIP_NAMES:
        return True
    if path.is_dir() and name.lower() == "__macosx":
        return True
    return False


def count_files(folder: Path, exclude_zip: bool = True) -> int:
    """Count real files under *folder* (zips optionally excluded)."""
    if not folder.exists():
        return 0
    return sum(
        1 for f in folder.rglob("*")
        if f.is_file()
        and (not exclude_zip or f.suffix.lower() != ".zip")
        and not is_junk(f)
    )


def already_have_enough(folder: Path, min_files: int, force: bool = False) -> bool:
    """True when *folder* already holds at least *min_files* files."""
    if force:
        return False
    n = count_files(folder)
    if n >= min_files:
        print(f"\n[skip] {folder} already has {n} extracted file(s) "
              f"(>= {min_files}) — skipping download. Use --force to redo.")
        return True
    return False


# ------------------------------------------------------------------
# Fingerprints & state markers (idempotency)
# ------------------------------------------------------------------
#
# A fingerprint is a SHA-256 over each file's *relative path* and *size*
# (content hashing hundreds of thousands of images would be far too slow;
# path+size is a robust practical signal for dataset changes).

def dir_fingerprint(path) -> str | None:
    """Return a stable fingerprint for all files under *path*, or None if missing."""
    root = Path(path)
    if not root.exists():
        return None

    h = hashlib.sha256()
    count = 0
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        try:
            size = f.stat().st_size
        except OSError:
            size = -1
        h.update(f"{f.relative_to(root)}|{size}\n".encode("utf-8", "replace"))
        count += 1
        if count >= MAX_FILES_PER_FINGERPRINT:
            break

    if count == 0:
        return None
    h.update(f"#files={count}".encode())
    return h.hexdigest()


def _state_path(state_dir, key: str) -> Path:
    return Path(state_dir) / f"{key}.json"


def load_state(state_dir, key: str) -> dict | None:
    """Load a previously saved state marker, or None."""
    p = _state_path(state_dir, key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_state(state_dir, key: str, payload: dict) -> None:
    """Persist a state marker (creating the directory as needed)."""
    d = Path(state_dir)
    d.mkdir(parents=True, exist_ok=True)
    tmp = _state_path(d, key).with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, _state_path(d, key))


def inputs_up_to_date(state_dir, key: str, fingerprint: str | None,
                      required_paths=()) -> bool:
    """True when a saved fingerprint matches and every required artifact exists."""
    if not fingerprint:
        return False
    state = load_state(state_dir, key)
    if not state or state.get("fingerprint") != fingerprint:
        return False
    return all(Path(p).exists() for p in required_paths)
