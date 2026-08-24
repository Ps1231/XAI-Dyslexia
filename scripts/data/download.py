"""Network download helpers: direct HTTP and the Kaggle CLI."""
from __future__ import annotations

import subprocess
from pathlib import Path

import requests
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TaskProgressColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)


def download_file(url: str, dest_path: Path, headers=None) -> bool:
    """Stream *url* to *dest_path* with a rich progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[HTTP] Downloading {url} -> {dest_path}")
    try:
        r = requests.get(url, stream=True, timeout=120, headers=headers or {})
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task(dest_path.name, total=total or None)
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))
        size_mb = dest_path.stat().st_size / 1e6
        print(f"  Saved {dest_path} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ERROR downloading {url}: {e}")
        return False


def download_kaggle_dataset(kaggle_ref: str, dest_folder: Path):
    """Download + unzip a Kaggle dataset (requires `kaggle` CLI + credentials)."""
    dest_folder.mkdir(parents=True, exist_ok=True)
    print(f"\n[Kaggle] Downloading {kaggle_ref} -> {dest_folder}")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_ref,
             "-p", str(dest_folder), "--unzip"],
            check=True
        )
    except FileNotFoundError:
        print("  ERROR: kaggle CLI not found. Run: pip install kaggle")
    except subprocess.CalledProcessError as e:
        print(f"  ERROR downloading {kaggle_ref}: {e}")


def download_kaggle_dataset_nozip(kaggle_ref: str, dest_folder: Path):
    """Download a Kaggle dataset without auto-unzip (for password-protected archives)."""
    dest_folder.mkdir(parents=True, exist_ok=True)
    print(f"\n[Kaggle] Downloading {kaggle_ref} -> {dest_folder} (no auto-unzip)")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_ref,
             "-p", str(dest_folder)],
            check=True
        )
        for z in dest_folder.glob("*.zip"):
            print(f"  Downloaded archive: {z.name}")
    except FileNotFoundError:
        print("  ERROR: kaggle CLI not found. Run: pip install kaggle")
    except subprocess.CalledProcessError as e:
        print(f"  ERROR downloading {kaggle_ref}: {e}")
